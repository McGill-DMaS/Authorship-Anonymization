from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
import tensorflow as tf
from tensorflow.python.ops.distributions import categorical

_transpose_batch_time = decoder._transpose_batch_time  # pylint: disable=protected-access



def _call_sampler(sample_n_fn, sample_shape, name=None):
  """Reshapes vector of samples."""
  with ops.name_scope(name, "call_sampler", values=[sample_shape]):
    sample_shape = ops.convert_to_tensor(
        sample_shape, dtype=dtypes.int32, name="sample_shape")
    # Ensure sample_shape is a vector (vs just a scalar).
    pad = math_ops.cast(math_ops.equal(array_ops.rank(sample_shape), 0),
                        dtypes.int32)
    sample_shape = array_ops.reshape(
        sample_shape,
        array_ops.pad(array_ops.shape(sample_shape),
                      paddings=[[pad, 0]],
                      constant_values=1))
    samples = sample_n_fn(math_ops.reduce_prod(sample_shape))
    batch_event_shape = array_ops.shape(samples)[1:]
    final_shape = array_ops.concat([sample_shape, batch_event_shape], 0)
    return array_ops.reshape(samples, final_shape)


def categorical_sample(logits, dtype=dtypes.int32,
                       sample_shape=(), seed=None):
  """Samples from categorical distribution."""
  logits = ops.convert_to_tensor(logits, name="logits")
  event_size = array_ops.shape(logits)[-1]
  batch_shape_tensor = array_ops.shape(logits)[:-1]
  def _sample_n(n):
    """Sample vector of categoricals."""
    if logits.shape.ndims == 2:
      logits_2d = logits
    else:
      logits_2d = array_ops.reshape(logits, [-1, event_size])
    sample_dtype = dtypes.int64 if logits.dtype.size > 4 else dtypes.int32
    draws = random_ops.multinomial(
        logits_2d, n, seed=seed, output_dtype=sample_dtype)
    draws = array_ops.reshape(
        array_ops.transpose(draws),
        array_ops.concat([[n], batch_shape_tensor], 0))
    return math_ops.cast(draws, dtype)
  return _call_sampler(_sample_n, sample_shape)


class SampleEmbeddingHelper(GreedyEmbeddingHelper):
  """A helper for use during inference.
  Uses sampling (from a distribution) instead of argmax and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, start_tokens, end_token,
               softmax_temperature=None, seed=None):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      softmax_temperature: (Optional) `float32` scalar, value to divide the
        logits by before computing the softmax. Larger values (above 1.0) result
        in more random samples, while smaller values push the sampling
        distribution towards the argmax. Must be strictly greater than 0.
        Defaults to 1.0.
      seed: (Optional) The sampling seed.
    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    super(SampleEmbeddingHelper, self).__init__(
        embedding, start_tokens, end_token)
    self._softmax_temperature = softmax_temperature
    self._seed = seed

  def sample(self, time, outputs, state, name=None):
    """sample for SampleEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, we sample instead of argmax (greedy).
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))

    outputs = tf.nn.softmax(outputs)

    if self._softmax_temperature is None:
      logits = outputs
    else:
      logits = outputs / self._softmax_temperature

    sample_ids = categorical_sample(logits=logits, seed=self._seed)

    return sample_ids
    

def _unstack_ta(inp):
  return tensor_array_ops.TensorArray(
      dtype=inp.dtype, size=array_ops.shape(inp)[0],
      element_shape=inp.get_shape()[1:]).unstack(inp)


class SampleEmbeddingHelperDPO(GreedyEmbeddingHelper):
  """A helper for use during inference.
  Uses sampling (from a distribution) instead of argmax and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, inputs, sequence_length, embedding, start_tokens, end_token,
               softmax_temperature=None, seed=None):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      softmax_temperature: (Optional) `float32` scalar, value to divide the
        logits by before computing the softmax. Larger values (above 1.0) result
        in more random samples, while smaller values push the sampling
        distribution towards the argmax. Must be strictly greater than 0.
        Defaults to 1.0.
      seed: (Optional) The sampling seed.
    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    super(SampleEmbeddingHelperDPO, self).__init__(
        embedding, start_tokens, end_token)
    self._softmax_temperature = softmax_temperature
    self._seed = seed
    self._embedding = embedding

    inputs = ops.convert_to_tensor(inputs, name="inputs")
    self._inputs = inputs
    inputs = nest.map_structure(_transpose_batch_time, inputs)

    self._input_tas = nest.map_structure(_unstack_ta, inputs)
    self._sequence_length = ops.convert_to_tensor(
      sequence_length, name="sequence_length")
    if self._sequence_length.get_shape().ndims != 1:
      raise ValueError(
        "Expected sequence_length to be a vector, but received shape: %s" %
        self._sequence_length.get_shape())

    self._zero_inputs = nest.map_structure(
      lambda inp: array_ops.zeros_like(inp[0, :]), inputs)

    self._batch_size = array_ops.size(sequence_length)

  def sample(self, time, outputs, state, name=None):
    """sample for SampleEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, we sample instead of argmax (greedy).
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))



    logits = outputs

    if self._softmax_temperature is None:
      logits = logits
    else:
      logits = logits / self._softmax_temperature

    ## try
    probs = tf.nn.softmax(logits)
    sample_id_sampler = categorical.Categorical(probs=probs)
    sample_ids = sample_id_sampler.sample(seed=self._seed)
    return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """next_inputs_fn for GreedyEmbeddingHelper."""

    next_time = time + 1
    finished = (next_time >= self._sequence_length)
    all_finished = math_ops.reduce_all(finished)

    def read_from_ta(inp):
      return inp.read(next_time)

    next_inputs = control_flow_ops.cond(
      all_finished, lambda: self._zero_inputs,
      lambda: nest.map_structure(read_from_ta, self._input_tas))
    return (finished, next_inputs, state)
