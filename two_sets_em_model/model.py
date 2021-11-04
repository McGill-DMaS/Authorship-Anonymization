from __future__ import print_function
import os
from config import args
from utils_c import *
from modified_tf_classes import BasicDecoder, SampleEmbeddingHelperDPO
import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, params, train=True):
        self.params = params
        self.scopes = {
            'E': 'Encoder',
            'G': 'Generator'}

        self.build_placeholders()

        self.build_global_helpers()

        self.table = tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=args.id_path)

        self.build_word_embedding_layer()
        print("[1/3] Embedding layer Built")

        # initialize the base
        self.build_train_vae_graph()
        print("[2/3] VAE Graph Built")

        self.build_posterior_inference_graph()
        print("[3/3] Posterior Inference Graph Built")

        self.saver = tf.train.Saver()
        
        self.model_prefix = './saved_erae_model/'
        self.model_path = self.model_prefix + 'model.ckpt'

        if not os.path.exists(self.model_prefix):
            os.makedirs(self.model_prefix)

    def build_word_embedding_layer(self):
        if not args.use_bert:
            self.initializer, self.word_embedding, self.word2id_, self.id2word_, self.word_embed_ = load_embedding_layer(id2w=self.params['idx2word'])
        else:
            self.initializer, self.word_embedding, self.word2id_, self.id2word_, self.word_embed_ = load_partial_bert_layer(
                id2w=self.params['idx2word'])

    def build_train_vae_graph(self):

        self.temperature = self.temperature_fn()

        latent_vec = self.encoder(self.enc_inp)
        output1, output2, ids1 = self.generator(latent_vec)

        outputs = (output1, output2)
        self.train_vae_nll_loss = self.seq_loss_fn(*outputs)

        self.train_vae_nll_embed_loss = self.seq_embed_loss_fn(*outputs)

        loss_use_op = self.train_vae_nll_loss
        self.nll_loss = self.train_vae_nll_loss 
        self.nll_embed_loss = args.lambda_embed * self.train_vae_nll_embed_loss  \
                                 + args.lambda_n * self.nll_loss

        self.train_nll_embed = self.optimizer.apply_gradients(
            self.gradient_clipped(self.nll_embed_loss, scope=self.scopes['E'],
                                  scope2=self.scopes['G']))

        self.train_vae_use = self.optimizer.apply_gradients(
            self.gradient_clipped(loss_use_op, scope=self.scopes['E'],
                                  scope2=self.scopes['G']))

        self.incease_step = tf.assign_add(self.global_step, 1, name='increment')


    def build_posterior_inference_graph(self):
        latent_vec = self.encoder(self.enc_inp, reuse=True)
        self.post_sample_ids, self.post_sample_probs = self.generator(latent_vec, inference=True, sample=True, probs=True)


    def id2word(self, ids):
        values = self.table.lookup(ids)
        value_str = tf.reduce_join(values, separator=' ', axis=-1)
        return value_str

    def encoder(self, inputs, reuse=None, gumbel=False):
        with tf.variable_scope(self.scopes['E'], reuse=reuse):

            if not gumbel:
                x = tf.nn.embedding_lookup(self.word_embedding, inputs)
            else:
                x = tf.matmul(tf.reshape(inputs, [-1, self.params['vocab_size']]), self.word_embedding)
                x = tf.reshape(x, [self.batch_size, args.max_len, args.embedding_dim])

            _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.rnn_cell_single(args.rnn_size // 2, reuse=reuse, name='EnFw'),
                cell_bw=self.rnn_cell_single(args.rnn_size // 2, reuse=reuse, name='EnBw'),
                inputs=x,
                sequence_length=self.enc_seq_len,
                dtype=tf.float32)

            birnn_state = tf.concat((state_fw, state_bw), -1)

            z = tf.layers.dense(birnn_state, args.latent_size, reuse=reuse)
            return z

    def generator(self, latent_vec, reuse=None,
                  inference=False, sample=False, probs=False):
        embedding = self.word_embedding
        if not inference:
            with tf.variable_scope(self.scopes['G'], reuse=reuse):
                init_state = tf.layers.dense(latent_vec, args.rnn_size, tf.nn.elu, reuse=reuse, name='ini_state')

                lin_proj = tf.layers.Dense(self.params['vocab_size'], _scope='decoder/dense', _reuse=reuse,
                                           name='linear')

                helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=tf.nn.embedding_lookup(embedding, self.dec_inp),
                    sequence_length=self.dec_seq_len)

                self.g_rnn_cell = self.rnn_cell(reuse=reuse, num_layer=args.num_layer)
                decoder = BasicDecoder(
                    cell=self.g_rnn_cell,
                    helper=helper,
                    initial_state=tuple([init_state] * args.num_layer),
                    concat_z=latent_vec,
                )
                decoder_output, _, _len = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    impute_finished=True)

                return decoder_output.rnn_output, lin_proj.apply(decoder_output.rnn_output), decoder_output.sample_id
        else:
            reuse = True

            with tf.variable_scope(self.scopes['G'], reuse=reuse):

                cell = self.rnn_cell(reuse=reuse, num_layer=args.num_layer)

                init_state = tf.layers.dense(latent_vec, args.rnn_size, tf.nn.elu, reuse=reuse, name='ini_state')
                latent_vec_batch = tf.contrib.seq2seq.tile_batch(
                    latent_vec, multiplier=1)

                sampleHelper = SampleEmbeddingHelperDPO(
                    inputs=tf.nn.embedding_lookup(embedding, self.dec_inp),
                    sequence_length=self.dec_seq_len,
                    embedding=embedding,
                    start_tokens=tf.tile(tf.constant([self.params['<start>']], dtype=tf.int32), [self.batch_size]),
                    end_token=self.params['<end>'],
                )
                
                decoder_initial_state = tuple([init_state] * args.num_layer)
                decoder = BasicDecoder(
                    cell=cell,
                    helper=sampleHelper,
                    initial_state=decoder_initial_state,
                    output_layer=tf.layers.Dense(self.params['vocab_size'], _reuse=True),
                    concat_z=latent_vec_batch,
                )

                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    maximum_iterations=2 * tf.reduce_max(self.enc_seq_len))
                if probs:
                    return decoder_output.sample_id, decoder_output.rnn_output
                return decoder_output.sample_id

    def tf_pad_output(self, output):
        shape = tf.shape(output)
        if shape[1] == args.max_len + 1:
            return output
        else:
            paddings = [[0, 0], [0, args.max_len + 1 - shape[1]], [0, 0]]
            return tf.pad(output, paddings)

    def build_placeholders(self):
        self.enc_inp = tf.placeholder(tf.int32, [None, args.max_len])
        self.dec_inp = tf.placeholder(tf.int32, [None, args.max_len + 1])
        self.dec_out = tf.placeholder(tf.int32, [None, args.max_len + 1])
        self.labels = tf.placeholder(tf.int64, [None])
        self.ids = tf.placeholder(tf.int32, [None, args.max_len + 1])

    def build_global_helpers(self):

        self.batch_size = tf.shape(self.enc_inp)[0]
        self.enc_seq_len = tf.count_nonzero(self.enc_inp, 1, dtype=tf.int32)
        self.dec_seq_len = self.enc_seq_len + (args.max_len - self.enc_seq_len) + 1
        self.global_step = tf.Variable(0, trainable=False)
        self.nll_len = tf.reduce_max(self.dec_seq_len)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.gaussian = tf.truncated_normal([self.batch_size, args.latent_size])

    def gradient_clipped(self, loss_op, scope=None, scope2=None):
        params = tf.trainable_variables(scope=scope)
        if scope2 is not None:
            params += tf.trainable_variables(scope=scope2)
        gradients = tf.gradients(loss_op, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.clip_norm)
        return zip(clipped_gradients, params)

    def seq_loss_fn(self, training_rnn_out, training_logits):
        
        mask = tf.sequence_mask(
            self.dec_seq_len, self.nll_len, dtype=tf.float32)
        return tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
                logits=training_logits,
                targets=self.dec_out,
                weights=mask,
                average_across_timesteps=False,
                average_across_batch=True))

    def embed_loss_fn(self, indices, values, k=5):
        embedding = self.word_embedding
        mask = tf.sequence_mask(
            self.dec_seq_len, self.nll_len, dtype=tf.float32)
        # get embedding
        k_embed = tf.nn.embedding_lookup(embedding, indices)
        gt_embed = tf.nn.embedding_lookup(embedding, self.dec_out)

        # tile label
        gt_embed = tf.expand_dims(gt_embed, -2)
        gt_embed_k = tf.tile(gt_embed, (1, 1, k, 1))

        # norm vector
        k_embed = tf.nn.l2_normalize(k_embed, dim=-1)
        gt_embed_k = tf.nn.l2_normalize(gt_embed_k, dim=-1)

        # cosine sim
        sim = tf.reduce_sum(tf.multiply(k_embed, gt_embed_k), axis=-1)

        values = tf.clip_by_value(values, 0.00001, 0.85)
        sim = tf.clip_by_value(sim, 0.00001, 0.85)

        # mean of k token
        loss = tf.reduce_mean(tf.multiply(sim, tf.log(values)), axis=-1)

        # average loss per sentence
        loss = tf.reduce_mean(loss, axis=0)

        # average loss per token
        loss = tf.reduce_mean(tf.multiply(loss, mask))
        return loss

    def get_ramdom_k_embed(self, inputs, k):
        """
        return random k indices with the shape of (batch, sentence length, k) and it's probability.
        """
        bt = tf.shape(inputs)[0]  # batch size
        le = tf.shape(inputs)[1]  # sentence length
        vc = tf.shape(inputs)[2]  # vocab size

        indice = tf.random_uniform([bt * le, k], dtype=tf.int32, maxval=vc)
        indice = tf.reshape(
            indice, [-1, 1]
        )

        bat_ind = tf.reshape(
            tf.range(bt), [-1, 1])
        bat_ind = tf.reshape(
            tf.tile(bat_ind, [1, le * k]), [-1, 1]
        )
        le_ind = tf.reshape(
            tf.range(le), [-1, 1]
        )
        le_ind = tf.reshape(
            tf.tile(le_ind, [1, bt * k]), [-1, 1]
        )

        bat_le_ind = tf.concat([bat_ind, le_ind], axis=1)

        ind = tf.concat([bat_le_ind, indice], -1)
        vals = tf.gather_nd(inputs, ind)

        return tf.reshape(vals, [bt, le, k]), tf.reshape(indice, [bt, le, k])

    def seq_embed_loss_fn(self, training_rnn_out, training_logits, k = 5):

        training_logits = tf.nn.softmax(training_logits)
        values, indices = tf.nn.top_k(training_logits, k=k)
        values_r, indices_r = self.get_ramdom_k_embed(training_logits, k=k)
        loss = self.embed_loss_fn(indices, values, k=k)
        loss_r = self.embed_loss_fn(indices_r, values_r, k=k)

        return -(loss + loss_r)

    def temperature_fn(self):
        return args.temperature_anneal_max * inverse_sigmoid((10 / args.temperature_anneal_bias) * (
                tf.to_float(self.global_step - args.temperature_start_step) - tf.constant(
                args.temperature_anneal_bias / 2)))

    def rnn_cell_single(self, rnn_size=None, reuse=False, name=""):
        rnn_size = args.rnn_size if rnn_size is None else rnn_size
        name = "GRU"
        return tf.nn.rnn_cell.GRUCell(rnn_size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse, name=name)

    def rnn_cell(self, rnn_size=None, reuse=None, num_layer=1):
        cells = [self.rnn_cell_single(rnn_size, reuse=reuse) for _ in range(num_layer)]
        return tf.contrib.rnn.MultiRNNCell(cells)

    def get_new_w(self, w):
        if w in self.params['word2idx']:
            idx = self.params['word2idx'][w]
        else:
            idx = self.params['word2idx']['<unk>']
        return idx if idx < self.params['vocab_size'] else self.params['word2idx']['<unk>']
    
    def train_vae_session(self, sess, enc_inp, dec_inp, dec_out, labels):

        _, nll_loss, step = sess.run(
            [self.train_vae_use, self.train_vae_nll_loss, self.global_step],
            {self.enc_inp: enc_inp, self.dec_inp: dec_inp, self.dec_out: dec_out, self.labels: labels})
        return {'nll_loss': nll_loss, 'step': step}

    def train_vae_embed_nll_session(self, sess, enc_inp, dec_inp, dec_out, labels):

        _, embed_loss, nll_loss, step= sess.run(
            [self.train_nll_embed, self.train_vae_nll_embed_loss, self.train_vae_nll_loss, self.global_step],
            {self.enc_inp: enc_inp, self.dec_inp: dec_inp, self.dec_out: dec_out, self.labels: labels})
        return {'embed_loss': embed_loss, 'step': step}

    def increase_step_session(self, sess):
        sess.run(self.incease_step)

    def pad(self, ids):
        result = []
        for i in range(len(ids)):
            temp = ids[i]
            if len(temp) < args.max_len + 1:
                temp = np.concatenate((temp, [self.params['word2idx']['<pad>'], ] * (args.max_len + 1 - len(temp))))
            else:
                temp = temp[:args.max_len + 1]
            result.append(temp)
        return np.array(result, dtype=np.int32)


    def exponential_mechanism(self, output):
        output = tf.nn.softmax(output)
        return output * args.epsilon / (2 * args.delta)

    def get_input_length(self, input):
        start = len(input) - 1
        while input[start] == self.params['word2idx']['<pad>']:
            start -= 1
        return start

    def ids2sentences(self, s):
        res = []
        for ids in s:
            temp = []
            for id in ids:
                temp.append(self.params['idx2word'][id])
            res.append(' '.join(res).replace('<pad>', ''))
        return res

    def predict_dp_two_sets(self, sess, sentence, label):
        import nltk
        sentence_ = []
        for w in nltk.tokenize.word_tokenize(sentence):
                sentence_.append(self.get_new_w(w))
        sentence = sentence_[:args.max_len]

        sentence = sentence + [self.params['word2idx']['<pad>']] * (args.max_len - len(sentence))

        sentence_dein = [self.params['word2idx']['<start>']] + sentence

        predicted_ids_sample, predicted_probs_sample = sess.run(
            [self.post_sample_ids, self.post_sample_probs],
            {self.enc_inp: np.atleast_2d(sentence),
             self.dec_inp: np.atleast_2d(sentence_dein),
             self.labels: np.atleast_1d(label)})

        res = []

        predicted_probs = predicted_probs_sample[0]

        def build_two_sets(probs, k=5):
            # return lexical set and semantic set
            probs = np.array(probs)
            l_set = np.random.choice(probs.shape[0], k, p=probs, replace=True)
            l_set_probs = probs[l_set]

            marks = np.ones(probs.shape[0], dtype=np.bool)
            marks[l_set] = False

            whole_idxs = np.arange(probs.shape[0])
            s_set = whole_idxs[marks]
            s_set_probs = probs[marks]

            return l_set, s_set, l_set_probs, s_set_probs

        def choose_set(l_set, s_set, l_set_probs, s_set_probs, eps=80):
            probs = [0, 0]
            probs[0] = np.sum(l_set_probs) / (np.sum(l_set_probs) + np.sum(s_set_probs))
            probs[1] = 1 - probs[0]
            probs = exponential_mechanism(probs, eps, 1)
            po = [(l_set, l_set_probs), (s_set, s_set_probs)]
            indxs = [0, 1]
            indx = int(np.random.choice(indxs, 1, p=probs))
            return po[indx]

        for probs in predicted_probs:
            probs = softmax(probs, -1)

            # build set
            l_set, s_set, l_set_probs, s_set_probs = build_two_sets(probs, k=5)

            # choose set
            c_set, c_set_probs = choose_set(l_set, s_set, l_set_probs, s_set_probs, eps=args.epsilon)

            # choose token
            token_eps = 0.1
            c_set_probs = exponential_mechanism(c_set_probs, token_eps, 1)
            token_idx = int(np.random.choice(c_set, 1, p=c_set_probs))

            if token_idx == self.params['word2idx']['<end>']:
                break
            res.append(token_idx)
        o_pred = ' '.join([self.params['idx2word'][idx] for idx in res])
        return o_pred, o_pred

    def save_model(self, sess, step=None):
        self.saver.save(sess, self.model_path, global_step=step)

    def load_model(self, sess, logging):
        ckpt = tf.train.get_checkpoint_state(self.model_prefix)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.model_prefix, ckpt_name)
            self.saver.restore(sess, fname)
            message = "Checkpoint successfully loaded from {}".format(fname)
            logging.info(message)
            return True
        else:
            message = "Checkpoint could not be loaded from {}".format(self.model_prefix)
            logging.info(message)
            return False
