# Authorship Anonymization through Differetial Privacy

This is the source code of paper: [ER-AE: Differentially Private Text Generation for Authorship Anonymization](https://arxiv.org/abs/1907.08736)

Requirements:
```
# Python 3.6
# Tensorflow 1.9.0 resource: https://pypi.tuna.tsinghua.edu.cn/simple/tensorflow/
$ pip3 install <path>\tensorflow-1.9.0-cp36-cp36-win_amd64.whl
$ pip3 install sklearn scipy nltk
```

In `model.py`, `get_ramdom_k_embed` function is to generate random k tokens for loss calculation, `embed_loss_fn` is to calculate embedding loss value, `predict_dp_two_sets` is to generate differentially private text through two-set exponential mechanism.

Sample traing data:

`my_dict.json`
``` 
{
    "i": 0,
    "you": 1
}

```

`id2w.txt`
``` 
i
you
```

`yelp.txt`
``` 
1 sample training sentence
```

Disclaimer: Part of the tensorflow autoencoder code refers to the repo: https://github.com/bohaohan/finch. It's originally folked from zhedongzheng/tensorflow-nlp but that part of the code has been deleted in the original repository. 