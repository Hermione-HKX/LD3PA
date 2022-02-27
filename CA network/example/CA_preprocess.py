#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CA network preprocessing. Transforming the sentence and object information to fit network.

import numpy as np
import gensim
from nltk.tokenize import word_tokenize

# Open the pretrained word embedding model: glove-50d
model = gensim.models.KeyedVectors.load_word2vec_format \
    ('glove.6B.50d.bin', binary=False)


def build_inputs(cmd, env):
    """
    Base the command and object information to build network input.
    cmd: The list of commands.
    env: The list of objects in environment.
    """
    obj_n = 10  # Maximum number of possible objects in the environment is set to 10.
    n_c = 25  # The maximum length of command is set to 20.
    # ----------- command -------------
    cmd_emb_l = []
    for c in cmd:
        token = word_tokenize(c)
        cmd_emb = [[0.5] * 50]  # add Clt
        for t in token:
            cmd_emb.append(model[t])
        cmd_emb_l.append(position_embedding(np.asmatrix(cmd_emb)))
    # ----------- command -------------

    # ----------- environment -------------
    env_emb_l = []
    for e in env:
        env_emb = []
        for _ in e:
            env_emb.append(model[_])
        env_emb_l.append(np.asmatrix(env_emb))
    # ----------- environment -------------

    # ------------------ padding and mask ----------------------
    cmd_p, cmd_mask, env_p = [[] for _ in range(3)]

    for _ in cmd_emb_l:
        if _.shape[0] < n_c:
            p = n_c - _.shape[0]
            cmd_p.append(np.vstack((_, np.zeros((p, 50)))))
            d_mask = np.ones((n_c, n_c)) * -1e6
            d_mask[0:_.shape[0], 0:_.shape[0]] = 0
            d_mask[_.shape[0]:, _.shape[0]:] = 0
            cmd_mask.append(d_mask)
        else:
            d_mask = np.zeros((n_c, n_c))
            cmd_mask.append(d_mask)
            cmd_p.append(_)

    for _ in env_emb_l:
        if _.shape[0] < obj_n:
            p = obj_n - _.shape[0]
            env_p.append(np.vstack((_, np.zeros((p, 50)))))
        else:
            env_p.append(_)

    # ---------- list to input ----------
    cmd_p_i = np.reshape(cmd_p, (-1, n_c, 50))
    env_p_i = np.reshape(env_p, (-1, obj_n, 50))
    cmd_mask_i = np.reshape(cmd_mask, (-1, n_c, n_c))

    return cmd_p_i, env_p_i, cmd_mask_i


def position_embedding(inputs):
    """
    Add position embedding into sentences.
    """
    msg = np.array(inputs)
    pos_embedding = []
    seq_len = msg.shape[0]
    emb_dim = msg.shape[1]
    raw_emb = [1.0 / 100.0 ** (2.0 * value / emb_dim) for value in
               range(0, emb_dim)]
    a = 0
    while a < seq_len:
        pos = 0
        pos_emb = []
        while pos < emb_dim:
            if pos % 2 == 0:
                pos_emb.append(0.5 * np.sin(raw_emb[pos]))
                pos += 1
            elif pos % 2 != 0:
                pos_emb.append(0.5 * np.cos(raw_emb[pos]))
                pos += 1
                pass
        pos_embedding.append((msg[a] + pos_emb))
        a += 1
    output = np.asmatrix(pos_embedding)
    return output


if __name__ == '__main__':
    data = np.load('CA network/example/CA_example_data.npz',
                   allow_pickle=True)
    build_inputs(data['example_cmd'], data['example_env'])
