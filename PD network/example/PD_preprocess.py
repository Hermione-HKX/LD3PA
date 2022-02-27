#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PD network preprocessing. Transforming the sentence and position information to fit network.

import numpy as np
import gensim
import nltk
import copy
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Open the pretrained word embedding model: glove-50d
model = gensim.models.KeyedVectors.load_word2vec_format \
    ('glove.6B.50d.bin', binary=False)


def build_inputs(obj_n, desc, obj_p, env, XY):
    """
    Base the environment description, environment and object positions to build network input.
    obj_n: Maximum number of possible objects in the environment.
    desc: The list of description of the environment.
    obj_p: The list of objects' positions in environment.
    env: The list of objects in environment.
    XY: The maximum length and width of the working area. XY = [X, Y]
    And this network use 50-d word embedding so k is 50.
    """
    n_d = 20  # The maximum length of description is set to 20.
    n_p = 50 - n_d  # n_p is k-n_d
    # ----------- description -------------
    desc_emb_l = []
    for d in desc:
        token = word_tokenize(d)
        desc_emb = [[0.5] * 50]  # add Clt
        for t in token:
            desc_emb.append(model[t])
        desc_emb_l.append(position_embedding(np.asmatrix(desc_emb)))
    # ----------- description -------------

    # ----------- environment -------------
    env_emb_l = []
    for e in env:
        env_emb = []
        for _ in e:
            env_emb.append(model[_])
        env_emb_l.append(np.asmatrix(env_emb))
    # ----------- environment -------------

    # ----------- object position -------------
    """
    Change position to a vector.
    There are lots of ways to build the positional embedding, this code use the way provide in the paper.
    The conversion process can be thought of as a sample of a sin function. 
    The density of sampling points in a sin cycle will affect the result of the conversion. Therefore, the X can be
    multiplied by a coefficient, which is taken here as 20, and can also be modified according to the change of 
    parameters k, n_p and so on.
    """
    obj_pos_l = []
    for pos in obj_p:
        pos_emb_one = []
        for p in pos:
            x, y = p[0], p[1]
            raw_emb = [(np.sin(4 * 3.14 * value * x / (XY[0] * 20))) for value in range(0, int(n_p / 2))] + \
                      [(np.sin(4 * 3.14 * value * y / (XY[1] * 20))) for value in range(0, int(n_p / 2))]
            pos_emb_one.append(raw_emb)
        mat_pos = np.asmatrix(pos_emb_one)
        obj_pos_l.append(mat_pos)
    # ----------- object position -------------

    # ------------------ padding and mask ----------------------
    desc_p, desc_mask, pos_p, env_p, mix_mask = [[] for _ in range(5)]

    for _ in desc_emb_l:
        if _.shape[0] < n_d:
            p = n_d - _.shape[0]
            desc_p.append(np.vstack((_, np.zeros((p, 50)))))
            d_mask = np.ones((n_d, n_d)) * -1e6
            d_mask[0:_.shape[0], 0:_.shape[0]] = 0
            d_mask[_.shape[0]:, _.shape[0]:] = 0
            desc_mask.append(d_mask)
        else:
            d_mask = np.zeros((n_d, n_d))
            desc_mask.append(d_mask)
            desc_p.append(_)

    # feature MIX and obj_pos padding and mask
    for _ in obj_pos_l:
        if _.shape[0] < obj_n:
            p = obj_n - _.shape[0]

            pos_p.append(np.vstack((_, np.zeros((p, n_p)))))
            m_mask = np.ones((obj_n + 1, obj_n + 1)) * -1e6
            m_mask[0:_.shape[0] + 1, 0:_.shape[0] + 1] = 0
            m_mask[_.shape[0] + 1:, _.shape[0] + 1:] = 0
            mix_mask.append(m_mask)
        else:
            # -----------
            pos_p.append(_)
            m_mask = np.zeros((obj_n + 1, obj_n + 1))
            mix_mask.append(m_mask)

    for _ in env_emb_l:
        if _.shape[0] < obj_n:
            p = obj_n - _.shape[0]
            env_p.append(np.vstack((_, np.zeros((p, 50)))))
        else:
            env_p.append(_)

    # ---------- list to input ----------
    desc_p_i = np.reshape(desc_p, (-1, n_d, 50))
    env_p_i = np.reshape(env_p, (-1, obj_n, 50))
    pos_p_i = np.reshape(pos_p, (-1, obj_n, n_p))
    desc_mask_i = np.reshape(desc_mask, (-1, n_d, n_d))
    mix_mask_i = np.reshape(mix_mask, (-1, obj_n + 1, obj_n + 1))

    return desc_p_i, env_p_i, pos_p_i, desc_mask_i, mix_mask_i


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
    """obj_n, desc, obj_p, env, XY"""
    obj_num = 10
    example_data = np.load('PD network/example/example_data/example_'
                           + str(obj_num) + '.npz', allow_pickle=True)
    example_desc, example_env, example_pos, example_label = \
        example_data['example_desc'], example_data['example_env'], example_data['example_pos'], example_data['example_label']

    build_inputs(obj_num, example_desc, example_pos, example_env, [80, 80])
