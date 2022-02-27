#!/usr/bin/env python
# -*- coding: utf-8 -*-

from CA_preprocess import build_inputs
from keras.layers import *
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
import numpy as np


class Attention(Layer):
    """Multi-head Self Attention Layer for encoding command"""

    def __init__(self, **kwargs):
        self.multiheads = 5
        self.head_dim = 10
        self.output_dim = 50  # self.multiheads * head_dim
        self.batch_size = 1
        self.emb_dim = 50
        self.len = 25  # the maximum length of command
        super(Attention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """network output shape"""
        return (-1, self.len, self.output_dim)

    def build(self, input_shape):
        """build trainable matrix"""
        self.WQ = self.add_weight(name='WQ',
                                  shape=(self.emb_dim, self.emb_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(self.emb_dim, self.emb_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(self.emb_dim, self.emb_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, XY):
        X, Y = XY
        Q_seq = K.dot(X, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, self.len, self.multiheads, self.head_dim))
        Q_seq = K.permute_dimensions(Q_seq, pattern=(0, 2, 1, 3))
        K_seq = K.dot(X, self.WK)
        K_seq = K.reshape(K_seq, (-1, self.len, self.multiheads, self.head_dim))
        K_seq = K.permute_dimensions(K_seq, pattern=(0, 2, 1, 3))

        V_seq = K.dot(X, self.WV)
        V_seq = K.reshape(V_seq, (-1, self.len, self.multiheads, self.head_dim))
        V_seq = K.permute_dimensions(V_seq, pattern=(0, 2, 1, 3))

        A = tf.einsum('bhjd,bhkd->bhjk', Q_seq, K_seq) / self.head_dim ** 0.5

        padding = K.reshape(Y, (-1, 1, self.len, self.len))
        padding = tf.tile(padding, [1, self.multiheads, 1, 1])
        A = Add()([A, padding])
        A = K.softmax(A)
        O_seq = tf.einsum('bhqk,bhkd->bhqd', A, V_seq)
        O_seq = K.permute_dimensions(O_seq, pattern=(0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, self.len, self.output_dim))
        return O_seq


def predict(data):
    """
    load model and predict
    data: example data
    """
    model = model_from_json(open('CA network/model/CA_network.json').read(),
                            custom_objects={'Attention': Attention, 'tf': tf})
    model.load_weights('CA network/model/CA_network_weight.h5')

    cmd_p_i, env_p_i, cmd_mask_i = build_inputs(data['example_cmd'], data['example_env'])

    print('predict:\n', model.predict(x={'input_cmd': cmd_p_i,
                                         'input_env': env_p_i,
                                         'input_cmd_m': cmd_mask_i}, batch_size=1))

    print('label answer:\n', data['example_rel_label'], '\n', data['example_role_label'])


if __name__ == '__main__':
    """
    Build network input and predict perspective.
    The maximum object number in the environment is 10.
    Some examples are provided. You can choose them to test this code. 
    Or you can build you own input.
    """
    example_data = np.load('CA network/example/CA_example_data.npz',
                           allow_pickle=True)
    predict(example_data)
