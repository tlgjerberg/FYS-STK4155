import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.keras.backend.set_floatx("float64")


num_points = 11
start = tf.constant(0, dtype=tf.float64)
stop = tf.constant(1, dtype=tf.float64)
stop_t = stop

X, T = tf.meshgrid(tf.linspace(start, stop, num_points),
                   tf.linspace(start, stop_t, num_points))

x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])


class DNModel(tf.keras.Model):
    """docstring for ."""

    def __init__(self):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(20, activation=tf.nn.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
    x = self.dense_1(inputs)

    return self.out(x)


def RHS(x):
    return 0


def trial_solution():
    return
