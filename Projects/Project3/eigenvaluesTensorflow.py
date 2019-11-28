import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Q = np.random.rand(3, 3)

A = tf.constant(((Q.T + Q) / 2))
A_debug = [[3, 2, 4], [2, 0, 2], [4, 2, 3]]
# lmda_debug = [1, 8, 1]
# eigvec_debug = [[-0.74, 0.667, -0.21], [0.29, 0.333, -0.77], [0.59, 0.]]


# T = 11
# x = np.zeros((A.shape[0], T))
# x[0][0] = 1


# print(x[:, 0])

tf.keras.backend.set_floatx("float64")


class DNModel(tf.keras.Model):
    """docstring for ."""

    def __init__(self):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(60, activation=tf.nn.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(30, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)

        return self.out(x)


@tf.function
def g_analytic(x, t):
    return None


@tf.function
def RHS(model, A, x, t):
    TS = trial_solution(model, x, t)
    return (tf.linalg.matmul(tf.linalg.matmul(TS.T, TS), tf.linalg.matmul(A, TS))
            - tf.linalg.matmul(tf.linalg.matmul(tf.linalg.matmul(TS.T, A), TS), TS))


@tf.function
def I(x):

    return None


@tf.function
def trial_solution(model, x, t):
    return (1 - t) * x + t * model(x)


# Define loss function
@tf.function
def loss(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)

        trial = trial_solution(model, x, t)

    d_trial_dt = tape.gradient(trial, t)

    del tape

    return tf.losses.MSE(
        tf.zeros_like(d_trial_dt), d_trial_dt - RHS(model, A, x, t))

# Define gradient method
@tf.function
def grad(model, x, t):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, t)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Initial model and optimizer
model = DNModel()
optimizer = tf.keras.optimizers.Adam(0.01)


# Run training loop
num_epochs = 1000

num_points = 11

start = tf.constant(0, dtype=tf.float64)
stop = tf.constant(1, dtype=tf.float64)
T = tf.linspace(start, stop, num_points)

x = [1, 0, 0]
x = tf.constant(x, dtype=tf.float64)

for t in T:

    for epoch in range(num_epochs):
        # Apply gradients in optimizer
        cost, gradients = grad(model, x, t)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Output loss improvement
        print(
            f"Step: {optimizer.iterations.numpy()}, "
            + f"Loss: {tf.math.reduce_mean(cost.numpy())}"
        )


g = tf.reshape(g_analytic(x, t), (num_points, num_points))
g_nn = tf.reshape(trial_solution(model, x, t), (num_points, num_points))

diff = tf.abs(g - g_nn)
print(f"Max diff: {tf.reduce_max(diff)}")
print(f"Mean diff: {tf.reduce_mean(diff)}")
