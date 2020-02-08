import random
import os
import itertools
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from game import GameState
from mcts import TreeSearch

class RandomModel:
    def evaluate(self, state):
        initial_turn = state.turn()
        while state.state() is GameState.Continue:
            move = random.choice(state.legal_moves())
            state.play(move)
        if state.state() is GameState.Tie:
            return 0
        elif initial_turn is state.winner():
            return -1
        else:
            return +1

class NeuralNetworkModel(tf.keras.Model):
    def __init__(self):
        super(NeuralNetworkModel, self).__init__()
        self.conv1 = Conv2D(30, 3, activation='relu', padding='same')
        self.conv2 = Conv2D(1, 3, activation='relu', padding='same')
        self.flatten = Flatten()
        self.d1 = Dense(32, activation='relu')
        self.d2 = Dense(1, activation='tanh')

    def call(self, state_tensor):
        x = self.conv1(state_tensor) # body

        policy = self.conv2(x) # policy head

        x = self.flatten(x) # value head
        x = self.d1(x)
        value = self.d2(x)
        return policy, value

    def evaluate(self, state):
        _, value = self.call(np.array([state.to_tensor()]))
        return value[0]

    def loss(self, estimated_policy, estimated_value,
                    true_policy, true_value):
        batch_size = estimated_policy.shape[0]
        mse = tf.keras.losses.MSE(true_value, estimated_value)
        ent = tf.nn.softmax_cross_entropy_with_logits(
                    tf.reshape(true_policy, (batch_size, -1)),
                    tf.reshape(estimated_policy, (batch_size, -1)))
        return mse + ent


def distil_amplify(initial_state, model, optimizer):
    data = []
    for i in range(200):
        data.extend(compare_models(model, model, initial_state))
    states, policies, values = zip(*data)
    with tf.GradientTape() as tape:
         estimated_policies, estimated_values = model(np.array(states))
         losses = model.loss(estimated_policies, estimated_values,
                            np.array(policies), np.array(values))
    print("loss:", np.mean(losses), "batch size:", len(data))
    grads = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def train(iteration, initial_state):
    model = NeuralNetworkModel()
    optimizer = tf.keras.optimizers.Adam()

    checkpoint_directory = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    for i in range(iteration):
        distil_amplify(initial_state, model, optimizer)

    checkpoint.save(file_prefix=checkpoint_prefix)

def compare_models(model1, model2, initial_state):
    state = deepcopy(initial_state)
    cycle = itertools.cycle([model1, model2])
    iterations = 50
    history = []
    while state.state() is GameState.Continue:
        ts = TreeSearch(state, next(cycle), 2)
        for i in range(iterations):
            ts.iterate()
        policy = ts.policy()
        history.append((state.to_tensor(),
            state.policy_to_tensor(policy)))
        p = dict(policy)
        next_move = max(p, key=p.get)
        state.play(next_move)

    values = itertools.cycle([0])

    if state.winner() is initial_state.turn():
        values = itertools.cycle([1, -1])
    elif state.winner():
        # if winner exists is not initialstate's turn
        values = itertools.cycle([-1, 1])

    history = [(x, p, next(values)) for (x, p) in history]
    return history
