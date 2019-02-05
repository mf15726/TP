import random
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import os.path
from copy import deepcopy
import math
#with tf.Session() as sess:

def end_game(state):
    state_mod = deepcopy(state)
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] == 0:
                continue
            if i < 4:
                if state[i][j] == state[i+1][j] == state[i+2][j] == state[i+3][j]:
                    game_winner = state[i][j]
                    return game_winner
            if j < 3:
                if state[i][j] == state[i][j+1] == state[i][j+2] == state[i][j+3]:
                    game_winner = state[i][j]
                    return game_winner
            if i < 4 and j < 3:
                if state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3]:
                    game_winner = state[i][j]
                    return game_winner
                if state[i][j+3] == state[i+1][j+2] == state[i+2][j+1] == state[i+3][j]:
                    game_winner = state[i][j+3]
                    return game_winner
    else:
        game_winner = 0
        return game_winner


class Learned_Player(object):
    def __init__(self, player, alpha, epsilon, gamma):
        self.sess = tf.Session()
        self.player = player
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.state_index = []

        self.board_width = 7
        self.board_height = 6
        # Network Parameters
        self.n_input = 168
        self.n_nodes_1 = self.n_input * 2
        self.n_nodes_2 = self.n_input * 2
        self.n_nodes_3 = self.n_input * 2
        self.n_nodes_4 = self.n_input * 2
        self.n_classes = self.board_width

        self.input = tf.placeholder(tf.float32, [7,6])
#        self.available = tf.placeholder(tf.float32, [7])
        self.x_p1 = tf.cast(tf.equal(self.input, 1), tf.float32)
        self.x_p2 = tf.cast(tf.equal(self.input, 2), tf.float32)
        self.x_empty = tf.cast(tf.equal(self.input, 0), tf.float32)
        self.x_unavail = tf.cast(tf.equal(self.input, 3), tf.float32)
        self.x_bin = [self.x_empty,self.x_p1,self.x_p2,self.x_unavail]
#        self.x = tf.reshape(self.input, shape=[1,self.n_input])
        self.x = tf.reshape(self.x_bin, shape=[1,self.n_input])
        self.reward = tf.placeholder(tf.float32, [7])
        self.y = tf.reshape(self.reward, [1, self.n_classes])
        self.Q_val = self.neural_network()

        #cost
#        self.cost = tf.reduce_mean(tf.square(self.y - self.Q_val))
#        self.cost = tf.square(self.Q_val - self.y)
        self.cost = tf.square(self.y - self.Q_val)
        #optimiser

#        self.optimiser = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)
#        self.optimiser = tf.train.AdamOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)
        self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost)
#        self.optimizer = tf.train.AdograadOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)



#    def symmetry(self):
#        for i in range(len(self.input)):
#            for j in range(21):
#                if state[i][0:3] ==


    def neural_network(self):

        l1 = tf.layers.dense(
            inputs=self.x,
            units=self.n_input,
#            kernel_initializer = tf.constant_initializer(0,1),
            bias_initializer=tf.constant_initializer(0, 1),
            activation=tf.nn.leaky_relu,
#            activity_regularizer=tf.nn.softmax
        )

#        l2 = tf.layers.dense(
#            inputs=l1,
#            units=self.n_nodes_1,
#            bias_initializer=tf.constant_initializer(0, 1),
#            activation=tf.nn.leaky_relu,
#            activity_regularizer=tf.nn.softmax
#        )

#        l3 = tf.layers.dense(
#            inputs=l2,
#            units=self.n_nodes_2,
#            bias_initializer=tf.constant_initializer(0, 1),
#            activation=tf.nn.relu
#        )

#        l4 = tf.layers.dense(
#            inputs=l3,
#            units=self.n_nodes_3,
#            bias_initializer=tf.constant_initializer(0, 1),
#            activation=tf.nn.relu
#        )

#        l5 = tf.layers.dense(
#            inputs=l4,
#            units=self.n_nodes_4,
#            bias_initializer=tf.constant_initializer(0, 1),
#            activation=tf.nn.relu
#        )

        l_out = tf.layers.dense(
            inputs=l1,
            units=self.n_classes,
#            kernel_initializer = tf.constant_initializer(0,1),
            bias_initializer=tf.constant_initializer(0, 1),
            activation=tf.nn.leaky_relu,
#            activity_regularizer=tf.nn.softmax
        )

        l_norm = tf.contrib.layers.softmax(
        logits=l_out
        )

        return l_norm

    def free_space_finder(self, state):
        free_space = []
        temp_state = deepcopy(state)
        if len(free_space) == 7:
            return free_space
        for i in range(len(state)):
            skip = False
            if 0 in state[i]:
                ind = state[i].index(0)
                free_space.append((i,ind))
                for j in range(ind+1,6):
                    temp_state[i][j] = 3
                skip = True
                continue
            if skip == False:
                free_space.append((i,7))
        return free_space, temp_state

    def random(self,state):
       free_space, _ = self.free_space_finder(state)
       temp = random.randint(0, len(free_space) - 1)
       if 7 in free_space[temp]:
           while 7 in free_space[temp]:
               temp = random.randint(0, len(free_space) - 1)
       return free_space[temp]

    def action(self, state):
        rand = random.randint(1,100)
        move = None
        if rand <= 100*self.epsilon:
            move = self.random(state)
            return move
        else:
            free_space, input_state = self.free_space_finder(state)
            predictions = self.sess.run([self.Q_val], feed_dict={self.input: input_state})
#            print predictions
            opt_val = -float('Inf')
            for index, val in enumerate(predictions[0][0]):
                if 7 == free_space[index][1]:
                    continue
                if val > opt_val and free_space[index]:
                    opt_val = val
                    move = free_space[index]
            if move is None:
                print state
            self.state_index.append((deepcopy(state),move))
            return move

    def learn(self, winner):
        for i in range(len(self.state_index)):
            max_Q_list = []
            temp_state = self.state_index[i][0]
            free_space, input_state = self.free_space_finder(temp_state)
#            predictions = self.sess.run([self.Q_val], feed_dict={self.input: temp_state})
#            opt_val = -100
#            temp2 = deepcopy(predictions[0][0])
#            if end_game(temp_state) != 0:
#                max_Q_list = [0] * 7
            for space in free_space:
                opt_val = -float('Inf')
                if 7 == space[1]:
                    max_Q_list.append(-1)
                    continue
                winner2 = end_game(temp_state)
                if winner2 != 0:
                    if winner2 == self.player:
                        opt_val = 1
                    else:
                        opt_val = -1
                    continue
#                temp_state[space[0]][space[1]] = (self.player % 2) + 1
                input_state[space[0]][space[1]] = (self.player % 2) + 1
                predictions = self.sess.run([self.Q_val], feed_dict={self.input: input_state})
                for index, val in enumerate(predictions[0][0]):
                    if val > opt_val:
                        opt_val = val
#                        move = free_space[index]
#                temp_state[space[0]][space[1]] = 0
                input_state[space[0]][space[1]] = 0
                max_Q_list.append(self.gamma * opt_val)
#            if opt_val == -100:
#                opt_val = 0
            if winner == self.player:
                max_Q_list[self.state_index[i][1][0]] = 1 + max_Q_list[self.state_index[i][1][0]]
            if winner == (self.player % 2) + 1:
                max_Q_list[self.state_index[i][1][0]] = max_Q_list[self.state_index[i][1][0]] - 1
            _, __, cost = self.sess.run([self.Q_val, self.optimiser, self.cost], feed_dict={self.input: temp_state, self.reward: max_Q_list})
    #        print _
        self.state_index = []
        return 0
