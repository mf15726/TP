import random
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import os.path
from copy import deepcopy
import math
#with tf.Session() as sess:

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
        self.n_nodes_1 = 256
        self.n_nodes_2 = 256
        self.n_nodes_3 = 256
        self.n_nodes_4 = 256
        self.n_input = 126
        self.n_classes = self.board_width

        self.input = tf.placeholder(tf.float32, [7,6])
#        self.available = tf.placeholder(tf.float32, [7])
        self.x_p1 = tf.cast(tf.equal(self.input, 1), tf.float32)
        self.x_p2 = tf.cast(tf.equal(self.input, 2), tf.float32)
        self.x_empty = tf.cast(tf.equal(self.input, 0), tf.float32)
#        self.x_avail = tf.cond(self.input in self.available)
        self.temp = [self.x_empty,self.x_p1,self.x_p2]
#        self.x = tf.reshape(self.input, shape=[1,self.n_input])
        self.x = tf.reshape(self.temp, shape=[1,self.n_input])
        self.reward = tf.placeholder(tf.float32, [7])
        self.y = tf.reshape(self.reward, [1, self.n_classes])
        self.Q_cal = tf.placeholder(tf.float32, [self.n_classes])
        self.Q = tf.reshape(self.Q_cal, [1, self.n_classes])
        self.Q_val = self.neural_network()

        #cost
#        self.cost = tf.reduce_mean(tf.square(self.y - self.Q_val))
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

        l2 = tf.layers.dense(
            inputs=l1,
            units=self.n_nodes_1,
            bias_initializer=tf.constant_initializer(0, 1),
            activation=tf.nn.leaky_relu,
#            activity_regularizer=tf.nn.softmax
        )

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
            inputs=l2,
            units=self.n_classes,
#            kernel_initializer = tf.constant_initializer(0,1),
            bias_initializer=tf.constant_initializer(0, 1),
            activation=tf.nn.leaky_relu,
#            activity_regularizer=tf.nn.softmax
        )

        return l_out

    def free_space_finder(self, state):
        free_space = []
        if len(free_space) == 7:
            return free_space
        for i in range(len(state)):
            skip = False
            if 0 in state[i]:
                free_space.append((i,state[i].index(0)))
                skip = True
                continue
            if skip == False:
                free_space.append((i,7))
        return free_space

    def random(self,state):
       free_space = self.free_space_finder(state)
       temp = random.randint(0, len(free_space) - 1)
       if 7 in free_space[temp]:
           while 7 in free_space[temp]:
               temp = random.randint(0, len(free_space) - 1)
       return free_space[temp]



    def action(self, state):
        rand = random.randint(1,100)
        if rand <= 100*self.epsilon:
            move = self.random(state)
            return move
        else:
            free_space = self.free_space_finder(state)
            predictions = self.sess.run([self.Q_val], feed_dict={self.input: state})
#            print predictions
            opt_val = -10000000000000000000000000000000000000000000000000
            for index, val in enumerate(predictions[0][0]):
                if 7 == free_space[index][1]:
                    continue
                if val > opt_val and free_space[index]:
                    opt_val = val
                    move = free_space[index]
#            print opt_val
            self.state_index.append((deepcopy(state),move))
            return move

    def learn(self, winner):
        for i in range(len(self.state_index)):
            temp_state = self.state_index[i][0]
            free_space = self.free_space_finder(temp_state)
            predictions = self.sess.run([self.Q_val], feed_dict={self.input: temp_state})
            opt_val = -100
            temp2 = deepcopy(predictions[0][0])
            for index, val in enumerate(predictions[0][0]):
                if 7 == free_space[index][1]:
                    continue
#                if val > opt_val and free_space[index]:
                if val > opt_val:
                    opt_val = val
                    move = free_space[index]
            if opt_val == -100:
                opt_val = 0
            if winner == self.player:
                temp2[self.state_index[i][1][0]] = 1 + temp2[self.state_index[i][1][0]]
            if winner == (self.player % 2) + 1:
                temp2[self.state_index[i][1][0]] = temp2[self.state_index[i][1][0]] - 1
            _, __ = self.sess.run([self.optimiser, self.cost], feed_dict={self.input: temp_state, self.reward: temp2, self.Q: predictions[0]})
        self.state_index = []
        return 0
