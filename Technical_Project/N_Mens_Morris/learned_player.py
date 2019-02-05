#Packages
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import csv
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from math import log
import networkx as nx

class Learned_Player(object):
	def __init__(self, player, alpha, epsilon, gamma):
		
		self.sess = tf.Session()
        self.player = player
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.state_index = []

        self.n_classes = 24
        self.n_input = 80
        self.n_nodes_1 = self.n_classes * 2
        self.n_nodes_2 = self.n_classes * 2
        self.n_nodes_3 = self.n_classes * 2
        self.n_nodes_4 = self.n_classes * 2


        self.input = tf.placeholder(tf.float32, [24])
        self.game_type = tf.placeholder(tf.float32, [1])
#        self.available = tf.placeholder(tf.float32, [7])
        self.x_p1 = tf.cast(tf.equal(self.input, 1), tf.float32)
        self.x_p2 = tf.cast(tf.equal(self.input, 2), tf.float32)
        self.x_empty = tf.cast(tf.equal(self.input, 0), tf.float32)
        self.ttemp = [0] * 4
		self.x_game_type = deepcopy(self.ttemp)
        self.x_game_type[tf.divide(self.game_type,3)] = 1
        self.x_bin = [self.x_empty,self.x_p1,self.x_p2,self.x_game_type]
#        self.x = tf.reshape(self.input, shape=[1,self.n_input])
        self.x = tf.reshape(self.x_bin, shape=[1,self.n_input])
        self.reward = tf.placeholder(tf.float32,[self.n_classes])
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

	def neural_network(self):

		l1 = tf.layers.dense(
			inputs=self.x,
			units=self.n_input,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
			activity_regularizer=tf.nn.softmax
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
            kernel_initializer = tf.constant_initializer(0,1),
            bias_initializer=tf.constant_initializer(0, 1),
            activation=tf.nn.leaky_relu,
            activity_regularizer=tf.nn.softmax
        )

        l_norm = tf.contrib.layers.softmax(
        logits=l_out
        )

        return l_norm
