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

adj_3 = [[0,1,0,1,1,0,0,0,0],
       [1,0,1,0,1,0,0,0,0],
       [0,1,0,0,1,1,0,0,0],
       [1,0,0,0,1,0,1,0,0],
       [1,1,1,1,0,1,1,1,1],
       [0,0,1,0,1,0,0,0,1],
       [0,0,0,1,1,0,0,1,0],
       [0,0,0,0,1,0,1,0,1],
       [0,0,0,0,1,1,0,1,0]]


adj_dict_3 = {
"[0, 0]": [[0,1], [1,0], [1,1]],
"[0, 1]": [[0,0], [0,2], [1,1]],
"[0, 2]": [[0,1], [1,1], [1,2]],
"[1, 0]": [[0,0], [1,1], [2,0]],
"[1, 1]": [[0,0], [0,1], [0,2], [1,0], [1,2], [2,0], [2,1], [2,2]],
"[1, 2]": [[0,2], [1,1], [2,2]],
"[2, 0]": [[2,1], [1,0], [1,1]],
"[2, 1]": [[2,0], [2,2], [1,1]],
"[2, 2]": [[2,1], [1,1], [1,2]],
}

mill_dict_3 = {
"[0, 0]": [[[0,1], [0,2]], [[1,0], [2,0]], [[1,1], [2,2]]],
"[0, 1]": [[[0,0], [0,2]], [[1,1], [2,1]]],
"[0, 2]": [[[1,2], [2,2]], [[1,1], [2,0]], [[0,0], [0,1]]],
"[1, 0]": [[[0,0], [2,0]], [[1,1], [1,2]]],
"[1, 1]": [[[0,0], [2,2]], [[0,1], [2,1]], [[0,2], [2,0]], [[1,0], [1,2]]],
"[1, 2]": [[[0,2], [2,2]], [[1,1], [1,0]]],
"[2, 0]": [[[2,1], [2,2]], [[1,1], [0,2]], [[0,0], [1,0]]],
"[2, 1]": [[[2,0], [2,2]], [[1,1], [0,1]]],
"[2, 2]": [[[2,1], [2,0]], [[1,2], [0,2]], [[0,0], [1,1]]]
}

adj_6 = [[0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
       [1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
       [0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
       [0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0],
       [0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
       [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
       [1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0],
       [0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0],
       [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
       [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1],
       [0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0],
       [0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0],
       [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
       [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
       [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0]]

adj_dict_6 = {
"[0, 0]": [[0,1], [2,0]],
"[0, 1]": [[0,0], [0,2], [1,1]],
"[0, 2]": [[0,1], [2,3]],
"[1, 0]": [[1,1], [2,1]],
"[1, 1]": [[0,1], [1,0], [1,2]],
"[1, 2]": [[1,1], [2,2]],
"[2, 0]": [[2,1], [0,0], [4,0]],
"[2, 1]": [[3,0], [1,0], [2,0]],
"[2, 2]": [[3,2], [1,2], [2,3]],
"[2, 3]": [[0,2], [2,2], [4,2]],
"[3, 0]": [[2,1], [3,1]],
"[3, 1]": [[3,2], [4,1], [3,0]],
"[3, 2]": [[2,2], [3,1]],
"[4, 0]": [[4,1], [2,0]],
"[4, 1]": [[4,0], [4,2], [3,1]],
"[4, 2]": [[2,3], [4,1]]
}

mill_dict_6 = {
"[0, 0]": [[[0,1], [0,2]], [[4,0], [2,0]]],
"[0, 1]": [[[0,0], [0,2]]],
"[0, 2]": [[[2,3], [4,2]], [[0,0], [0,1]]],
"[1, 0]": [[[3,0], [2,1]], [[1,1], [1,2]]],
"[1, 1]": [[[1,0], [1,2]]],
"[1, 2]": [[[3,2], [2,2]], [[1,1], [1,0]]],
"[2, 0]": [[[0,0], [4,0]]],
"[2, 1]": [[[3,0], [1,0]]],
"[2, 2]": [[[1,2], [3,2]]],
"[2, 3]": [[[0,2], [4,2]]],
"[3, 0]": [[[1,0], [2,1]], [[3,1], [3,2]]],
"[3, 1]": [[[3,0], [3,2]]],
"[3, 2]": [[[3,1], [3,0]], [[1,2], [2,2]]],
"[4, 0]": [[[2,0], [0,0]], [[4,1], [4,2]]],
"[4, 1]": [[[4,0], [4,2]]],
"[4, 2]": [[[2,3], [0,2]], [[4,0], [4,1]]]
}

adj_9 = np.matrix([[0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                   [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0],
                   [0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0],
                   [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
                   [0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0],
                   [0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
                   [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0]])

adj_dict_9 = {
"[0, 0]": [[0,1], [3,0]],
"[0, 1]": [[0,0], [0,2], [1,1]],
"[0, 2]": [[0,1], [3,5]],
"[1, 0]": [[1,1], [3,1]],
"[1, 1]": [[0,1], [1,0], [1,2], [2,1]],
"[1, 2]": [[1,1], [3,4]],
"[2, 0]": [[2,1], [3,2]],
"[2, 1]": [[1,1], [2,0], [2,2]],
"[2, 2]": [[3,3], [2,1]],
"[3, 0]": [[0,0], [6,0], [3,1]],
"[3, 1]": [[3,2], [5,0], [1,0], [3,0]],
"[3, 2]": [[2,0], [4,0], [3,1]],
"[3, 3]": [[2,2], [4,2], [3,4], [3,5]],
"[3, 4]": [[3,5], [5,2], [1,2]],
"[3, 5]": [[0,2], [6,2], [3,4]],
"[4, 0]": [[3,2], [4,0]],
"[4, 1]": [[4,0], [4,2], [5,1]],
"[4, 2]": [[3,3], [4,1]],
"[5, 0]": [[3,1], [5,1]],
"[5, 1]": [[5,0], [5,2], [3,1], [6,1]],
"[5, 2]": [[5,1], [3,4]],
"[6, 0]": [[3,0], [6,1]],
"[6, 1]": [[6,0], [6,2], [5,1]],
"[6, 2]": [[6,1], [3,5]]
}

mill_dict_9 = {
"[0, 0]": [[[0,1], [0,2]], [[3,0], [6,0]]],
"[0, 1]": [[[0,0], [0,2]], [[2,1], [1,1]]],
"[0, 2]": [[[0,0], [0,1]], [[3,5], [6,2]]],
"[1, 0]": [[[5,0], [3,1]], [[1,1], [1,2]]],
"[1, 1]": [[[1,0], [1,2]], [[0,1], [2,1]]],
"[1, 2]": [[[3,4], [5,2]], [[1,1], [1,0]]],
"[2, 0]": [[[3,2], [4,0]], [[2,1], [2,2]]],
"[2, 1]": [[[2,0], [2,2]], [[0,1], [1,1]]],
"[2, 2]": [[[3,3], [4,2]], [[2,0], [2,1]]],
"[3, 0]": [[[3,1], [3,2]], [[0,0], [6,0]]],
"[3, 1]": [[[3,0], [3,2]], [[1,0], [5,0]]],
"[3, 2]": [[[2,0], [4,0]], [[3,0], [3,2]]],
"[3, 3]": [[[2,2], [4,2]], [[3,4], [3,5]]],
"[3, 4]": [[[3,3], [3,5]], [[1,2], [5,2]]],
"[3, 5]": [[[0,2], [6,2]], [[3,3], [3,4]]],
"[4, 0]": [[[2,0], [3,2]], [[4,1], [4,2]]],
"[4, 1]": [[[4,0], [4,2]], [[5,1], [6,1]]],
"[4, 2]": [[[3,3], [2,2]], [[4,0], [4,1]]],
"[5, 0]": [[[3,1], [1,0]], [[5,1], [5,2]]],
"[5, 1]": [[[5,0], [5,2]], [[3,1], [1,0]]],
"[5, 2]": [[[5,1], [5,0]], [[1,2], [3,4]]],
"[6, 0]": [[[3,0], [0,0]], [[6,1], [6,2]]],
"[6, 1]": [[[6,0], [6,2]], [[4,1], [5,1]]],
"[6, 2]": [[[6,0], [6,1]], [[0,2], [3,5]]]
}

adj_12 = np.matrix([[0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                   [1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                   [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0],
                   [0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0],
                   [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
                   [0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0],
                   [0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
                   [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1],
                   [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
                   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0]])

adj_dict_12 = {
"[0, 0]": [[0,1], [3,0], [1,0]],
"[0, 1]": [[0,0], [0,2], [1,1]],
"[0, 2]": [[0,1], [3,5], [1,2]],
"[1, 0]": [[1,1], [3,1], [0,0], [2,0]],
"[1, 1]": [[0,1], [1,0], [1,2], [2,1]],
"[1, 2]": [[1,1], [3,4], [0,2], [2,2]],
"[2, 0]": [[2,1], [3,2], [1,0]],
"[2, 1]": [[1,1], [2,0], [2,2]],
"[2, 2]": [[3,3], [2,1], [1,2]],
"[3, 0]": [[0,0], [6,0], [3,1]],
"[3, 1]": [[3,2], [5,0], [1,0], [3,0]],
"[3, 2]": [[2,0], [4,0], [3,1]],
"[3, 3]": [[2,2], [4,2], [3,4]],
"[3, 4]": [[3,5], [5,2], [1,2], [3,3]],
"[3, 5]": [[0,2], [6,2], [3,4]],
"[4, 0]": [[3,2], [4,0], [5,0]],
"[4, 1]": [[4,0], [4,2], [5,1]],
"[4, 2]": [[3,3], [4,1], [5,2]],
"[5, 0]": [[3,1], [5,0], [4,0], [6,0]],
"[5, 1]": [[5,0], [5,2], [3,1], [6,1]],
"[5, 2]": [[5,1], [3,4], [4,2], [5,2]],
"[6, 0]": [[3,0], [6,1], [5,0]],
"[6, 1]": [[6,0], [6,2], [5,1]],
"[6, 2]": [[6,1], [3,5], [5,2]]
}

mill_dict_12 = {
"[0, 0]": [[[0,1], [0,2]], [[3,0], [6,0]], [[1,0], [2,0]]],
"[0, 1]": [[[0,0], [0,2]], [[2,1], [1,1]]],
"[0, 2]": [[[0,0], [0,1]], [[3,5], [6,2]], [[1,2], [2,2]]],
"[1, 0]": [[[5,0], [3,1]], [[1,1], [1,2]], [[0,0], [2,0]]],
"[1, 1]": [[[1,0], [1,2]], [[0,1], [2,1]]],
"[1, 2]": [[[3,4], [5,2]], [[1,1], [1,0]], [[0,2], [2,2]]],
"[2, 0]": [[[3,2], [4,0]], [[2,1], [2,2]], [[1,0], [0,0]]],
"[2, 1]": [[[2,0], [2,2]], [[0,1], [1,1]]],
"[2, 2]": [[[3,3], [4,2]], [[2,0], [2,1]], [[1,2], [0,2]]],
"[3, 0]": [[[3,1], [3,2]], [[0,0], [6,0]]],
"[3, 1]": [[[3,0], [3,2]], [[1,0], [5,0]]],
"[3, 2]": [[[2,0], [4,0]], [[3,0], [3,2]]],
"[3, 3]": [[[2,2], [4,2]], [[3,4], [3,5]]],
"[3, 4]": [[[3,3], [3,5]], [[1,2], [5,2]]],
"[3, 5]": [[[0,2], [6,2]], [[3,3], [3,4]]],
"[4, 0]": [[[2,0], [3,2]], [[4,1], [4,2]], [[5,0], [6,0]]],
"[4, 1]": [[[4,0], [4,2]], [[5,1], [6,1]]],
"[4, 2]": [[[3,3], [2,2]], [[4,0], [4,1]], [[5,2], [6,2]]],
"[5, 0]": [[[3,1], [1,0]], [[5,1], [5,2]], [[4,0], [6,0]]],
"[5, 1]": [[[5,0], [5,2]], [[3,1], [1,0]]],
"[5, 2]": [[[5,1], [5,0]], [[1,2], [3,4]], [[4,2], [6,2]]],
"[6, 0]": [[[3,0], [0,0]], [[6,1], [6,2]], [[4,0], [5,0]]],
"[6, 1]": [[[6,0], [6,2]], [[4,1], [5,1]]],
"[6, 2]": [[[6,0], [6,1]], [[0,2], [3,5]], [[4,2], [5,2]]]
}

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

# 		 self.optimiser = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)
		#        self.optimiser = tf.train.AdamOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)
#		self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost)
		#        self.optimizer = tf.train.AdograadOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)

	def neural_network(self):

		l1 = tf.layers.dense(
			inputs=self.x,
			units=self.n_input,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
			activity_regularizer=tf.nn.softmax
			kernal_regularizer=tf.contrib.layers.l1_regularizer(0.001)
		)

#        l2 = tf.layers.dense(
#            inputs=l1,
#            units=self.n_nodes_1,
#            bias_initializer=tf.constant_initializer(0, 1),
#            activation=tf.nn.leaky_relu,
#            activity_regularizer=tf.nn.softmax
#			 kernal_regularizer=tf.contrib.layers.l1_regularizer(0.001)
#        )

#        l3 = tf.layers.dense(
#            inputs=l2,
#            units=self.n_nodes_2,
#            bias_initializer=tf.constant_initializer(0, 1),
#            activation=tf.nn.leaky_relu
#            activity_regularizer=tf.nn.softmax
#			 kernal_regularizer=tf.contrib.layers.l1_regularizer(0.001)
#        )

#        l4 = tf.layers.dense(
#            inputs=l3,
#            units=self.n_nodes_3,
#            bias_initializer=tf.constant_initializer(0, 1),
#            activation=tf.nn.leaky_relu
#            activity_regularizer=tf.nn.softmax
#			 kernal_regularizer=tf.contrib.layers.l1_regularizer(0.001)
#        )

#        l5 = tf.layers.dense(
#            inputs=l4,
#            units=self.n_nodes_4,
#            bias_initializer=tf.constant_initializer(0, 1),
#            activation=tf.nn.leaky_relu
#            activity_regularizer=tf.nn.softmax
#			 kernal_regularizer=tf.contrib.layers.l1_regularizer(0.001)
#        )

        l_out = tf.layers.dense(
            inputs=l1,
            units=self.n_classes,
            kernel_initializer = tf.constant_initializer(0,1),
            bias_initializer=tf.constant_initializer(0,1),
            activation=tf.nn.leaky_relu,
            activity_regularizer=tf.nn.softmax
        )

        l_norm = tf.contrib.layers.softmax(
        logits=l_out
        )

        return l_norm
		
	def valid_move(self, state, game_type, free_space, pieces):
        valid_moves = []

        if game_type == 3:
            for piece in pieces:
                for space in adj_dict_3[str(piece)]:
                    if space in free_space:
                        valid_moves.append((piece,space))

        if game_type == 6:
            for piece in pieces:
                for space in adj_dict_6[str(piece)]:
                    if space in free_space:
                        valid_moves.append((piece,space))

        if game_type == 9:
            for piece in pieces:
                for space in adj_dict_9[str(piece)]:
                    if space in free_space:
                        valid_moves.append((piece,space))

        if game_type == 12:
            for piece in pieces:
                for space in adj_dict_12[str(piece)]:
                    if space in free_space:
                        valid_moves.append((piece,space))

		return valid_moves
		
	def random_place(self, state, free_space):
        temp = random.randint(0, len(free_space) - 1)
        return free_space[temp]
		
	def place(self, state, free_space, game_type, nodes):
		rand = random.randint(1,100)
        move = None
        if rand <= 100*self.epsilon:
            move = self.random_place(state)
            return move
        else:
			predictions = self.sess.run([self.Q_val], feed_dict={self.input: state})
            opt_val = -float('Inf')
            for index, val in enumerate(predictions[0][0]):
				while(index < nodes):
                	if val > opt_val and free_space[index]:
                    	opt_val = val
                    	move = free_space[index]
            if move is None:
                
            self.state_index.append((deepcopy(state),move))
            return move
        return free_space[temp]
