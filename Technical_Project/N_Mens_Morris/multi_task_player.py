#Packages
import numpy as np
import random
from copy import deepcopy
import csv
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from math import log
import networkx as nx
import operator
import cProfile, pstats


adj_dict_3 = [[1, 3, 4],
	      [0, 2, 4],
	      [1, 4, 5],
	      [0, 4, 6],
	      [0, 1, 2, 3, 5, 6, 7, 8],
	      [2, 4, 8],
	      [7, 3, 4],
	      [6, 8, 4],
	      [7, 4, 5]]

adj_dict_6 = [[1, 6],
	      [0, 2, 4],
	      [1, 9],
	      [4, 7],
	      [1, 3, 5],
	      [4, 8],
	      [7, 0, 13],
	      [10, 3, 6],
	      [12, 5, 9],
	      [2, 8, 15],
	      [7, 11],
	      [12, 14, 10],
	      [8, 11],
	      [14, 6],
	      [13, 15, 11],
	      [9, 14]]

adj_dict_9 = [[1, 9],
	      [0, 2, 4],
	      [1, 14],
	      [4, 10],
	      [1, 3, 5, 7],
	      [4, 13],
	      [7, 11],
	      [4, 6, 8],
	      [12, 7],
	      [0, 21, 10],
	      [11, 18, 3, 9],
	      [6, 15, 10],
	      [8, 17, 13, 14],
	      [14, 20, 5],
	      [2, 23, 13],
	      [11, 15],
	      [15, 17, 19],
	      [12, 16],
	      [10, 19],
	      [18, 20, 10, 22],
	      [19, 13],
	      [9, 22],
	      [21, 23, 19],
	      [22, 14]]

adj_dict_12 = [[1, 9, 3],
	       [0, 2, 4],
	       [1, 14, 5],
	       [4, 10, 0, 6],
	       [1, 3, 5, 7],
	       [4, 13, 2, 8],
	       [7, 11, 3],
	       [4, 6, 8],
	       [12, 7, 5],
	       [0, 21, 10],
	       [11, 18, 3],
	       [6, 15],
	       [8, 17],
	       [14, 20, 5],
	       [2, 23, 13],
	       [11, 15, 18],
	       [15, 17, 19],
	       [12, 16, 20],
	       [10, 18, 15, 21],
	       [18, 20, 10, 22],
	       [19, 13, 17, 20],
	       [9, 22, 18],
	       [21, 23, 19],
	       [22, 14, 20]]

decision_type_to = [1,0,0]
decision_type_from = [0,1,0]
decision_type_remove = [0,0,1]

sym3_1 = [2,5,8,1,4,7,0,3,6]
#sym3_1 = [6,7,8,15,16,17,24,25,26,3,4,5,12,13,14,21,22,23,0,1,2,9,10,11,18,19,20]
sym3_2 = [8,7,6,5,4,3,2,1,0]
#sym3_2 = [24,25,26,21,22,23,18,19,20,15,16,17,12,13,14,9,10,11,6,7,8,3,4,5,0,1,2]
sym3_3 = [6,3,0,7,4,1,8,5,2]
#sym3_3 = [18,19,20,9,10,11,0,1,2,21,22,23,12,13,14,3,4,5,24,25,26,15,16,17,6,7,8]
sym3_4 = [6,7,8,3,4,5,0,1,2]
#sym3_4 = [18,19,20,21,22,23,24,25,26,9,10,11,12,13,14,15,16,17,0,1,2,3,4,5,6,7,8]
sym3_5 = [0,3,6,1,4,7,2,5,8]
#sym3_5 = [0,1,2,9,10,11,18,19,20,3,4,5,12,13,14,21,22,23,6,7,8,15,16,17,24,25,26]
sym3_6 = [2,1,0,5,4,3,8,7,6]
#sym3_6 = [6,7,8,3,4,5,0,1,2,15,16,17,12,13,14,9,10,11,24,25,26,21,22,23,18,19,20]
sym3_7 = [8,5,2,7,4,1,6,3,0]
#sym3_7 = [24,25,26,15,16,17,6,7,8,21,22,23,12,13,14,3,4,5,18,19,20,9,10,11,0,1,2]

sym6_1 = [2,9,15,5,8,12,1,4,11,14,3,7,10,0,6,13]
#sym6_1 = [6,7,8,27,28,29,45,46,47,15,16,17,24,25,26,36,37,38,3,4,5,12,13,14,33,34,35,42,43,44,9,10,11,21,22,23,30,31,32,0,1,2,18,19,20,
#	 36,37,38]
sym6_2 = [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
#sym6_2 = [45,46,47,42,43,44,39,40,41,36,37,38,33,34,35,30,31,32,27,28,29,24,25,26,21,22,23,18,19,20,15,16,17,12,13,14,9,10,11,6,7,8,3,4,
#	 5,0,1,2]
sym6_3 = [13,6,0,10,7,3,14,11,4,1,12,8,5,15,9,2]
#sym6_3 = [39,40,41,18,19,20,0,1,2,30,32,32,21,22,23,9,10,11,42,43,44,33,34,34,12,13,14,3,4,5,36,37,38,24,25,26,15,16,17,45,46,47,27,28,
#	 29,6,7,8]
sym6_4 = [13,14,15,10,11,12,6,7,8,9,3,4,5,0,1,2]
#sym6_4 = [39,40,41,42,43,44,45,47,47,30,31,32,33,34,35,36,37,38,18,19,20,21,22,23,24,25,26,27,28,29,9,10,11,12,13,14,15,16,17,0,1,2,3,4,
#	 5,6,7,8]
sym6_5 = [0,6,13,3,7,10,1,4,11,14,5,8,12,2,9,15]
#sym6_5 = [0,1,2,18,19,20,39,40,41,9,10,11,21,22,23,30,31,32,3,4,5,12,13,14,33,34,35,42,43,44,15,16,17,24,25,26,36,37,38,6,7,8,27,28,29,
#	 45,46,47]
sym6_6 = [2,1,0,5,4,3,9,8,7,6,12,11,10,15,14,13]
#sym6_6 = [6,7,8,3,4,5,0,1,2,15,16,17,12,13,14,9,10,11,27,28,29,24,25,26,21,22,23,18,19,20,36,37,38,33,34,35,30,31,32,45,46,47,42,43,44,
#	 39,40,41]
sym6_7 = [15,9,2,12,8,5,14,11,4,1,10,7,3,13,6,0]
#sym6_7 = [45,46,47,27,28,29,6,7,8,36,37,38,24,25,26,15,16,17,42,43,44,33,34,35,12,13,14,3,4,5,30,31,32,21,22,23,9,10,11,39,40,41,18,19,
#	  20,0,1,2]

sym9_1 = [2,14,23,5,13,20,8,12,17,1,4,7,16,19,22,6,11,15,3,10,18,0,9,21]
#sym9_1 = [6,7,8,42,43,44,69,70,71,15,16,17,39,40,41,60,61,62,24,25,26,36,37,38,51,52,53,3,4,5,12,13,14,21,22,23,48,49,50,57,58,59,66,67,
#	 68,18,19,20,33,34,35,45,46,47,9,10,11,30,31,32,54,55,56,0,1,2,27,28,29,63,64,65]
sym9_2 = [23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
#sym9_2 = [69,70,71,66,67,68,63,64,65,60,61,62,57,58,59,54,55,56,51,52,53,48,49,50,45,46,47,42,43,44,39,40,41,36,37,38,33,34,35,30,31,32,
#	 27,28,29,24,25,26,21,22,23,18,19,20,15,16,17,12,13,14,9,10,11,6,7,8,3,4,5,0,1,2]
sym9_3 = [21,9,0,18,10,3,15,11,6,22,19,16,7,4,1,17,12,8,20,13,5,23,14,2]
#sym9_3 = [63,64,65,27,28,29,0,1,2,54,55,56,30,31,32,9,10,11,45,46,47,33,34,35,18,19,20,66,67,68,57,58,59,38,49,50,21,22,23,12,13,14,3,4,
#	  5,51,52,53,36,37,38,24,25,26,60,61,62,39,40,41,15,16,17,69,70,71,42,43,44,6,7,8]
sym9_4 = [21,22,23,18,19,20,15,16,17,9,10,11,12,13,14,6,7,8,3,4,5,0,1,2]
#sym9_4 = [63,64,65,66,67,68,69,70,71,54,55,56,57,58,59,60,61,62,45,46,47,48,49,50,51,52,53,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,
#	  42,43,44,18,19,20,21,22,23,24,25,26,9,10,11,12,13,14,15,16,17,0,1,2,3,4,5,6,7,8]
sym9_5 = [0,9,21,3,10,18,6,11,15,1,4,7,16,19,22,8,12,17,5,13,20,2,14,23]
#sym9_5 = [0,1,2,27,28,29,63,64,65,9,10,11,30,31,32,54,55,56,18,19,20,33,34,35,45,46,47,3,4,5,12,13,14,21,22,23,48,49,50,57,58,59,66,67,
#	  68,24,25,26,36,37,38,51,52,53,15,16,17,39,40,41,60,61,62,6,7,8,42,43,44,69,70,71]
sym9_6 = [2,1,0,5,4,3,8,7,6,14,13,12,11,10,9,17,16,15,20,19,18,23,22,21]
#sym9_6 = [6,7,8,3,4,5,0,1,2,15,16,17,12,13,14,9,10,11,24,25,26,21,22,23,18,19,20,42,43,44,39,40,41,36,37,38,33,34,35,30,31,32,27,28,29,
#	  51,52,53,48,49,50,45,46,47,60,61,62,57,58,59,54,55,56,69,70,71,66,67,68,63,64,65]
sym9_7 = [23,14,2,20,13,5,17,12,8,22,19,16,7,4,1,15,11,6,18,10,3,21,9,0]
#sym9_7 = [69,70,71,42,43,44,6,7,8,60,61,62,39,40,41,15,16,17,51,52,53,36,37,38,24,25,26,66,67,68,57,58,59,48,49,50,21,22,23,12,13,14,3,
#	  4,5,45,46,47,33,34,35,18,19,20,54,55,56,30,31,32,9,10,11,63,64,65,27,28,29,0,1,2]

sym3 = [sym3_1,sym3_2,sym3_3,sym3_4,sym3_5,sym3_6,sym3_7]
sym6 = [sym6_1,sym6_2,sym6_3,sym6_4,sym6_5,sym6_6,sym6_7]
sym9 = [sym9_1,sym9_2,sym9_3,sym9_4,sym9_5,sym9_6,sym9_7]

class Learned_Player(object):
	
	def __init__(self, epsilon, alpha, gamma, limit):

		self.sess = tf.Session()
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.limit = limit
		
		self.to_index = [(None, None, None)] * self.limit
		self.from_index = [(None, None, None)] * (self.limit - 6)
		self.remove_index = [(None, None, None)] * 19
		
		self.to_qval_index = [None] * self.limit
		self.from_qval_index = [None] * (self.limit - 6)
		self.remove_qval_index = [None] * 19
		
		self.n_classes_base = 256
		self.n_classes_3 = 9
		self.n_classes_6 = 16
		self.n_classes_9 = 24
		
		self.n_nodes_base_1 = 512
		self.n_nodes_base_2 = 512
		
		self.n_nodes_3_1 = 18
		self.n_nodes_3_2 = 18
		
		self.n_nodes_6_1 = 32
		self.n_nodes_6_2 = 32
		
		self.n_nodes_9_1 = 48
		self.n_nodes_9_2 = 48
		
		self.n_nodes_12_1 = 48
		self.n_nodes_12_2 = 48
		
		self.future_steps = 0
		self.symmetry_index = [None] * self.n_classes
		self.piece_adj_list = [None] * 12
		
		self.base_input = tf.placeholder(tf.float32, [24])
		self.x_p1 = tf.cast(tf.equal(self.base_input, 1), tf.float32)
		self.x_p2 = tf.cast(tf.equal(self.base_input, 2), tf.float32)
		self.x_empty = tf.cast(tf.equal(self.base_input, 0), tf.float32)
		
		#game_type = 1 at 0 if game_type = 3, 1 if 6, 2 if 9, 3 if 12
		self.game_type = tf.placeholder(tf.float32, [4])
		#ARE WE GOING TO USE THIS DURING TRANSFER
		
		#decision_type = 1 at 0 if place, 1 if choose piece to move, 2 if move piece to, 3 if remove piece
		self.decision_type = tf.placeholder(tf.float32, shape=[3])
		
		self.collect_board = [self.x_empty,self.x_p1,self.x_p2]
		self.collect_other = tf.concat([self.game_type, self.decision_type], 0)
		self.final_board = tf.reshape(self.collect_board, shape=[72])
		self.final_other = tf.reshape(self.collect_other, shape=[7])
		self.x_bin = tf.concat([self.final_board, self.final_other], 0)
		self.x = tf.reshape(self.x_bin, shape=[1,self.n_input])
		self.reward = tf.placeholder(tf.float32,[self.n_classes])
		self.y = tf.reshape(self.reward, [1, self.n_classes])
		self.Q_val_base = self.base_network()
		
		#cost functions
		self.cost = tf.reduce_mean(tf.squared_difference(self.y, self.Q_val_base))
		
		#optimisers
		self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost)
		
	def base_network(self):

		l1 = tf.layers.dense(
			inputs=self.x,
			units=self.n_input,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

		l2 = tf.layers.dense(
			inputs=l1,
			units=self.n_nodes_1,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

#		l3 = tf.layers.dense(
#		inputs=l2,
#		units=self.n_nodes_2,
#		bias_initializer=tf.constant_initializer(0, 1),
#		activation=tf.nn.leaky_relu,
#		kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#		activity_regularizer=tf.nn.softmax
#		)


		l_out = tf.layers.dense(
			inputs=l2,
			units=self.n_classes,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

		l_norm = tf.contrib.layers.softmax(
			logits=l_out
		)

		return l_norm

