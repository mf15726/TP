#Packages
import numpy as np
import random
from copy import deepcopy
import csv
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from math import log
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

class Multi_Task_Player(object):
	
	def __init__(self, epsilon, alpha, gamma, limit):

		self.sess = tf.Session()
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.limit = limit
		
		self.to_index = [(None, None, None)] * self.limit
		self.from_index = [(None, None, None)] * (self.limit - 6)
		self.remove_index = [(None, None, None)] * 19
		
		self.to_qval_base_index = [None] * self.limit
		self.from_qval_base_index = [None] * (self.limit - 6)
		self.remove_qval_base_index = [None] * 19
		
		self.to_future_index = [None] * self.limit
		self.from_future_index = [None] * (self.limit - 6)
		self.remove_future_index = [None] * 19
		
		self.to_qval_task_index = [None] * self.limit
		self.from_qval_task_index = [None] * (self.limit - 6)
		self.remove_qval_task_index = [None] * 19
		
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
		
		self.n_input_base = 75
		self.n_input_task = self.n_classes_base + 3
		
		self.future_steps = 0
		self.symmetry_index = [None] * self.n_classes_9
		self.symmetry_future_index = [None] * self.n_classes_9
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
		self.final_board = tf.reshape(self.collect_board, shape=[72])
		self.final_other = tf.reshape(self.decision_type, shape=[3])
		self.x_bin = tf.concat([self.final_board, self.final_other], 0)
		self.x = tf.reshape(self.x_bin, shape=[1,self.n_input_base])
		self.reward_base = tf.placeholder(tf.float32,[self.n_classes_base])
		self.reward_3 = tf.placeholder(tf.float32, self.n_classes_3)
		self.reward_6 = tf.placeholder(tf.float32, self.n_classes_6)
		self.reward_9 = tf.placeholder(tf.float32, self.n_classes_9)
		self.y_base = tf.reshape(self.reward_base, [1, self.n_classes_base])
		self.y_3 = tf.reshape(self.reward_3, [1, self.n_classes_3])
		self.y_6 = tf.reshape(self.reward_6, [1, self.n_classes_6])
		self.y_9 = tf.reshape(self.reward_9, [1, self.n_classes_9])
		
		#Task specific networks
		self.task_input = tf.placeholder(tf.float32, shape=[self.n_classes_base])
		self.collect_task = tf.concat([self.task_input, self.decision_type], 0)
		self.x_task = tf.reshape(self.collect_task, shape=[1, self.n_input_task])
		self.Q_val_base = self.base_network()
		self.Q_val_task3 = self.task3_network()
		self.Q_val_task6 = self.task6_network()
		self.Q_val_task9 = self.task9_network()
		self.Q_val_task12 = self.task12_network()
		
		#cost functions
		self.cost_base = tf.reduce_mean(tf.squared_difference(self.y_base, self.Q_val_base))
		self.cost_task3 = tf.reduce_mean(tf.squared_difference(self.y_3, self.Q_val_task3))
		self.cost_task6 = tf.reduce_mean(tf.squared_difference(self.y_6, self.Q_val_task6))
		self.cost_task9 = tf.reduce_mean(tf.squared_difference(self.y_9, self.Q_val_task9))
		self.cost_task12 = tf.reduce_mean(tf.squared_difference(self.y_9, self.Q_val_task12))
		
		#optimisers
		self.optimiser_base = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost_base)
		self.optimiser_3 = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost_task3)
		self.optimiser_6 = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost_task6)
		self.optimiser_9 = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost_task9)
		self.optimiser_12 = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost_task12)
		
	def base_network(self):

		l1 = tf.layers.dense(
			inputs=self.x,
			units=self.n_input_base,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

		l2 = tf.layers.dense(
			inputs=l1,
			units=self.n_nodes_base_1,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

#		l3 = tf.layers.dense(
#		inputs=l2,
#		units=self.n_nodes_base_2,
#		bias_initializer=tf.constant_initializer(0, 1),
#		activation=tf.nn.leaky_relu,
#		kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#		activity_regularizer=tf.nn.softmax
#		)


		l_out = tf.layers.dense(
			inputs=l2,
			units=self.n_classes_base,
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
	
	def task3_network(self):

		l1 = tf.layers.dense(
			inputs=self.x_task,
			units=self.n_input_task,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

		l2 = tf.layers.dense(
			inputs=l1,
			units=self.n_nodes_3_1,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

#		l3 = tf.layers.dense(
#		inputs=l2,
#		units=self.n_nodes_3_2,
#		bias_initializer=tf.constant_initializer(0, 1),
#		activation=tf.nn.leaky_relu,
#		kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#		activity_regularizer=tf.nn.softmax
#		)


		l_out = tf.layers.dense(
			inputs=l2,
			units=self.n_classes_3,
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

		
	def task6_network(self):

		l1 = tf.layers.dense(
			inputs=self.x_task,
			units=self.n_input_task,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

		l2 = tf.layers.dense(
			inputs=l1,
			units=self.n_nodes_6_1,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

#		l3 = tf.layers.dense(
#		inputs=l2,
#		units=self.n_nodes_6_2,
#		bias_initializer=tf.constant_initializer(0, 1),
#		activation=tf.nn.leaky_relu,
#		kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#		activity_regularizer=tf.nn.softmax
#		)


		l_out = tf.layers.dense(
			inputs=l2,
			units=self.n_classes_6,
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
	
	def task9_network(self):

		l1 = tf.layers.dense(
			inputs=self.x_task,
			units=self.n_input_task,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

		l2 = tf.layers.dense(
			inputs=l1,
			units=self.n_nodes_9_1,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

#		l3 = tf.layers.dense(
#		inputs=l2,
#		units=self.n_nodes_9_2,
#		bias_initializer=tf.constant_initializer(0, 1),
#		activation=tf.nn.leaky_relu,
#		kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#		activity_regularizer=tf.nn.softmax
#		)


		l_out = tf.layers.dense(
			inputs=l2,
			units=self.n_classes_9,
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
	
	def task12_network(self):

		l1 = tf.layers.dense(
			inputs=self.x_task,
			units=self.n_input_task,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

		l2 = tf.layers.dense(
			inputs=l1,
			units=self.n_nodes_12_1,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
#			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

#		l3 = tf.layers.dense(
#		inputs=l2,
#		units=self.n_nodes_12_2,
#		bias_initializer=tf.constant_initializer(0, 1),
#		activation=tf.nn.leaky_relu,
#		kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#		activity_regularizer=tf.nn.softmax
#		)


		l_out = tf.layers.dense(
			inputs=l2,
			units=self.n_classes_9,
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
	
	def piece_adj(self, state, game_type, space, pieces, player):
		self.piece_adj_list = [None] * 12
		
		counter = 0
		if game_type == 3:
			for item in adj_dict_3[space]:
				if state[item] == player:
					self.piece_adj_list[counter] = item
					counter += 1
					
		if game_type == 6:
			for item in adj_dict_6[space]:
				if state[item] == player:
					self.piece_adj_list[counter] = item
					counter += 1
		
		if game_type == 9:
			for item in adj_dict_9[space]:
				if state[item] == player:
					self.piece_adj_list[counter] = item
					counter += 1
					
		if game_type == 12:
			for item in adj_dict_12[space]:
				if state[item] == player:
					self.piece_adj_list[counter] = item
					counter += 1
					
	def valid_move(self, state, game_type, pieces):
		valid_moves = []
		if game_type == 3:
			for piece in pieces:
				if piece is None:
					continue
				for space in adj_dict_3[piece]:
					if state[space] == 0:
						valid_moves.append((piece,space))

		if game_type == 6:
			for piece in pieces:
				if piece is None:
					continue
				for space in adj_dict_6[piece]:
					if state[space] == 0:
						valid_moves.append((piece,space))

		if game_type == 9:
			for piece in pieces:
				if piece is None:
					continue
				for space in adj_dict_9[piece]:
					if state[space] == 0:
						valid_moves.append((piece,space))

		if game_type == 12:
			for piece in pieces:
				if piece is None:
					continue
				for space in adj_dict_12[piece]:
					if state[space] == 0:
						valid_moves.append((piece,space))

		return valid_moves
	
	def padding(self,state,game_type):
		temp = deepcopy(state)
		if game_type > 6:
			return temp
		if game_type == 3:
			temp.extend([0]*15)
		else:
			temp.extend([0]*8)
		return temp
	
	def convert_board(self, state, player):
		if player == 1:
			return state
		else:
			new_state = deepcopy(state)
			for item in new_state:
				item = (item % 2) + 1
			return new_state
		
	def max_next_Q(self, state, game_type, player, decision):
		predictions_base = self.sess.run([self.Q_val_base], feed_dict={self.input_base: input_state,
									      self.decision_type: decision_type})
		predictions_task = self.task_specific(game_type, decision_type, predicitions_base)
		val_base = np.argmax(predictions_base[0][0])
		val_task = np.argmax(predicitions_task[0][0])
		return val
	
	def task_specific(self, game_type, decision_type, predictions_base):
		if game_type == 3:
			predictions_task = self.sess.run([self.Q_val_task3], feed_dict={self.task_input: predictions_base[0][0],
										   self.decision_type: decision_type})
		elif game_type == 6:
			predictions_task = self.sess.run([self.Q_val_task6], feed_dict={self.task_input: predictions_base[0][0],
										   self.decision_type: decision_type})
		elif game_type == 9:
			predictions_task = self.sess.run([self.Q_val_task9], feed_dict={self.task_input: predictions_base[0][0],
										   self.decision_type: decision_type})
		else:
			predictions_task = self.sess.run([self.Q_val_task12], feed_dict={self.task_input: predictions_base[0][0],
										   self.decision_type: decision_type})
		return predictions_task
	
	def random_place(self, state):
		space_val = 1
		while space_val != 0:
			space = random.randint(0, len(state) - 1)
			space_val = state[space]
		return space
	
	def place(self, state, game_type, player, move_no):
		rand = random.randint(1,100)
		move = None
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		input_state = self.convert_board(state,player)
		input_state = self.padding(input_state,game_type)
		predictions_base = self.sess.run([self.Q_val_base], feed_dict={self.base_input: input_state,
									       self.decision_type: decision_type_to})
						 
		predictions_task = self.task_specific(game_type,decision_type_to,predictions_base)
		
		if rand <= 100*self.epsilon:
			move = self.random_place(state)
			self.to_qval_base_index.append(predictions_base[0][0])
			self.to_qval_task_index.append(predictions_task[0][0])
			self.to_index.append((deepcopy(input_state),move,player))
			return move
		else:
			opt_val = -float('Inf')
			for index, item in enumerate(state):
				if item != 0:
					continue
				val = predictions_task[0][0][index]
				if val > opt_val:
					opt_val = val
					move = index
			self.to_qval_base_index[move_no] = predictions_base[0][0]
			self.to_qval_task_index[move_no] = predictions_task[0][0]
			self.to_index[move_no] = ((deepcopy(input_state),move,player))
			return move
		
	def move(self, state, game_type, pieces, player, enable_flying, move_no):
		valid_moves = self.valid_move(state, game_type, pieces)
		if len(valid_moves) == 0 and not enable_flying:
			return (25, 25)
		move = None
		piece = None
		rand = random.randint(1,100)
		input_state = self.convert_board(state,player)
		input_state = self.padding(input_state,game_type)
		predictions_base_to = self.sess.run([self.Q_val_base], feed_dict={self.base_input: input_state,
										  self.decision_type: decision_type_to})
		predictions_task_to = self.task_specific(game_type,decision_type_to,predictions_base_to)
		predictions_base_from = self.sess.run([self.Q_val_base], feed_dict={self.base_input: input_state,
										   self.decision_type: decision_type_from})
		predictions_task_from = self.task_specific(game_type,decision_type_from,predictions_base_from)
		if rand <= 100*self.epsilon:
			random_move = self.random_move(state, valid_moves, enable_flying, pieces)
			self.to_index[move_no] = (deepcopy(input_state),random_move[0], player)
			self.from_index[int(move_no - (game_type * 2))] = (deepcopy(input_state),random_move[1],player)
			self.to_qval_base_index[move_no] = predictions_base_to[0][0]
			self.to_qval_task_index[move_no] = predictions_task_to[0][0]
			self.from_qval_base_index[int(move_no - (game_type * 2))] = predictions_base_from[0][0]
			self.from_qval_task_index[int(move_no - (game_type * 2))] = predictions_task_from[0][0]
#			print('Random move = ' + str(random_move))
			return random_move
		else:
			opt_val = -float('Inf')
			if enable_flying:
				adj_piece_list = pieces
				for index, item in enumerate(state):
					if item != 0:
						continue
					val = predictions_task_to[0][0][index]
#					print('Index, Val ' +str(index) + ' ' + str(val))
					if val > opt_val:
						opt_val = val
						move = index
			else:
				for index, item in enumerate(state):
					if item != 0:
#						print('We skip' + str(index))
						continue
					
					val = predictions_task_to[0][0][index]
#					print('OptVal = ' + str(opt_val))
#					print('Index, Val ' +str(index) + ' ' + str(val))
					if val > opt_val:
						self.piece_adj(state, game_type, index, pieces, player)
#						print('WE HAVE SUCCESS' + str(adj_piece))
						if self.piece_adj_list[0] is None:
							continue
						else:
							adj_piece_list = self.piece_adj_list
							opt_val = val
							move = index					
			if move is None:
				print('No move')
				return (25,25)
			
			opt_val = -float('Inf')
#			print('Adj Pieces ' +str(adj_piece_list))
			for item in adj_piece_list:
				if item is None:
					continue
#				print('Alright here we go ' + str(item))
				val = predictions_task_from[0][0][item]
#				print('VAl = ' +str(val) + ' Opt_Val = ' +str(opt_val))
				if val > opt_val:
					opt_val = val
					piece = item
#					print('Piece is ' +str(piece))
			if piece is None:
				print('No piece')
				return(25,25)
					
			predicted_move = (piece, move)
#			print('We predict ' +str(predicted_move))
			self.to_index[move_no] = (deepcopy(input_state), move, player)
			self.from_index[int(move_no - (game_type * 2))] = (deepcopy(input_state),piece,player)
			self.to_qval_base_index[move_no] = predictions_base_to[0][0]
			self.to_qval_task_index[move_no] = predictions_task_to[0][0]
			self.from_qval_base_index[int(move_no - (game_type * 2))] = predictions_base_from[0][0]
			self.from_qval_task_index[int(move_no - (game_type * 2))] = predictions_task_from[0][0]
		return predicted_move

	def random_move(self, state, valid_moves, enable_flying, piece_list):
		if len(valid_moves) == 1:
			temp = 0
		if enable_flying:
			free_space = self.free_space_finder(state)
			temp = random.randint(0, len(free_space) - 1)
			temp2 = random.randint(0, len(piece_list) - 1)
			while piece_list[temp2] is None:
				temp2 = random.randint(0, len(piece_list) - 1)
#				print('Valid = ' +str(valid_moves))
#				print('Piece List ' + str(piece_list))
			return (piece_list[temp2],free_space[temp])
		else:
			temp = random.randint(0, len(valid_moves) - 1)
			return valid_moves[temp]
	
	def random_remove_piece(self, piece_list):
		piece_to_remove = None
		while piece_to_remove is None:
			temp = random.randint(0, len(piece_list) - 1)
			piece_to_remove = piece_list[temp]
		return piece_to_remove
	
	def remove_piece(self, state, piece_list, game_type, player, pieces_removed):
		opponent = (player % 2) + 1
		rand = random.randint(1,100)
		input_state = self.convert_board(state,player)
		input_state = self.padding(input_state,game_type)
		predictions_base = self.sess.run([self.Q_val_base], feed_dict={self.base_input: input_state,
									       self.decision_type: decision_type_remove})
		predictions_task = self.task_specific(game_type,decision_type_remove,predictions_base)
		if rand <= 100*self.epsilon:
			piece = self.random_remove_piece(piece_list)
			self.remove_index[pieces_removed] = (deepcopy(input_state),piece,player)
			self.remove_qval_base_index[pieces_removed] = predictions_base[0][0]
			self.remove_qval_task_index[pieces_removed] = predictions_task[0][0]
			return piece
		else:
			opt_val = -float('Inf')
			for index, item in enumerate(state):
				if item != opponent:
					continue
				val = predictions_task[0][0][index]
				if val > opt_val:
					opt_val = val
					piece = index
			self.remove_index[pieces_removed] = (deepcopy(input_state),piece,player)
			self.remove_qval_base_index[pieces_removed] = predictions_base[0][0]
			self.remove_qval_task_index[pieces_removed] = predictions_task[0][0]
		return piece
	
	def free_space_finder(self, state):
		free_space = []
		for item in state:
			if item == 0:
				free_space.append(item)

		return free_space
	
	def reward_function(self, game_type, winner, player, qval_index, decision_type, input_state, task_classes):
		predictions_base = self.sess.run([self.Q_val_base], feed_dict={self.base_input: input_state, self.decision_type: decision_type})
		
		predictions_task = self.task_specific(game_type, decision_type, predictions_base)
		if winner == player:
			reward_base = [1] * self.n_classes_base
			reward_task = [1] * task_classes
		elif winner != 0:
			reward_base =  [-1] * self.n_classes_base
			reward_task = [-1] * task_classes
		else:
			reward_base = [0] * self.n_classes_base
			reward_task = [0] * task_classes
		
		reward_base = list(map(sum, zip((predictions_base[0][0]),reward_base)))
		reward_task = list(map(sum, zip((predictions_task[0][0]),reward_task)))
		
		for item in reward_base:
			for i in range(self.future_steps):
				reward_base[item] += self.gamma**(i+1) * self.max_next_Q(input_state, game_type, player, decision_type)
				
		for item in reward_task:
			for i in range(self.future_steps):
				reward_base[item] += self.gamma**(i+1) * self.max_next_Q(qval_index, game_type, player, decision_type)
				
		return reward_base, reward_task
		
	def symmetry(self, state, sym_box):
		for index, item in enumerate(state):
			if index == len(sym_box):
				break
			temp = sym_box[index]
			self.symmetry_index[index] = state[temp]
			
	def edit_to_index(self,state,move_no):
		self.to_future_index[move_no] = deepcopy(state)
		
	def edit_from_index(self,state,move_no,game_type):
		self.from_future_index[move_no-(game_type*2)] = deepcopy(state)
		
	def edit_remove_index(self,state,pieces_removed):
		self.remove_future_index[pieces_removed] = deepcopy(state)
		
	def learn3(self, game_type, winner):
		counter = 0
		task_classes = 9
		
		for index, item in enumerate(self.to_index):
			if None in item:
				break
			reward_base_to, reward_task_to = self.reward_function(game_type,winner,item[2],self.to_qval_base_index[index],decision_type_to,item[0],task_classes)
			
			self.sess.run([self.optimiser_base, self.optimiser_3], feed_dict={self.reward_base: reward_base_to,
											     self.reward_3: reward_task_to, 
											     self.base_input: item[0],
											     self.decision_type: decision_type_to,
											     self.task_input: self.to_qval_base_index[index]})
			for sym_state_index in sym3:
				print('We go')
				self.symmetry(item[0],sym_state_index)
				sym_reward_base_to, sym_reward_task_to = self.reward_function(game_type,winner,item[2],self.to_qval_base_index[index], decision_type_to, self.symmetry_index, task_classes)
				
				self.sess.run([self.optimiser_base, self.optimiser_3], feed_dict={self.reward_base: sym_reward_base_to,
												self.reward_3: sym_reward_task_to,
												self.base_input: self.symmetry_index,
								   				self.decision_type: decision_type_to,
												self.task_input: self.to_qval_base_index[index]})
		for index, item in enumerate(self.from_index):
			if None in item:
				break
			reward_base_from, reward_task_from = self.reward_function(game_type,winner,item[2],self.from_qval_base_index[index], decision_type_from, self.symmetry_index, task_classes)
			self.sess.run([self.optimiser_base, self.optimiser_3], feed_dict={self.reward_base: reward_base_from,
											     self.reward_3: reward_task_from, 
											     self.base_input: item[0],
											     self.decision_type: decision_type_from,
											     self.task_input: self.from_qval_base_index[index]})
			for sym_state_index in sym3:
				print('We goin')
				self.symmetry(item[0],sym_state_index)
				sym_reward_base_from, sym_reward_task_from = self.reward_function(game_type,winner,item[2],self.from_qval_base_index[index], decision_type_from, self.symmetry_index, task_classes)
				self.sess.run([self.optimiser_base, self.optimiser_3], feed_dict={self.reward_base: sym_reward_base_from,
												self.reward_3: sym_reward_task_from,
												self.base_input: self.symmetry_index,
								   				self.decision_type: decision_type_from,
												self.task_input: self.from_qval_base_index[index]})
				
		for index, item in enumerate(self.remove_index):
			if None in item:
				break
			reward_base_remove, reward_task_remove = self.reward_function(game_type,winner,item[2],self.remove_qval_base_index[index], decision_type_remove, self.symmetry_index, task_classes)
			self.sess.run([self.optimiser_base, self.optimiser_3], feed_dict={self.reward_base: reward_base_remove,
											     self.reward_3: reward_task_remove,
											     self.base_input: item[0],
											     self.decision_type: decision_type_remove,
											     self.task_input: self.remove_qval_base_index[index]})
			for sym_state_index in sym3:
				print('We gonna go again')
				self.symmetry(item[0],sym_state_index)
				sym_reward_base_remove, sym_reward_task_remove = self.reward_function(game_type,winner,item[2],self.from_qval_base_index[index], decision_type_remove, self.symmetry_index, task_classes)
				self.sess.run([self.optimiser_base, self.optimiser_3], feed_dict={self.reward_base: sym_reward_base_remove,
												self.reward_3: sym_reward_task_remove,
												self.base_input: self.symmetry_index,
								   				self.decision_type: decision_type_remove,
												self.task_input: self.remove_qval_base_index[index]})
	
	
		self.to_index = [(None, None, None)] * self.limit
		self.from_index = [(None, None, None)] * (self.limit - 6)
		self.remove_index = [(None, None, None)] * 19
		
		self.to_qval_base_index = [None] * self.limit
		self.from_qval_base_index = [None] * (self.limit - 6)
		self.remove_qval_base_index = [None] * 19
		
#		self.to_task_index = [(None, None, None)] * self.limit
#		self.from_task_index = [(None, None, None)] * (self.limit - 6)
#		self.remove_task_index = [(None, None, None)] * 19
		
		self.to_qval_task_index = [None] * self.limit
		self.from_qval_task_index = [None] * (self.limit - 6)
		self.remove_qval_task_index = [None] * 19
		
		return 0
		
	def learn6(self, game_type, winner):
		counter = 0
		task_classes = 16
		
		for index, item in enumerate(self.to_index):
			if None in item:
				break
			reward_base_to, reward_task_to = self.reward_function(game_type,winner,item[2],self.to_qval_base_index[index],decision_type_to,item[0],task_classes)
			
			self.sess.run([self.optimiser_base, self.optimiser_6], feed_dict={self.reward_base: reward_base_to,
											     self.reward_6: reward_task_to, 
											     self.base_input: item[0],
											     self.game_type: game_type_input,
											     self.decision_type: decision_type_to,
											     self.task_input: self.to_qval_base_index[index]})
			for sym_state_index in sym6:
				self.symmetry(item[0],sym_state_index)
				sym_reward_base_to, sym_reward_task_to = self.reward_function(game_type,winner,item[2],self.to_qval_base_index[index], decision_type_to, self.symmetry_index, task_classes)
				
				self.sess.run([self.optimiser_base, self.optimiser_6], feed_dict={self.reward_base: sym_reward_base_to,
												self.reward_6: sym_reward_task_to,
												self.base_input: self.symmetry_index,
												self.game_type: game_type_input,
								   				self.decision_type: decision_type_to,
												self.task_input: self.to_qval_base_index[index]})
		for index, item in enumerate(self.from_index):
			if None in item:
				break
			reward_base_from, reward_task_from = self.reward_function(game_type,winner,item[2],self.from_qval_base_index[index], decision_type_from, self.symmetry_index, task_classes)
			self.sess.run([self.optimiser_base, self.optimiser_6], feed_dict={self.reward_base: reward_base_from,
											     self.reward_6: reward_task_from, 
											     self.base_input: item[0],
											     self.game_type: game_type_input,
											     self.decision_type: decision_type_from,
											     self.task_input: self.from_qval_base_index[index]})
			for sym_state_index in sym6:
				self.symmetry(item[0],sym_state_index)
				sym_reward_base_from, sym_reward_task_from = self.reward_function(game_type,winner,item[2],self.from_qval_base_index[index], decision_type_from, self.symmetry_index, task_classes)
				self.sess.run([self.optimiser_base, self.optimiser_6], feed_dict={self.reward_base: sym_reward_base_from,
												self.reward_6: sym_reward_task_from,
												self.base_input: self.symmetry_index,
												self.game_type: game_type_input,
								   				self.decision_type: decision_type_from,
												self.task_input: self.from_qval_base_index[index]})
				
		for index, item in enumerate(self.remove_index):
			if None in item:
				break
			reward_base_remove, reward_task_remove = self.reward_function(game_type,winner,item[2],self.remove_qval_base_index[index], decision_type_remove, self.symmetry_index, task_classes)
			self.sess.run([self.optimiser_base, self.optimiser_6], feed_dict={self.reward_base: reward_base_remove,
											     self.reward_6: reward_task_remove,
											     self.base_input: item[0],
											     self.game_type: game_type_input,
											     self.decision_type: decision_type_remove,
											     self.task_input: self.remove_qval_base_index[index]})
			for sym_state_index in sym6:
				self.symmetry(item[0],sym_state_index)
				sym_reward_base_remove, sym_reward_task_remove = self.reward_function(game_type,winner,item[2],self.from_qval_base_index[index], decision_type_remove, self.symmetry_index, task_classes)
				self.sess.run([self.optimiser_base, self.optimiser_6], feed_dict={self.reward_base: sym_reward_base_remove,
											     	self.reward_6: sym_reward_task_remove,
												self.base_input: self.symmetry_index,
												self.game_type: game_type_input,
								   				self.decision_type: decision_type_remove,
												self.task_input: self.remove_qval_base_index[index]})
	
	
		self.to_index = [(None, None, None)] * self.limit
		self.from_index = [(None, None, None)] * (self.limit - 6)
		self.remove_index = [(None, None, None)] * 19
		
		self.to_qval_base_index = [None] * self.limit
		self.from_qval_base_index = [None] * (self.limit - 6)
		self.remove_qval_base_index = [None] * 19
		
#		self.to_task_index = [(None, None, None)] * self.limit
#		self.from_task_index = [(None, None, None)] * (self.limit - 6)
#		self.remove_task_index = [(None, None, None)] * 19
		
		self.to_qval_task_index = [None] * self.limit
		self.from_qval_task_index = [None] * (self.limit - 6)
		self.remove_qval_task_index = [None] * 19
		
		return 0
