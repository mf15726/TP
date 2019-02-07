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


adj_dict_3 = {
"0": [1, 3, 4],
"1": [0, 2, 4],
"2": [1, 4, 5],
"3": [0, 4, 6],
"4": [0, 1, 2, 3, 5, 6, 7, 8],
"5": [2, 4, 8],
"6": [7, 3, 4],
"7": [6, 8, 4],
"8": [7, 4, 5],
}

mill_dict_3 = {
"0": [[1, 2], [3, 6], [4, 8]],
"1": [[0, 2], [4, 7]],
"2": [[5, 8], [4, 6], [0,1]],
"3": [[0, 6], [4, 5]],
"4": [[0, 8], [1, 7], [2, 6], [3, 5]],
"5": [[2, 8], [4, 3]],
"6": [[7, 8], [2, 4], [0, 3]],
"7": [[6, 8], [1, 4]],
"8": [[7, 6], [2, 5], [0, 4]]
}

adj_dict_6 = {
"0": [1, 6],
"1": [0, 2, 4],
"2": [1, 9],
"3": [4, 7],
"4": [1, 3, 5],
"5": [4, 8],
"6": [7, 0, 13],
"7": [10, 3, 6],
"8": [12, 5, 9],
"9": [2, 8, 15],
"10": [7, 11],
"11": [12, 14, 10],
"12": [8, 11],
"13": [14, 6],
"14": [13, 15, 11],
"15": [9, 14]
}

mill_dict_6 = {
"0": [[1, 2], [13, 6]],
"1": [[0, 2]],
"2": [[0, 1], [9, 15]],
"3": [[10, 7], [4, 5]],
"4": [[3, 5]],
"5": [[12, 8], [4, 3]],
"6": [[0, 13]],
"7": [[10, 3]],
"8": [[5, 12]],
"9": [[2, 15]],
"10": [[3, 7], [11, 12]],
"11": [[10, 12]],
"12": [[10, 11], [5, 8]],
"13": [[0, 6], [14, 15]],
"14": [[13, 15]],
"15": [[2, 9], [13, 14]]
}

adj_dict_9 = {
"0": [1, 9],
"1": [0, 2, 4],
"2": [1, 14],
"3": [4, 10],
"4": [1, 3, 5, 7],
"5": [4, 13],
"6": [7, 11],
"7": [4, 6, 8],
"8": [12, 7],
"9": [0, 21, 10],
"10": [11, 18, 3, 9],
"11": [6, 15, 10],
"12": [8, 17, 13, 14],
"13": [14, 20, 5],
"14": [2, 23, 13],
"15": [11, 15],
"16": [15, 17, 19],
"17": [12, 16],
"18": [10, 19],
"19": [18, 20, 10, 22],
"20": [19, 13],
"21": [9, 22],
"22": [21, 23, 19],
"23": [22, 14]
}

mill_dict_9 = {
"0": [[1, 2], [9, 21]],
"1": [[0, 2], [7, 4]],
"2": [[0,  1], [14, 23]],
"3": [[18, 10], [4, 5]],
"4": [[3, 5], [1, 7]],
"5": [[13, 20], [4, 3]],
"6": [[11, 15], [7, 8]],
"7": [[6, 8], [1, 4]],
"8": [[12, 17], [6, 7]],
"9": [[10, 11], [0, 21]],
"10": [[9, 11], [3, 18]],
"11": [[6, 15], [9, 11]],
"12": [[8, 17], [13, 14]],
"13": [[12, 14], [5, 20]],
"14": [[2, 23], [12, 13]],
"15": [[6, 11], [16, 17]],
"16": [[15, 17], [19, 22]],
"17": [[12, 8], [15, 16]],
"18": [[10, 3], [19, 20]],
"19": [[18, 20], [10, 3]],
"20": [[19, 18], [5, 13]],
"21": [[9, 0], [22, 23]],
"22": [[21, 23], [16, 19]],
"23": [[21, 22], [2, 14]]
}

adj_dict_12 = {
"0": [1, 9, 3],
"1": [0, 2, 4],
"2": [1, 14, 5],
"3": [4, 10, 0, 6],
"4": [1, 3, 5, 7],
"5": [4, 13, 2, 8],
"6": [7, 11, 3],
"7": [4, 6, 8],
"8": [12, 7, 5],
"9": [0, 21, 10],
"10": [11, 18, 3],
"11": [6, 15],
"12": [8, 17],
"13": [14, 20, 5],
"14": [2, 23, 13],
"15": [11, 15, 18],
"16": [15, 17, 19],
"17": [12, 16, 20],
"18": [10, 18, 15, 21],
"19": [18, 20, 10, 22],
"20": [19, 13, 17, 20],
"21": [9, 22, 18],
"22": [21, 23, 19],
"23": [22, 14, 20]
}

mill_dict_12 = {
"0": [[1, 2], [9, 21], [3, 6]],
"1": [[0, 2], [7, 4]],
"2": [[0, 1], [14, 23], [5, 8]],
"3": [[18, 10], [4, 5], [0, 6]],
"4": [[3, 5], [1, 7]],
"5": [[13, 20], [4, 3], [2, 8]],
"6": [[11, 15], [7, 8], 9],
"7": [[6, 8], [1, 4]],
"8": [[12, 17], [6, 7], 20],
"9": [[10, 11], [0, 21]],
"10": [[9, 11], [3, 18]],
"11": [[6, 15], [9, 11]],
"12": [[8, 17], [13, 14]],
"13": [[12, 14], [5, 20]],
"14": [[2, 23], [12, 13]],
"15": [[6, 11], [16, 17], [18, 21]],
"16": [[15, 17], [19, 22]],
"17": [[12, 8], [15, 16], [20, 23]],
"18": [[10, 3], [19, 20], [15, 21]],
"19": [[18, 20], [10, 3]],
"20": [[19, 18], [5, 13], [17, 23]],
"21": [[9, 0], [22, 23], [15, 18]],
"22": [[21, 23], [16, 19]],
"23": [[21, 22], [2, 14], [17, 20]]
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
		self.Q_val_place = self.neural_network_place()
		self.Q_val_from = self.neural_network_from()

		#cost
		#        self.cost = tf.reduce_mean(tf.square(self.y - self.Q_val))
		#        self.cost = tf.square(self.Q_val - self.y)
		self.cost_place = tf.square(self.y - self.Q_val_place)
		self.cost_from = tf.square(self.y - self.Q_val_from)
		#optimiser

# 		 self.optimiser = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)
		#        self.optimiser = tf.train.AdamOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)
		self.optimiser_place = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost_place)
		self.optimiser_from = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost_from)
		#        self.optimizer = tf.train.AdograadOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)
		self.regularizer1 = tf.contrib.layers.l1_regularizer(scale=0.1)
		self.regularizer2 = tf.contrib.layers.l2_regularizer(scale=0.1)

	def neural_network_place(self):

		l1 = tf.layers.dense(
			inputs=self.x,
			units=self.n_input,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
			activity_regularizer=tf.nn.softmax,
			kernal_regularizer=self.regularizer1
		)

		l2 = tf.layers.dense(
			inputs=l1,
			units=self.n_input,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
			activity_regularizer=tf.nn.softmax,
			kernal_regularizer=self.regularizer1
		)

#		l3 = tf.layers.dense(
	#		inputs=l2,
	#		units=self.n_nodes_1,
	#		bias_initializer=tf.constant_initializer(0, 1),
	#		activation=tf.nn.leaky_relu,
	#		activity_regularizer=tf.nn.softmax,
	#		kernal_regularizer=self.regularizer1
#		)

#		l4 = tf.layers.dense(
	#		inputs=l3,
	#		units=self.n_nodes_2,
	#		bias_initializer=tf.constant_initializer(0, 1),
	#		activation=tf.nn.leaky_relu,
	#		activity_regularizer=tf.nn.softmax,
	#		kernal_regularizer=self.regularizer1
#		)

#		l5 = tf.layers.dense(
	#		inputs=l4,
	#		units=self.n_nodes_3,
	#		bias_initializer=tf.constant_initializer(0, 1),
	#		activation=tf.nn.leaky_relu,
	#		activity_regularizer=tf.nn.softmax,
	#		kernal_regularizer=self.regularizer1
#		)

		l_out = tf.layers.dense(
			inputs=l2,
			units=self.n_input,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
			activity_regularizer=tf.nn.softmax,
			kernal_regularizer=self.regularizer1
		)

		l_norm = tf.contrib.layers.softmax(
			logits=l_out
		)
		
		return l_norm
		
		
	def neural_network_from(self):

		l1 = tf.layers.dense(
			inputs=self.x,
			units=self.n_input,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
			activity_regularizer=tf.nn.softmax,
			kernal_regularizer=tf.contrib.layers.l1_regularizer(0.001)
		)

		l2 = tf.layers.dense(
			inputs=l1,
			units=self.n_input,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
			activity_regularizer=tf.nn.softmax,
			kernal_regularizer=tf.contrib.layers.l1_regularizer(0.001)
		)

#		l3 = tf.layers.dense(
	#		inputs=l2,
	#		units=self.n_nodes_1,
	#		bias_initializer=tf.constant_initializer(0, 1),
	#		activation=tf.nn.leaky_relu,
	#		activity_regularizer=tf.nn.softmax,
	#		kernal_regularizer=tf.contrib.layers.l1_regularizer(0.001)
#		)

#		l4 = tf.layers.dense(
	#		inputs=l3,
	#		units=self.n_nodes_2,
	#		bias_initializer=tf.constant_initializer(0, 1),
	#		activation=tf.nn.leaky_relu,
	#		activity_regularizer=tf.nn.softmax,
	#		kernal_regularizer=tf.contrib.layers.l1_regularizer(0.001)
#		)

#		l5 = tf.layers.dense(
	#		inputs=l4,
	#		units=self.n_nodes_3,
	#		bias_initializer=tf.constant_initializer(0, 1),
	#		activation=tf.nn.leaky_relu,
	#		activity_regularizer=tf.nn.softmax,
	#		kernal_regularizer=tf.contrib.layers.l1_regularizer(0.001)
#		)

		l_out = tf.layers.dense(
			inputs=l2,
			units=self.n_input,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
			activity_regularizer=tf.nn.softmax,
			kernal_regularizer=tf.contrib.layers.l1_regularizer(0.001)
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
		
	def place(self, state, free_space, game_type):
		rand = random.randint(1,100)
		move = None
		if rand <= 100*self.epsilon:
			move = self.random_place(state)
			return move
		else:
			predictions = self.sess.run([self.Q_val_place], feed_dict={self.input: state})
		opt_val = -float('Inf')
		for index, val in enumerate(predictions[0][0]):
			if index not in state:
				continue
			if val > opt_val:
				opt_val = val
				move = index
			if index == len(state):
				break
		self.state_index.append((deepcopy(state),move))
		return move
	
	def random_move(self, valid_moves):
		temp = random.randint(0, len(valid_moves) - 1)
		return valid_moves[temp]
	
	def move(self, state, game_type, free_space, pieces):
		valid_moves = self.valid_move(state, game_type, free_space, pieces)
		if len(valid_moves) == 0:
			return (25, 25)
		move = None
		rand = random.randint(1,100)
		if rand <= 100*self.epsilon:
			move = self.random_move(valid_moves)
			return move
		else:
			predictions = self.sess.run([self.Q_val_from], feed_dict={self.input: state})
			opt_val = -float('Inf')
			for index, val in enumerate(predictions[0][0]):
				if val > opt_val:
					opt_val = val
					if game_mode != 6:
						move = node_list_9[index]
					else:
						move = node_list_6[index]
		self.state_index.append((deepcopy(state),move))
		return move
