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

decision_type_to = [1,0,0]
decision_type_from = [0,1,0]
decision_type_remove = [0,0,1]

input_index_3_1 = [0,1,3,4]
input_index_3_2 = [2,1,5,4]
input_index_3_3 = [4,3,7,6]
input_index_3_4 = [4,5,7,8]

input_index_6_1 = [0,1,3,4,6,7]
input_index_6_2 = [2,1,4,5,9,8]
input_index_6_3 = [13,14,10,11,6,7]
input_index_6_4 = [15,14,12,11,9,8]

input_index_9_1 = [0,1,3,4,6,7,9,10,11]
input_index_9_2 = [2,1,5,4,8,7,14,13,12]
input_index_9_3 = [21,22,18,19,15,16,9,10,11]
input_index_9_4 = [23,22,20,19,17,16,14,13,12]

input_index_3 = [input_index_3_1, input_index_3_2, input_index_3_3, input_index_3_4]
input_index_6 = [input_index_6_1, input_index_6_2, input_index_6_3, input_index_6_4]
input_index_9 = [input_index_9_1, input_index_9_2, input_index_9_3, input_index_9_4]


class Shortened_Player(object):

	def __init__(self, epsilon, alpha, gamma, limit):

		self.sess = tf.Session()
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.limit = limit
		
		self.to_index = [(None, None, None)] * self.limit
		self.from_index = [(None, None, None)] * (self.limit - 6)
		self.remove_index = [(None, None, None)] * 19
		
		self.to_future_index = [None] * self.limit
		self.from_future_index = [None] * (self.limit - 6)
		self.remove_future_index = [None] * 19
		
		self.n_classes = 9
		
		self.input_index = [[None]*9,[None]*9,[None]*9,[None]*9]

		self.n_input = 34
		self.n_nodes_1 = self.n_classes * 2
		self.n_nodes_2 = self.n_classes * 2
		self.n_nodes_3 = self.n_classes * 2
		self.n_nodes_4 = self.n_classes * 2
		self.future_steps = 0
		
		self.symmetry_index = [None] * self.n_classes
		self.symmetry_future_index = [None] * self.n_classes
		self.piece_adj_list = [None] * 12
		
		self.input = tf.placeholder(tf.float32, [self.n_classes])
		self.x_p1 = tf.cast(tf.equal(self.input, 1), tf.float32)
		self.x_p2 = tf.cast(tf.equal(self.input, 2), tf.float32)
		self.x_empty = tf.cast(tf.equal(self.input, 0), tf.float32)
		
		#game_type = 1 at 0 if game_type = 3, 1 if 6, 2 if 9, 3 if 12
		self.game_type = tf.placeholder(tf.float32, shape=[4])
		
		#decision_type = 1 at 0 if place, 1 if choose piece to move, 2 if move piece to, 3 if remove piece
		self.decision_type = tf.placeholder(tf.float32, shape=[3])
		
		self.ttemp = [self.x_empty,self.x_p1,self.x_p2]
		self.tempp = tf.concat([self.game_type, self.decision_type], 0)
		self.tttemp = tf.reshape(self.ttemp, shape=[self.n_classes*3])
		self.temppp = tf.reshape(self.tempp, shape=[7])
		self.x_bin = tf.concat([self.tttemp, self.temppp], 0)
		self.x = tf.reshape(self.x_bin, shape=[1,self.n_input])
		self.reward = tf.placeholder(tf.float32,[self.n_classes])
		self.y = tf.reshape(self.reward, [1, self.n_classes])
		self.Q_val = self.neural_network()
		
		#cost
		self.cost = tf.reduce_mean(tf.squared_difference(self.y, self.Q_val))
		
		#optimiser
		self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost)
		
	def neural_network(self):

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

#		l4 = tf.layers.dense(
	#		inputs=l3,
	#		units=self.n_nodes_3,
	#		bias_initializer=tf.constant_initializer(0, 1),
	#		activation=tf.nn.leaky_relu,
	#		kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
	#		activity_regularizer=tf.nn.softmax
#		)

#		l5 = tf.layers.dense(
	#		inputs=l4,
	#		units=self.n_nodes_4,
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
	
	def padding(self,state,game_type):
		if game_type > 6:
			return state
		temp = deepcopy(state)
		if game_type == 3:
			temp.extend([0]*5)
		else:
			temp.extend([0]*3)
		return temp
	
	def convert_board(self, state, player):
		if player == 1:
			return state
		else:
			new_state = deepcopy(state)
			for item in new_state:
				item = (item % 2) + 1
			return new_state
		
	def random_place(self, state):
		space_val = 1
		while space_val != 0:
			space = random.randint(0, len(state) - 1)
			space_val = state[space]
		return space
	
	def board_to_input(self, state, game_type, decision_type):
		if game_type == 3:
			for index, item in enumerate(state):
				if index >= len(input_index_3):
					self.input_index[index] = 0
				else:
					temp = input_index_3[index]
					self.input_index[index] = state[temp]
		elif game_type == 6:
			for index, item in enumerate(state):
				if index >= len(input_index_6):
					self.input_index[index] = 0
				else:
					temp = input_index_6[index]
					self.input_index[index] = state[item]
					
		else:
			for index, item in enumerate(state):
				temp = input_index_9[index]
				self.input_index[index] = state[item]
			
			
			
	def place(self, state, game_type, player, move_no):
		rand = random.randint(1,100)
		move = None
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		input_state = self.convert_board(state,player)
		
		
#		predictions_to = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
#										   self.decision_type: decision_type_to})
		if rand <= 100*self.epsilon:
			move = self.random_place(state)
			self.to_qval_index[move_no] = predictions_to[0][0]
			self.to_index[move_no] = (deepcopy(input_state),move,player,None)
			return move
