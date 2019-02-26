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

class Learned_Player(object):
	
	def __init__(self, epsilon, alpha, gamma, limit):

		self.sess = tf.Session()
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.limit = limit
		self.state_index = []
	
		self.to_index = [(None, None, None)] * self.limit
		self.from_index = [(None, None, None)] * (self.limit - 6)
		self.remove_index = [(None, None, None)] * 19
		
		self.to_future_index = [None] * self.limit
		self.from_future_index = [None] * (self.limit - 6)
		self.remove_future_index = [None] * 19
		
		self.to_qval_index = [None] * self.limit
		self.from_qval_index = [None] * (self.limit - 6)
		self.remove_qval_index = [None] * 19

		self.n_classes = 24
		self.n_input = 79
		self.n_nodes_1 = self.n_classes * 2
		self.n_nodes_2 = self.n_classes * 2
		self.n_nodes_3 = self.n_classes * 2
		self.n_nodes_4 = self.n_classes * 2
		self.future_steps = 0
		self.symmetry_index = [None] * self.n_classes
		self.symmetry_future_index = [None] * self.n_classes
		self.piece_adj_list = [None] * 12

		self.input = tf.placeholder(tf.float32, [24])
		self.x_p1 = tf.cast(tf.equal(self.input, 1), tf.float32)
		self.x_p2 = tf.cast(tf.equal(self.input, 2), tf.float32)
		self.x_empty = tf.cast(tf.equal(self.input, 0), tf.float32)
		
		#game_type = 1 at 0 if game_type = 3, 1 if 6, 2 if 9, 3 if 12
		self.game_type = tf.placeholder(tf.float32, [4])
#		self.game_type_list = [self.game_3,self.game_6,self.game_9,self.game_12]
#		self.x_game_type = tf.reshape(self.game_type, shape=[1,4])
		
		#decision_type = 1 at 0 if place, 1 if choose piece to move, 2 if move piece to, 3 if remove piece
		self.decision_type = tf.placeholder(tf.float32, shape=[3])
		
		self.ttemp = [self.x_empty,self.x_p1,self.x_p2]
#		self.tempp = [self.game_type,self.decision_type]
		self.tempp = tf.concat([self.game_type, self.decision_type], 0)
		self.tttemp = tf.reshape(self.ttemp, shape=[72])
		self.temppp = tf.reshape(self.tempp, shape=[7])
		self.x_bin = tf.concat([self.tttemp, self.temppp], 0)
		self.x = tf.reshape(self.x_bin, shape=[1,self.n_input])
		self.reward = tf.placeholder(tf.float32,[self.n_classes])
		self.y = tf.reshape(self.reward, [1, self.n_classes])
		self.Q_val = self.neural_network()
#		self.Q_val_from = self.neural_network_from()
		self.Q_val_stored = tf.placeholder(tf.float32, shape=[self.n_classes])
		#cost
		#        self.cost = tf.reduce_mean(tf.square(self.y - self.Q_val))
		#        self.cost = tf.square(self.Q_val - self.y)
		self.cost = tf.reduce_mean(tf.squared_difference(self.y, self.Q_val))
#		self.cost = tf.square(self.y - self.Q_val_stored)
#		self.cost_from = tf.square(self.y - self.Q_val_from)
		#optimiser

# 		 self.optimiser = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)
		#        self.optimiser = tf.train.AdamOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)
		self.optimiser = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost)
#		self.optimiser_from = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost_from)
		#        self.optimizer = tf.train.AdograadOptimizer(learning_rate=alpha, decay=0.9).minimize(self.cost)

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
		predictions = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_to})
		val = np.argmax(predictions[0][0])
		return val
	
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
		predictions_to = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_to})
		
		if rand <= 100*self.epsilon:
			move = self.random_place(state)
			self.to_qval_index[move_no] = predictions_to[0][0]
			self.to_index[move_no] = (deepcopy(input_state),move,player,None)
			return move
		else:
			opt_val = -float('Inf')
			for index, item in enumerate(state):
				if item != 0:
					continue
				val = predictions_to[0][0][index]
				if val > opt_val:
					opt_val = val
					move = index
			self.to_qval_index[move_no] = predictions_to[0][0]
			self.to_index[move_no] = (deepcopy(input_state),move,player)
			return move
	
	def move(self, state, game_type, pieces, player, enable_flying, move_no):
		valid_moves = self.valid_move(state, game_type, pieces)
		if len(valid_moves) == 0 and not enable_flying:
			return (25, 25)
		move = None
		piece = None
		rand = random.randint(1,100)
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		input_state = self.convert_board(state,player)
		input_state = self.padding(input_state,game_type)
		predictions_to = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_to})
		if rand <= 100*self.epsilon:
			random_move = self.random_move(state, valid_moves, enable_flying, pieces)
			predictions_from = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_from})
			self.to_index[move_no] = (deepcopy(input_state),random_move[0], player)
			self.from_index[int(move_no - (game_type * 2))] = (deepcopy(input_state),random_move[1],player)
			self.to_qval_index[move_no] = predictions_to[0][0]
			self.from_qval_index[int(move_no - (game_type * 2))] = predictions_from[0][0]
#			print('Random move = ' + str(random_move))
			return random_move
		else:
			opt_val = -float('Inf')
			if enable_flying:
				adj_piece_list = pieces
				for index, item in enumerate(state):
					if item != 0:
						continue
					val = predictions_to[0][0][index]
#					print('Index, Val ' +str(index) + ' ' + str(val))
					if val > opt_val:
						opt_val = val
						move = index
			else:
				for index, item in enumerate(state):
					if item != 0:
#						print('We skip' + str(index))
						continue
					
					val = predictions_to[0][0][index]
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
			
			predictions_from = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_from})
			
			opt_val = -float('Inf')
#			print('Adj Pieces ' +str(adj_piece_list))
			for item in adj_piece_list:
				if item is None:
					continue
#				print('Alright here we go ' + str(item))
				val = predictions_from[0][0][item]
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
		self.to_index[move_no] = (deepcopy(input_state),move,player)
		self.from_index[int(move_no - (game_type * 2))] = (deepcopy(input_state),piece,player)
		self.to_qval_index[move_no] = predictions_to[0][0]
		self.from_qval_index[int(move_no - (game_type * 2))] = predictions_from[0][0]
#		if enable_flying:
#			print('PRED MOVE ' + str(predicted_move))
		return predicted_move
	
	def free_space_finder(self, state):
		free_space = []
		for item in state:
			if item == 0:
				free_space.append(item)

		return free_space
	
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
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		input_state = self.convert_board(state,player)
		input_state = self.padding(input_state,game_type)
		predictions_remove = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_remove})
		if rand <= 100*self.epsilon:
			piece = self.random_remove_piece(piece_list)
			self.remove_index[pieces_removed] = (deepcopy(input_state),piece,player)
			self.remove_qval_index[pieces_removed] = predictions_remove[0][0]
			return piece
		else:
			opt_val = -float('Inf')
			for index, item in enumerate(state):
				if item != opponent:
					continue
				val = predictions_remove[0][0][index]
				if val > opt_val:
					opt_val = val
					piece = index
			self.remove_index[pieces_removed] = (deepcopy(input_state),piece,player)
			self.remove_qval_index[pieces_removed] = predictions_remove[0][0]
		return piece
	
	def reward_function(self, game_type, winner, player, qval_index, decision_type, input_state, game_type_input, future_state):
		predictions = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type})
		if winner == player:
			reward = [1] * self.n_classes
		elif winner != 0:
			reward =  [-1] * self.n_classes
		else:
			reward = [0] * self.n_classes
		reward = list(map(sum, zip((predictions[0][0]),reward)))
		
		for item in reward:
			for i in range(self.future_steps):
				reward[item] += self.gamma**(i+1) * self.max_next_Q(future_state, game_type, player, decision)
			
		return reward
	
	def symmetry(self, state, sym_box, reward, decision_type):
		for index, item in enumerate(state):
			if index == len(sym_box):
				break
			temp = sym_box[index]
			self.symmetry_index[index] = state[temp]
		if decision_type == [1,0,0]:
			self.symmetry_future_index[index] = self.to_future_index[temp]
		elif decision_type == [0,1,0]:
			self.symmetry_future_index[index] = self.from_future_index[temp]
		else:
			self.symmetry_future_index[index] = self.remove_future_index[temp]
				
		
	def edit_to_index(self,state,move_no):
		self.to_future_index[move_no] = deepcopy(state)
		
	def edit_from_index(self,state,move_no,game_type):
		self.from_future_index[move_no-(game_type*2)] = deepcopy(state)
		
	def edit_remove_index(self,state,pieces_removed):
		self.remove_future_index[pieces_removed] = deepcopy(state)
	
	def learn(self, game_type, winner):
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		counter = 0
		if game_type == 3:
			sym_list = sym3
		elif game_type == 6:
			sym_list = sym6
		else:
			sym_list = sym9
		
		for index, item in enumerate(self.to_index):
			if None in item:
				if index != 0:
					print('LEARN1 ' + str(index))
#					print(self.to_index)
					break
			reward_to = self.reward_function(game_type,winner,item[2],self.to_qval_index[index], decision_type_to, item[0], game_type_input, self.to_future_index[index])
			self.sess.run([self.optimiser], feed_dict={self.reward: reward_to, self.input: item[0], self.game_type: game_type_input,
								   self.decision_type: decision_type_to})
			for sym_state_index in sym_list:
				self.symmetry(item[0],sym_state_index,reward_to, decision_type_to)
				sym_reward_to = self.reward_function(game_type,winner,item[2],self.to_qval_index[index], decision_type_to, self.symmetry_index, game_type_input, self.to_future_index[index])
#				predictions_sym = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
#										   self.decision_type: decision_type_to})
				self.sess.run([self.optimiser], feed_dict={self.reward: sym_reward_to, self.input: self.symmetry_index, self.game_type: game_type_input,
								   self.decision_type: decision_type_to})
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.place_qval_index})
		for index, item in enumerate(self.from_index):
			if None in item:
				print(self.from_index)
				break
			reward_from = self.reward_function(game_type,winner,item[2],self.from_qval_index[index], decision_type_from, item[0], game_type_input, self.from_future_index[index]) 
			self.sess.run([self.optimiser], feed_dict={self.reward: reward_from, self.input: self.symmetry_index, self.game_type: game_type_input,
								   self.decision_type: decision_type_from})
			for sym_state_index in sym_list:
				self.symmetry(item[0],sym_state_index, reward_from, decision_type_from)
				sym_reward_from = self.reward_function(game_type,winner,item[2],self.to_qval_index[index], decision_type_from, self.symmetry_index, game_type_input, self.from_future_index[index])
				self.sess.run([self.optimiser], feed_dict={self.reward: sym_reward_from, self.input: self.symmetry_index, self.game_type: game_type_input,
								   self.decision_type: decision_type_from})
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.choose_qval_index})
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.move_qval_index})
		for index, item in enumerate(self.remove_index):
			if None in item:
				break
			reward_remove = self.reward_function(game_type,winner,item[2],self.remove_qval_index[index], decision_type_remove, item[0],  game_type_input, self.remove_future_index[index])
			self.sess.run([self.optimiser], feed_dict={self.reward: reward_remove, self.input: item[0], self.game_type: game_type_input,
								   self.decision_type: decision_type_remove})
			for sym_state_index in sym_list:
				self.symmetry(item[0],sym_state_index,reward_remove, decision_type_remove)
				sym_reward_remove = self.reward_function(game_type,winner,item[2],self.to_qval_index[index], decision_type_remove, self.symmetry_index, game_type_input, self.remove_future_index[index])
				self.sess.run([self.optimiser], feed_dict={self.reward: sym_reward_remove, self.input: self.symmetry_index, self.game_type: game_type_input,
								   self.decision_type: decision_type_remove})
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.place_remove_index})
			
			
		self.to_index = [(None, None, None)] * self.limit
		self.from_index = [(None, None, None)] * (self.limit - 6)
		self.remove_index = [(None, None, None)] * 19
		
		self.to_qval_index = [None] * self.limit
		self.from_qval_index = [None] * (self.limit - 6)
		self.remove_qval_index = [None] * 19
		
		return 0
