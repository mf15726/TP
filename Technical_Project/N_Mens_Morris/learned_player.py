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
import operator


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

class Learned_Player(object):
	
	def __init__(self, epsilon, alpha, gamma):

		self.sess = tf.Session()
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.state_index = []
		
		self.place_index = []
		self.choose_index = []
		self.move_index = []
		self.remove_index = []
		
		self.place_qval_index = []
		self.choose_qval_index = []
		self.move_qval_index = []
		self.remove_qval_index = []

		self.n_classes = 24
		self.n_input = 80
		self.n_nodes_1 = self.n_classes * 2
		self.n_nodes_2 = self.n_classes * 2
		self.n_nodes_3 = self.n_classes * 2
		self.n_nodes_4 = self.n_classes * 2

		self.input = tf.placeholder(tf.float32, [24])
		self.x_p1 = tf.cast(tf.equal(self.input, 1), tf.float32)
		self.x_p2 = tf.cast(tf.equal(self.input, 2), tf.float32)
		self.x_empty = tf.cast(tf.equal(self.input, 0), tf.float32)
		
		#game_type = 1 at 0 if game_type = 3, 1 if 6, 2 if 9, 3 if 12
		self.game_type = tf.placeholder(tf.float32, [4])
#		self.game_3 = tf.cast(tf.equal(self.game_type, 3), tf.float32)
#		self.game_6 = tf.cast(tf.equal(self.game_type, 6), tf.float32)
#		self.game_9 = tf.cast(tf.equal(self.game_type, 9), tf.float32)
#		self.game_12 = tf.cast(tf.equal(self.game_type, 12), tf.float32)
#		self.game_type_list = [self.game_3,self.game_6,self.game_9,self.game_12]
#		self.x_game_type = tf.reshape(self.game_type, shape=[1,4])
		
		#decision_type = 1 at 0 if place, 1 if choose piece to move, 2 if move piece to, 3 if remove piece
		self.decision_type = tf.placeholder(tf.float32, shape=[4])
#		self.decision_place = tf.cast(tf.equal(self.decision_type, 0), tf.float32)
#		self.decision_move_to = tf.cast(tf.equal(self.decision_type, 1), tf.float32)
#		self.decision_move_from = tf.cast(tf.equal(self.decision_type, 2), tf.float32)
#		self.decision_take = tf.cast(tf.equal(self.game_type, 3), tf.float32)
#		self.decision_list = [self.decision_place,self.decision_move_to,self.decision_move_from,self.decision_take]
#		self.x_decision_type = tf.reshape(self.decision_type, shape=[1,4])
		
#		self.x_bin = [self.x_empty,self.x_p1,self.x_p2]
#		self.x_bin = [self.x_empty,self.x_p1,self.x_p2,self.x_game_type,self.x_decision_type]
		self.ttemp = [self.x_empty,self.x_p1,self.x_p2]
		self.tempp = [self.game_type,self.decision_type]
		self.tttemp = tf.reshape(self.ttemp, shape=[72])
		self.temppp = tf.reshape(self.tempp, shape=[8])
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
		self.cost = tf.square(self.y - self.Q_val)
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
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
			activity_regularizer=tf.nn.softmax
		)

		l2 = tf.layers.dense(
			inputs=l1,
			units=self.n_nodes_1,
			kernel_initializer = tf.constant_initializer(0,1),
			bias_initializer=tf.constant_initializer(0, 1),
			activation=tf.nn.leaky_relu,
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
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
			kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
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
	
	def padding(self,state,game_type):
		temp = deepcopy(state)
		if game_type > 6:
			return temp
		if game_type == 3:
			temp.extend([0]*15)
		else:
			temp.extend([0]*8)
		return temp
		
	def place(self, state, free_space, game_type, player):
		rand = random.randint(1,100)
		move = None
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		decision_type_place = [1,0,0,0]
		input_state = self.padding(state,game_type)
		predictions_place = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_place})
		
		if rand <= 100*self.epsilon:
			move = self.random_place(state,free_space)
			self.place_qval_index.append(predictions_place[0][0])
			self.place_index.append((deepcopy(state),move,player))
			return move
		else:
			opt_val = -float('Inf')
			for index, val in enumerate(predictions_place[0][0]):
				if val > opt_val and index in free_space:
					opt_val = val
					move = index
				if index == len(state):
					break
#			predictions_place.to_list()
			self.place_qval_index.append(predictions_place[0][0])
			self.place_index.append((deepcopy(state),move,player))
			return move
	
	def random_move(self, valid_moves):
		temp = random.randint(0, len(valid_moves) - 1)
		return valid_moves[temp]
	
	
	def move(self, state, game_type, free_space, pieces, player, enable_flying):
		valid_moves = self.valid_move(state, game_type, free_space, pieces)
		if len(valid_moves) == 0 and not enable_flying:
			return (25, 25)
		move = None
		piecee = None
		rand = random.randint(1,100)
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		decision_type_choose = [0,1,0,0]
		decision_type_move = [0,0,1,0]
		input_state = self.padding(state,game_type)
		predictions_choose = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_choose})
		if rand <= 100*self.epsilon:
			random_move = self.random_move(valid_moves)
			predictions_move = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_move})
			self.choose_index.append((deepcopy(state),random_move[0], player))
			self.move_index.append((deepcopy(state),random_move[1],player))
			self.choose_qval_index.append(predictions_choose[0][0])
			self.move_qval_index.append(predictions_move[0][0])
			return random_move
		else:
			opt_val = -float('Inf')
			for index, val in enumerate(predictions_choose[0][0]):
				if val > opt_val and index in pieces:
					for item in valid_moves:
						if index == item[0] and item[1] in free_space:
							opt_val = val
							piece = index
							continue
					if index == len(state):
						break
			if piece == None:
				return (25,25)
			valid_spaces = []
			if enable_flying:
				valid_spaces = deepcopy(free_space)
			else:
				for item in valid_moves:
					if piece == item[0]:
						valid_spaces.append(item[1])
			predictions_move = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_move})
#			print('Valid space = '+ str(valid_spaces)
			opt_val = -float('Inf')
			for index, val in enumerate(predictions_move[0][0]):
				if val > opt_val and index in valid_spaces:
					opt_val = val
					move = index
					if index == len(state):
						break
			if move == None:
				return(25,25)
					
			predicted_move = (piece, move)
		self.choose_index.append((deepcopy(state),piece,player))
		self.move_index.append((deepcopy(state),move,player))
		self.choose_qval_index.append(predictions_choose[0][0])
		self.move_qval_index.append(predictions_move[0][0])
		return predicted_move
	
	def remove_piece(self, state, piece_list, game_type, player):
		rand = random.randint(1,100)
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		decision_type_remove = [0,0,0,1]
		input_state = self.padding(state,game_type)
		predictions_remove = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_remove})
		if rand <= 100*self.epsilon:
			temp = random.randint(0, len(piece_list) - 1)
			self.remove_index.append((deepcopy(state),piece_list[temp],player))
			self.remove_qval_index.append(predictions_remove[0][0])
			return piece_list[temp]
		else:
			opt_val = -float('Inf')
			for index, val in enumerate(predictions_remove[0][0]):
				if val > opt_val and index in piece_list:
					opt_val = val
					piece = index
					if index == len(state):
						break
			self.remove_index.append((deepcopy(state),piece,player))
			self.remove_qval_index.append(predictions_remove[0][0])
		return piece
	
	def reward_function(self,game_type, winner, player, qval_index):
		if winner == player:
			reward = [1] * self.n_classes
		elif winner != 0:
			reward =  [-1] * self.n_classes
		else:
			reward = [0] * self.n_classes
		return list(map(operator.add, qval_index,reward))
	
	def learn(self, game_type, winner):
		input_state = self.padding(self.place_index[0][0],game_type)
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		decision_type_place = [1,0,0,0]
		decision_type_choose = [0,1,0,0]
		decision_type_move = [0,0,1,0]
		decision_type_remove = [0,0,0,1]
		counter = 0
		for item in self.place_index:
			reward_place = self.reward_function(game_type,winner,item[2],self.place_qval_index[counter])
			self.sess.run([self.optimiser], feed_dict={self.reward: reward_place, self.input: input_state, self.game_type: game_type_input,
								   self.decision_type: decision_type_place})
			counter += 1 
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.place_qval_index})
		counter = 0
		for item in self.choose_index:
			reward_choose = self.reward_function(game_type,winner,item[2],self.choose_qval_index[counter]) 
			reward_move =  self.reward_function(game_type,winner,item[2],self.move_qval_index[counter])
			self.sess.run([self.optimiser], feed_dict={self.reward: reward_choose, self.input: input_state, self.game_type: game_type_input,
								   self.decision_type: decision_type_choose})
			self.sess.run([self.optimiser], feed_dict={self.reward: reward_move, self.input: input_state, self.game_type: game_type_input,
								   self.decision_type: decision_type_move})
			counter += 1
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.choose_qval_index})
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.move_qval_index})
		counter = 0
		for item in self.remove_index:
			reward_remove = self.reward_function(game_type,winner,item[2],self.remove_qval_index[counter])
			self.sess.run([self.optimiser], feed_dict={self.reward: reward_remove, self.input: input_state, self.game_type: game_type_input,
								   self.decision_type: decision_type_remove})
			counter += 1
#			self.sess.run([self.optimiser], feed_dict={self.reward: reward, self.Q_val_stored: self.place_remove_index})
			
			
		
		
		self.place_index = []
		self.choose_index = []
		self.move_index = []
		self.remove_index = []
		
		self.place_qval_index = []
		self.choose_qval_index = []
		self.move_qval_index = []
		self.remove_qval_index = []
