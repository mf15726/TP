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
		
		self.to_qval_index = [None] * self.limit
		self.from_qval_index = [None] * (self.limit - 6)
		self.remove_qval_index = [None] * 19
		
		self.to_future_qval_index = [[0 for x in range(24)] for i in range(self.limit)]
		self.from_future_qval_index = [[0 for x in range(24)] for i in range(self.limit-6)]
		self.remove_future_qval_index = [[0 for x in range(24)] for i in range(19)]
    

		self.sym_qval_index = [0] * 24
		self.temp_qval_index = [0] * 24
		self.symmetry_index = [0] * self.n_classes
		self.symmetry_future_index = [0] * self.n_classes
		self.piece_adj_list = [None] * 12
		
		self.to_index = [(None, None, None),(None, None, None), (None, None, None), (None, None, None)] * self.limit
		self.from_index = [(None, None, None), (None, None, None), (None, None, None), (None, None, None)] * (self.limit - 6)
		self.remove_index = [(None, None, None), (None, None, None), (None, None, None), (None, None, None)] * 19
		
		self.to_future_index = [None] * self.limit
		self.from_future_index = [None] * (self.limit - 6)
		self.remove_future_index = [None] * 19
		
		self.to_qval_index = [None, None, None, None] * self.limit
		self.from_qval_index = [None, None, None, None] * (self.limit - 6)
		self.remove_qval_index = [None, None, None, None] * 19
		
		self.n_classes = 9
		
		self.input_index = [[0]*24,[0]*24,[0]*24,[0]*24]
		self.sym_qval_index = [0] * 24
		self.temp_qval_index = [0] * 24
		
		self.n_input = 34
		self.n_nodes_1 = self.n_classes * 2
		self.n_nodes_2 = self.n_classes * 2
		self.n_nodes_3 = self.n_classes * 2
		self.n_nodes_4 = self.n_classes * 2
		self.future_steps = 1
		
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
#		self.optimiser = tf.train.RMSPropOptimizer(learning_rate=alpha,decay=0.9).minimize(self.loss)
		
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
	
	def find_space(self,game_type,space):
		space_index = [None] * 4
		if game_type == 3:
			for index, item in enumerate(input_index_3):
				if space in item:
					space_index[index] = input_index_3.index(space)
		elif game_type == 6:
			for index, item in enumerate(input_index_6):
				if space in item:
					space_index[index] = input_index_6.index(space)
		else:
			for index, item in enumerate(input_index_9):
				if space in item:
					space_index[index] = input_index_9.index(space)
		return space_index
					
	
	def q_reward(self,state,game_type_input,move,decision,index,future_qval_index):
		new_state = self.convert_board(state, 2)
		self.board_to_input(input_state, game_type, decision_type)
		predictions = self.sess.run([self.Q_val], feed_dict={self.input: state, self.game_type: game_type_input,
									   self.decision_type: decision})
		value = np.amin(predictions[0][0])
#		print(value)
#		print(predictions[0][0])    
		future_qval_index[ind][move] = value
		
	def q_reward_move(self,state,game_type_input,move,decision,index,future_qval_index):
		new_state = self.convert_board(state, 2)
		predictions = self.sess.run([self.Q_val], feed_dict={self.input: new_state, self.game_type: game_type_input,
										   self.decision_type: decision})
		value = np.amin(predictions[0][0])
		if value < future_qval_index[index][move]:
			future_qval_index[index][move] = value	
	
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
			for ind, input_index enumerate(input_index_3):
				for index, item in enumerate(state):
					if index >= len(input_index_3[ind]):
						self.input_index[index] = 0
					else:
						temp = input_index_3[ind][index]
						self.input_index[ind][index] = state[temp]
		elif game_type == 6:
			for ind, input_index i enumerate(input_index_6):
				for index, item in enumerate(state):
					if index >= len(input_index_6[ind]):
						self.input_index[index] = 0
					else:
						temp = input_index_6[ind][index]
						self.input_index[ind][index] = state[temp]
					
		else:
			for index, item in enumerate(state):
				temp = input_index_9[index]
				self.input_index[index] = state[item]
			
			
	def find_move(self, game_type, input_ind, index):
		if game_type == 3:
			move = input_index_3[input_ind][index]
		elif game_type == 6:
			move = input_index_3[input_ind][index]
		else:
			move = input_index_9[input_ind][index]
		return move
			
			
	def place(self, state, game_type, player, move_no):
		rand = random.randint(1,100)
		move = None
		game_type_input = [0] * 4
		game_type_input[int((game_type/3)-1)] = 1
		input_state = self.convert_board(state,player)
		self.board_to_input(input_state, game_type, decision_type)
		opt_val = -float('Inf')
		for ind, mini_state in enumerate(self.input_index):
			predictions_to = self.sess.run([self.Q_val], feed_dict={self.input: mini_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_to})
			for index, item in enumerate(mini_state):
				if item != 0:
					continue
				space = self.find_move(game_type,ind,index)
				input_state[space] = 1
				self.board_to_input(input_state, game_type, decision_type)
				self.q_reward(input_state,game_type_input,index,decision_type_to,move_no,self.to_future_qval_index)
				input_state[index] = 0
				val = predictions_to[0][0][index]
				if val > opt_val:
					opt_val = val
					move_index = index
					input_ind = ind
			self.to_qval_index[move_no][ind] = predictions_to[0][0]
			self.to_index[move_no][ind] = ((deepcopy(input_state),move,player))
			self.board_to_input(input_state, game_type, decision_type)
		if rand <= 100*self.epsilon:
			move = self.random_place(state)
			return move
		else:
			move = self.find_move(game_type, input_ind, move_index)
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
		self.board_to_input(input_state, game_type, decision_type)
		opt_val = -float('Inf')
		if enable_flying:
			adj_piece_list = pieces
			for ind, mini_state in enumerate(self.input_index):
				predictions_to = self.sess.run([self.Q_val], feed_dict={self.input: mini_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_to})	
			for index, item in enumerate(state):
				if item != 0:
					continue
				space = self.find_move(game_type,ind,index)
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
		self.to_index[move_no] = (deepcopy(input_state),piece,player)
		self.from_index[int(move_no - (game_type * 2))] = (deepcopy(input_state),move,player)
		self.to_qval_index[move_no] = predictions_to[0][0]
		self.from_qval_index[int(move_no - (game_type * 2))] = predictions_from[0][0]
		if rand <= 100*self.epsilon:
			random_move = self.random_move(state, valid_moves, enable_flying, pieces)
			return random_move
		else:
			return predicted_move
		
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
#		print(predictions_to[0][0])
#		print(move_no)
		if rand <= 100*self.epsilon:
			random_move = self.random_move(state, valid_moves, enable_flying, pieces)
			predictions_from = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_from})
			if enable_flying:
				self.piece_adj_list = pieces
				for index, item in enumerate(state):
					if item != 0:
						continue
					input_state[index] = 1
					self.to_future_qval_index[move_no][index] = float('Inf')
					for piece in self.piece_adj_list:
						if item is None:
							continue
						input_state[piece] = 0
						self.q_reward_move(input_state,game_type_input,index,decision_type_to,move_no,self.to_future_qval_index)
						input_state[piece] = 1
					input_state[index] = 0
			else:
				
				for index, item in enumerate(state):
					if item != 0:
#						print('We skip' + str(index))
						continue
					val = predictions_to[0][0][index]
					self.piece_adj(state, game_type, index, pieces, player)
					if self.piece_adj_list[0] is None:
						continue
					else:
						input_state[index] = 1
						for piece in self.piece_adj_list:
							if piece is None:
								continue
							input_state[piece] = 0
							self.q_reward_move(input_state,game_type_input,index,decision_type_to,move_no,self.to_future_qval_index)
							input_state[piece] = 1
						input_state[index] = 0
						
			self.piece_adj(state, game_type, random_move[1], pieces, player)
			for item in self.piece_adj_list:
				if item is None:
					continue
				self.q_reward_move(input_state,game_type_input,item,decision_type_from,move_no-(game_type*2),self.from_future_qval_index)
			self.to_index[move_no] = (deepcopy(input_state),random_move[0], player)
			self.from_index[int(move_no - (game_type * 2))] = (deepcopy(input_state),random_move[1],player)
			self.to_qval_index[move_no] = predictions_to[0][0]
			self.from_qval_index[int(move_no - (game_type * 2))] = predictions_from[0][0]
#			print('Random move = ' + str(random_move))
			return random_move
		else:
			opt_val = -float('Inf')
			if enable_flying:
				self.piece_adj_list = pieces
				for index, item in enumerate(state):
					if item != 0:
						continue
					input_state[index] = 1
					self.to_future_qval_index[move_no][index] = float('Inf')
					for piece in adj_piece_list:
						input_state[piece] = 0
						self.q_reward_move(input_state,game_type_input,index,decision_type_to,move_no,self.to_future_qval_index)
						input_state[piece] = 1
					input_state[index] = 0
					for item in self.piece_adj_list:
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
					self.piece_adj(state, game_type, index, pieces, player)
					if self.piece_adj_list[0] is None:
						continue
					else:
						input_state[index] = 1
						for piece in self.piece_adj_list:
							if piece is None:
								continue
							input_state[piece] = 0
							self.q_reward_move(input_state,game_type_input,index,decision_type_to,move_no,self.to_future_qval_index)
							input_state[piece] = 1
						input_state[index] = 0
					if val > opt_val:
						self.piece_adj(state, game_type, index, pieces, player)
#						print('WE HAVE SUCCESS' + str(adj_piece))
						if self.piece_adj_list[0] is None:
							continue
						else:
							opt_val = val
							move = index					
			if move is None:
				print('No move')
				return (25,25)
			
			predictions_from = self.sess.run([self.Q_val], feed_dict={self.input: input_state, self.game_type: game_type_input,
										   self.decision_type: decision_type_from})
			
			opt_val = -float('Inf')
			self.piece_adj(state, game_type, move, pieces, player)
#			print('Adj Pieces ' +str(adj_piece_list))
			for item in self.piece_adj_list:
				if item is None:
					continue
				input_state[item] = 0
				self.q_reward_move(input_state,game_type_input,item,decision_type_from,move_no-(game_type*2),self.from_future_qval_index)
				input_state[item] = 1
#				print('Alright here we go ' + str(item))
				val = predictions_from[0][0][item]
#				print('VAl = ' +str(val) + ' Opt_Val = ' +str(opt_val))
				if val > opt_val:
					opt_val = val
					piece = item
#					print('Piece is ' +str(piece))
			if piece is None:
				print(move)
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
	def edit_to_index(self,state,game_type,move_no,player):
		new_state = self.padding(state,game_type)
		new_state = self.convert_board(new_state,player)
		self.to_future_index[move_no] = deepcopy(new_state)
		
	def edit_from_index(self,state,move_no,game_type,player):
		new_state = self.padding(state,game_type)
		new_state = self.convert_board(new_state,player)
		self.from_future_index[move_no-(game_type*2)] = deepcopy(new_state)
		
	def edit_remove_index(self,state,game_type,pieces_removed,player):
		new_state = self.padding(state,game_type)
		new_state = self.convert_board(new_state,player)
		self.remove_future_index[pieces_removed] = deepcopy(new_state)
			
