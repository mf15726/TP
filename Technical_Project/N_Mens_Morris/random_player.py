import random
import numpy as np

adj_dict_3 = [[1,3,4],[0, 2, 4],[1, 4, 5],[0, 4, 6],[0, 1, 2, 3, 5, 6, 7, 8],[2, 4, 8],[7, 3, 4],[6, 8, 4],[7, 4, 5]]


adj_dict_6 = [[1, 6],[0, 2, 4],[1, 9],[4, 7],[1, 3, 5],[4, 8],[7, 0, 13],[10, 3, 6],[12, 5, 9],[2, 8, 15],[7, 11],[12, 14, 10],[8, 11],
	     [14, 6],[13, 15, 11],[9, 14]]


adj_dict_9 = [[1, 9],[0, 2, 4],[1, 14],[4, 10],[1, 3, 5, 7],[4, 13],[7, 11],[4, 6, 8],[12, 7],[0, 21, 10],[11, 18, 3, 9],[6, 15, 10],
	      [8, 17, 13, 14],[14, 20, 5],[2, 23, 13],[11, 15],[15, 17, 19],[12, 16],[10, 19],[18, 20, 10, 22],[19, 13],[9, 22],
	      [21, 23, 19],[22, 14]]

adj_dict_12 = [[1, 9, 3],[0, 2, 4],[1, 14, 5],[4, 10, 0, 6],[1, 3, 5, 7],[4, 13, 2, 8],[7, 11, 3],[4, 6, 8],[12, 7, 5],[0, 21, 10],
	       [11, 18, 3],[6, 15],[8, 17],[14, 20, 5],[2, 23, 13],[11, 15, 18],[15, 17, 19],[12, 16, 20],[10, 18, 15, 21],
	       [18, 20, 10, 22],[19, 13, 17, 20],[9, 22, 18],[21, 23, 19],[22, 14, 20]]

class Random_Player(object):
	def __init__(self):
		
		self.state_index = []

	def place(self, state, game_type, player, move_no):
		space_val = 1
		while space_val != 0:
			space = random.randint(0, len(state) - 1)
			space_val = state[space]
		return space
	
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

	
	def remove_piece(self, state, piece_list, game_type, player, removed_pieces):
		piece_to_remove = None
		while piece_to_remove is None:
			temp = random.randint(0, len(piece_list) - 1)
			piece_to_remove = piece_list[temp]
		return piece_to_remove

	
	def free_space_finder(self, state):
		free_space = []
		for item in state:
			if item == 0:
				free_space.append(item)

		return free_space
	
	def move(self, state, game_type, pieces, player, enable_flying, move_no):
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
		
	def edit_to_index(self,state,move_no):
		return 0
		
	def edit_from_index(self,state,move_no,game_type):
		return 0
		
	def edit_remove_index(self,state,pieces_removed):
		return 0
