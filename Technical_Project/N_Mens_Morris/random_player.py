import random
import numpy as np

adj_dict_3 = [[1,3,4],[0, 2, 4],[1, 4, 5],[0, 4, 6],[0, 1, 2, 3, 5, 6, 7, 8],[2, 4, 8],[7, 3, 4],[6, 8, 4],[7, 4, 5]]

badj_dict_3 = {
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

adj_dict_6 = [[1, 6],[0, 2, 4],[1, 9],[4, 7],[1, 3, 5],[4, 8],[7, 0, 13],[10, 3, 6],[12, 5, 9],[2, 8, 15],[7, 11],[12, 14, 10],[8, 11],
	     [14, 6],[13, 15, 11],[9, 14]]

badj_dict_6 = {
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

adj_dict_9 = [[1, 9],[0, 2, 4],[1, 14],[4, 10],[1, 3, 5, 7],[4, 13],[7, 11],[4, 6, 8],[12, 7],[0, 21, 10],[11, 18, 3, 9],[6, 15, 10],
	      [8, 17, 13, 14],[14, 20, 5],[2, 23, 13],[11, 15],[15, 17, 19],[12, 16],[10, 19],[18, 20, 10, 22],[19, 13],[9, 22],
	      [21, 23, 19],[22, 14]]

badj_dict_9 = {
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

adj_dict_12 = [[1, 9, 3],[0, 2, 4],[1, 14, 5],[4, 10, 0, 6],[1, 3, 5, 7],[4, 13, 2, 8],[7, 11, 3],[4, 6, 8],[12, 7, 5],[0, 21, 10],
	       [11, 18, 3],[6, 15],[8, 17],[14, 20, 5],[2, 23, 13],[11, 15, 18],[15, 17, 19],[12, 16, 20],[10, 18, 15, 21],
	       [18, 20, 10, 22],[19, 13, 17, 20],[9, 22, 18],[21, 23, 19],[22, 14, 20]]

badj_dict_12 = {
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

class Random_Player(object):
	def __init__(self):
		
		self.state_index = []

	def place(self,state, free_space, game_type, player):
        	temp = random.randint(0, len(free_space) - 1)
        	return free_space[temp]

	def valid_move(self, state, game_type, free_space, pieces):
		valid_moves = []
		if game_type == 3:
			for piece in pieces:
				for space in adj_dict_3[piece]:
					if space in free_space:
						valid_moves.append((piece,space))

		if game_type == 6:
			for piece in pieces:
				for space in adj_dict_6[piece]:
					if space in free_space:
						valid_moves.append((piece,space))

		if game_type == 9:
			for piece in pieces:
				for space in adj_dict_9[piece]:
					if space in free_space:
						valid_moves.append((piece,space))

		if game_type == 12:
			for piece in pieces:
				for space in adj_dict_12[piece]:
					if space in free_space:
						valid_moves.append((piece,space))

		return valid_moves

	def remove_piece(self, state, piece_list, game_type, player):
		temp = random.randint(0, len(piece_list) - 1)
		return piece_list[temp]

	
	def move(self, state, game_type, free_space, pieces, player, enable_flying):
		valid_moves = self.valid_move(state, game_type, free_space,pieces)
		if len(valid_moves) == 0:
			return (25, 25)
		temp = random.randint(0, len(valid_moves) - 1)
		if enable_flying:
			temp2 = random.randint(0, len(free_space) - 1)
			if free_space == valid_moves[0]:
				temp2 -= 1
			return (valid_moves[temp][0],free_space[temp2])
		else:
			return valid_moves[temp]
