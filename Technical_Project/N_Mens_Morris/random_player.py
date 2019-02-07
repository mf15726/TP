import random
import numpy as np
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

class Random_Player(object):
	def __init__(self):
		
		self.state_index = []

	def place(self,state, free_space, game_type):
        	temp = random.randint(0, len(free_space) - 1)
        	return free_space[temp]

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

	def remove_piece(self, piece_list, game_type, nodes):
		temp = random.randint(0, len(piece_list) - 1)
		return piece_list[temp]

	def move(self, state, game_type, free_space, pieces, nodes):
		valid_moves = self.valid_move(state, game_type, free_space,pieces)
		if len(valid_moves) == 0:
			return (25, 25)
        temp = random.randint(0, len(valid_moves) - 1)
#        prev_pos, move = valid_moves[temp]
        return valid_moves[temp]
