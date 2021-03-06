#Packages
import numpy as np
import random
from copy import deepcopy
import csv
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from math import log
import cProfile, pstats
#Classes
from learned_player import Learned_Player
from random_player import Random_Player
from human_player import Human_Player
from multi_task_player import Multi_Task_Player

#game_type = 3, 6, 9, 12 (Men's Morris)

mill_dict_3 = [[[1, 2], [3, 6], [4, 8]],
	       [[0, 2], [4, 7]],
	       [[5, 8], [4, 6], [0,1]],
	       [[0, 6], [4, 5]],
	       [[0, 8], [1, 7], [2, 6], [3, 5]],
	       [[2, 8], [4, 3]],
	       [[7, 8], [2, 4], [0, 3]],
	       [[6, 8], [1, 4]],
	       [[7, 6], [2, 5], [0, 4]]]

mill_dict_6 = [[[1, 2], [13, 6]],
	       [[0, 2]],
	       [[0, 1], [9, 15]],
	       [[10, 7], [4, 5]],
	       [[3, 5]],
	       [[12, 8], [4, 3]],
	       [[0, 13]],
	       [[10, 3]],
	       [[5, 12]],
	       [[2, 15]],
	       [[3, 7], [11, 12]],
	       [[10, 12]],
	       [[10, 11], [5, 8]],
	       [[0, 6], [14, 15]],
	       [[13, 15]],
	       [[2, 9], [13, 14]]]

mill_dict_9 = [[[1, 2], [9, 21]],
	       [[0, 2], [7, 4]],
	       [[0, 1], [14, 23]],
	       [[18, 10], [4, 5]],
	       [[3, 5], [1, 7]],
	       [[13, 20], [4, 3]],
	       [[11, 15], [7, 8]],
	       [[6, 8], [1, 4]],
	       [[12, 17], [6, 7]],
	       [[10, 11], [0, 21]],
	       [[9, 11], [3, 18]],
	       [[6, 15], [10, 9]],
	       [[8, 17], [13, 14]],
	       [[12, 14], [5, 20]],
	       [[2, 23], [12, 13]],
	       [[6, 11], [16, 17]],
	       [[15, 17], [19, 22]],
	       [[12, 8], [15, 16]],
	       [[10, 3], [19, 20]],
	       [[18, 20], [16, 22]],
	       [[19, 18], [5, 13]],
	       [[9, 0], [22, 23]],
	       [[21, 23], [16, 19]],
	       [[21, 22], [2, 14]]]

mill_dict_12 = [[[1, 2], [9, 21], [3, 6]],
		[[0, 2], [7, 4]],
		[[0, 1], [14, 23], [5, 8]],
		[[18, 10], [4, 5], [0, 6]],
		[[3, 5], [1, 7]],
		[[13, 20], [4, 3], [2, 8]],
		[[11, 15], [7, 8], [3 ,0]],
		[[6, 8], [1, 4]],
		[[12, 17], [6, 7], [2, 5]],
		[[10, 11], [0, 21]],
		[[9, 11], [3, 18]],
		[[6, 15], [9, 10]],
		[[8, 17], [13, 14]],
		[[12, 14], [5, 20]],
		[[2, 23], [12, 13]],
		[[6, 11], [16, 17], [18, 21]],
		[[15, 17], [19, 22]],
		[[12, 8], [15, 16], [20, 23]],
		[[10, 3], [19, 20], [15, 21]],
		[[18, 20], [16, 22]],
		[[19, 18], [5, 13], [17, 23]],
		[[9, 0], [22, 23],[15, 18]],
		[[21, 23], [16, 19]],
		[[21, 22], [2, 14], [17, 20]]]


def printboard(game_type,state):
#    gameboard = nx.graph()
	if game_type == 3:
		print(str(state[0])+'-'+str(state[1])+'-'+str(state[2]))
		print('|\|/|')
		print(str(state[3])+'-'+str(state[4])+'-'+str(state[5]))
		print('|/|\|')
		print(str(state[6])+'-'+str(state[7])+'-'+str(state[8]))

	if game_type == 6:
		print(str(state[0])+'---'+str(state[1])+'---'+str(state[2]))
		print('|   |   |')
		print('| '+str(state[3])+'-'+str(state[4])+'-'+str(state[5])+' |')
		print('| |   | |')
		print(str(state[6])+'-'+str(state[7])+'   '+str(state[8])+'-'+str(state[9]))
		print('| |   | |')
		print('| '+str(state[10])+'-'+str(state[11])+'-'+str(state[12])+' |')
		print('|   |   |')
		print(str(state[13])+'---'+str(state[14])+'---'+str(state[15]))

	if game_type == 9:
		print(str(state[0])+'-----'+str(state[1])+'-----'+str(state[2]))
		print('|     |     |')
		print('| '+str(state[3])+'---'+str(state[4])+'---'+str(state[5])+' |')
		print('| |   |   | |')
		print('| | '+str(state[6])+'-'+str(state[7])+'-'+str(state[8])+' | |')
		print('| | |   | | |')
		print(str(state[9])+'-'+str(state[10])+'-'+str(state[11])+'   '+str(state[12])+'-'+str(state[13])+'-'+str(state[14]))
		print('| | |   | | |')
		print('| | '+str(state[15])+'-'+str(state[16])+'-'+str(state[17])+' | |')
		print('| |   |   | |')
		print('| '+str(state[18])+'---'+str(state[19])+'---'+str(state[20])+' |')
		print('|     |     |')
		print(str(state[21])+'-----'+str(state[22])+'-----'+str(state[23]))

	if game_type == 12:
		print(str(state[0])+'-----'+str(state[1])+'-----'+str(state[2]))
		print('|\    |    /|')
		print('| '+str(state[3])+'---'+str(state[4])+'---'+str(state[5])+' |')
		print('| |\  |  /| |')
		print('| | '+str(state[6])+'-'+str(state[7])+'-'+str(state[8])+' | |')
		print('| | |   | | |')
		print(str(state[9])+'-'+str(state[10])+'-'+str(state[11])+'   '+str(state[12])+'-'+str(state[13])+'-'+str(state[14]))
		print('| | |   | | |')
		print('| | '+str(state[15])+'-'+str(state[16])+'-'+str(state[17])+' | |')
		print('| |/  |  \| |')
		print('| '+str(state[18])+'---'+str(state[19])+'---'+str(state[20])+' |')
		print('|/    |    \|')
		print(str(state[21])+'-----'+str(state[22])+'-----'+str(state[23]))

def end_game(state):
	count1 = state.count(1)
	count2 = state.count(2)
#	print(('Count1 = ') + str(count1))
#	print(('Count2 = ') + str(count2))
	if count1 <= 2:
		return 2
	if count2 <= 2:
		return 1
	else:
		return 0

def det_mill(state, move, game_type):
	if game_type == 3:
		for item in mill_dict_3[move]:
			if state[move] == state[item[0]] == state[item[1]]:
				return True

	if game_type == 6:
		if isinstance(mill_dict_6[move][0],int):
			if state[move] == state[mill_dict_6[0]] == state[mill_dict_6[1]]:
				return True
			else:
				return False
		for item in mill_dict_6[move]:
			if state[move] == state[item[0]] == state[item[1]]:
				return True

	if game_type == 9:
		for item in mill_dict_9[move]:	
			if state[move] == state[item[0]] == state[item[1]]:
				return True

	if game_type == 12:
		for item in mill_dict_12[move]:	
			if state[move] == state[item[0]] == state[item[1]]:
				return True
	return False

def free_space_finder(state):
	free_space = []
	for i in range(len(state)):
		if state[i] == 0:
			free_space.append(i)

	return free_space

#def free_space_finder(state):
#	free_space = [0] * 9
#	for i in range(len(state)):
		

def flying_check(state, player, game_type):
	if game_type == 3:
		return False
	count = state.count(player)
	if count == 3:
		return True
	else:
		return False

def repeated_board(state,game_states):
	if game_states.count(state) > 5:
		return True
	return False
	
def game_play(player1,player2,game_type,print_board,flying,limit):
	winner = 0
	move_no = 0
	player1_piece_list = [None] * game_type
	player2_piece_list = [None] * game_type
	game_states = [None] * total_move_no
	pieces_removed = 0
	if game_type == 3:
		state = [0,0,0,0,0,0,0,0,0]
	elif game_type == 6:
		state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	else:
		state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	if print_board:
		printboard(game_type,state)
	free_space = free_space_finder(state)
	while winner == 0:
		player = (move_no % 2) + 1
		if move_no < game_type * 2:
			if player == 1:
				move = player1.place(state,game_type,player,move_no)
				player1_piece_list[int(move_no/2)] = move
			else:
				move = player2.place(state,game_type,player,move_no)
				player2_piece_list[int((move_no - 1)/2)] = move
#			print('Placed by Player ' + str(player) + ' ' +  str(move))
#			print('Free Space = ' +str(free_space))
			state[move] = player
			if print_board:
				printboard(game_type,state)
			if det_mill(state, move, game_type):
				if print_board:
					print('Mill Created by Player ' + str(player))
				if game_type == 3:
					return player
				if player == 1:
					removed_piece = player1.remove_piece(state,player2_piece_list,game_type,player,pieces_removed)
#					print('P2 Plist = ' + str(player2_piece_list))
#					print('Removed piece = ' + str(removed_piece))
					state[removed_piece] = 0
					_ = player2_piece_list.index(removed_piece)
					player2_piece_list[_] = None
					player1.edit_to_index(state,game_type,move_no,player)
					player1.edit_remove_index(state,game_type,pieces_removed,player)
					pieces_removed += 1
				else:
					removed_piece = player2.remove_piece(state,player1_piece_list,game_type,player,pieces_removed)
#					print('P1 Plist = ' + str(player1_piece_list))
#					print('Removed piece = ' + str(removed_piece))
					state[removed_piece] = 0
					_ = player1_piece_list.index(removed_piece)
					player1_piece_list[_] = None
					player2.edit_to_index(state,game_type,move_no,player)
					player2.edit_remove_index(state,game_type,pieces_removed,player)
					pieces_removed += 1
			else:
				if player == 1:
					player1.edit_to_index(state,game_type,move_no,player)
				else:
					player2.edit_to_index(state,game_type,move_no,player)
		else:
			if move_no == game_type * 2:
				winner = end_game(state)
				if winner != 0:
					return winner
				if flying:
					p1_fly = flying_check(state,1,game_type)
					p2_fly = flying_check(state,2,game_type)
			if player == 1:
				prev_pos, move = player1.move(state,game_type,player1_piece_list,player,p1_fly,move_no)
				if move == 25:
					if game_type == 12 and move_no == game_type * 2:
						return 0
					else:
						return 2
#				print('P1 move to = ' + str(move) + ' from = ' + str(prev_pos))
				if prev_pos not in player1_piece_list:
					print('Prev_Pos ' +str(prev_pos))
					print('Piece_List ' +str(player1_piece_list))
				ind = player1_piece_list.index(prev_pos)
				player1_piece_list[ind] = move
#				print('P1PList = ' + str(player1_piece_list))
			else:
				prev_pos, move = player2.move(state,game_type,player2_piece_list,player,p2_fly,move_no)
				if move == 25:
					return 1
#				print('P2 move to = ' + str(move) + ' from = ' + str(prev_pos))
				if prev_pos not in player2_piece_list:
					print('Prev_Pos ' +str(prev_pos))
					print('Piece_List ' +str(player2_piece_list))
				ind = player2_piece_list.index(prev_pos)
				player2_piece_list[ind] = move
#				print('P2PList = ' + str(player2_piece_list))
			state[move] = player
			state[prev_pos] = 0
			if print_board:
				printboard(game_type,state)
			if det_mill(state, move, game_type):
				if print_board:
					print('Mill Created by Player ' + str(player))
				if player == 1:
					removed_piece = player1.remove_piece(state,player2_piece_list,game_type,player,pieces_removed)
#					print('P2 Plist = ' + str(player2_piece_list))
#					print('Removed piece = ' + str(removed_piece))
					state[removed_piece] = 0
					_ = player2_piece_list.index(removed_piece)
					player2_piece_list[_] = None
					player1.edit_to_index(state,game_type,move_no,player)
					player1.edit_from_index(state,move_no,game_type,player)
					player1.edit_remove_index(state,game_type,pieces_removed,player)
					pieces_removed += 1
					if flying:
						p2_fly = flying_check(state,2,game_type)
				else:
					removed_piece = player2.remove_piece(state,player1_piece_list,game_type,player,pieces_removed)
#					print('P1 Plist = ' + str(player1_piece_list))
#					print('Removed piece = ' + str(removed_piece))
					state[removed_piece] = 0
					_ = player1_piece_list.index(removed_piece)
					player1_piece_list[_] = None
					player2.edit_to_index(state,game_type,move_no,player)
					player2.edit_from_index(state,move_no,game_type,player)
					player2.edit_remove_index(state,game_type,pieces_removed,player)
					pieces_removed += 1
					if flying:
						p1_fly = flying_check(state,1,game_type)
				if print_board:
					printboard(game_type,state)
				winner = end_game(state)
			else:
				if player == 1:
					player1.edit_to_index(state,game_type,move_no,player)
					player1.edit_from_index(state,move_no,game_type,player)
				else:
					player2.edit_to_index(state,game_type,move_no,player)
					player2.edit_from_index(state,move_no,game_type,player)
		game_states[move_no] = deepcopy(state)
		move_no += 1
		if repeated_board(state,game_states):
			return 0
		if move_no == limit:
			return 0




	return winner

winner_list = []
enable_flying = True
game_type = 3
see_board = False
total_move_no = 1000
game_states = [None] * (total_move_no + 1)

human_player = Human_Player()
random_player = Random_Player()
learned_player = Learned_Player(epsilon=1, alpha=0.00035, gamma=0.000000001, limit=total_move_no)
multi_task_player = Multi_Task_Player(epsilon=1, alpha=0.00035, gamma=0.00000001, limit=total_move_no)
learned_player.sess.run(tf.global_variables_initializer())
multi_task_player.sess.run(tf.global_variables_initializer())
#pr = cProfile.Profile()
#pr.enable()
def play_and_learn(total_game_no,player1,player2):
	winner_list = [None] * total_game_no
	t1_win_list = []
	t2_win_list = []
	t1_loss_list = []
	t2_loss_list = []
	graph_name = 'Loss'
	line_name = 'cost'
	for i in range(total_game_no):
		player1.epsilon = 1-((i+1)/(total_game_no+1))
		player2.epsilon = 1-((i+1)/(total_game_no+1))
#		print(player1.epsilon)
		if i % 250 == 0:
			print('At epoch ' + str(i))
			player1.epsilon = 0
			player2.epsilon = 0
			test_winner_list1 = play_dont_learn(100,player1,random_player)
			test_winner_list2 = play_dont_learn(100,random_player,player2)
			t1_wins = test_winner_list1.count(1)
			t2_wins = test_winner_list2.count(2)
			t1_loss = test_winner_list1.count(2)
			t2_loss = test_winner_list2.count(1)
			print('Agent wins as player 1 ' + str(test_winner_list1.count(1)))
			print('Agent loses as player 1 ' + str(test_winner_list1.count(2)))
			print('Agent wins as player 2 ' + str(test_winner_list2.count(2)))
			print('Agent loses as player 2 ' + str(test_winner_list2.count(1)))
			t1_win_list.append(t1_wins)
			t2_win_list.append(t2_wins)
			t1_loss_list.append(t1_loss)
			t2_loss_list.append(t2_loss)
		winner = game_play(player1, player2, game_type, see_board, enable_flying, total_move_no)
		print('Winner of game ' + str(i+1) + ' is Player ' + str(winner))
		winner_list[i] = winner
		if winner != 0:
			if isinstance(player1, Multi_Task_Player) or isinstance(player2, Multi_Task_Player):
				if game_type == 3:
					multi_task_player.learn3(game_type, winner)
				elif game_type == 6:
					multi_task_player.learn6(game_type, winner)
				elif game_type == 9:
					multi_task_player.learn9(game_type, winner)
				else:
					multi_task_player.learn12(game_type, winner)
		if isinstance(player1, Learned_Player) or isinstance(player2, Learned_Player):
			loss = learned_player.learn(game_type, winner)
			print('Loss of game ' +str(i+1) + ' is ' +str(loss))
			tbc.save_value(graph_name, line_name, i, loss)
			tbc.flush_line(line_name)
		tbc.close()
	return winner_list, t1_win_list, t2_win_list, t1_loss_list, t2_loss_list

def play_dont_learn(total_game_no,player1,player2):
	winner_list = [None] * total_game_no
	for i in range(total_game_no):
		winner = game_play(player1, player2, game_type, False, enable_flying, total_move_no)
#		print('Winner of game ' + str(i+1) + ' is Player ' + str(winner))
		winner_list[i] = winner
	return winner_list

#winner_list = play_and_learn(1000,multi_task_player,multi_task_player)
#winner_list = play_and_learn(1000000,random_player,random_player)
winner_list, test1_win, test2_win, test1_loss, test2_loss = play_and_learn(50000,learned_player,learned_player)
print('P1 wins = ' + str(winner_list.count(1)))
print('P2 wins = ' + str(winner_list.count(2)))

print('P1 test wins ' + str(test1_win))
print('P2 test wins ' + str(test2_win))
print('P1 test loss ' + str(test1_loss))
print('P2 test loss ' + str(test2_loss))
#cProfile.run('play_and_learn(1000,learned_player,learned_player)')
#pr.disable()
