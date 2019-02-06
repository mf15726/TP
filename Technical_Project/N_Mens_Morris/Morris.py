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
#Classes
from learned_player import Learned_Player
from random_player import Random_Player
from human_player import Human_Player

#game_type = 3, 6, 9, 12 (Men's Morris)

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
"13": [[0, 6]  [14, 15]],
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

    count1 = 0
    count2 = 0
    for row in state:
        temp1 = row.count(1)
        temp2 = row.count(2)
        count1 += temp1
        count2 += temp2
    print(('Count1 = ') + str(count1))
    print(('Count2 = ') + str(count2))
    if count1 <= 2:
        return 2
    if count2 <= 2:
        return 1
    else:
        return 0

def det_mill(state, move, game_type):
    if game_type == 3:
        for item in mill_dict_3[str(move)]:
            if state[move] == state[item[0]] == state[item[1]]:
                return True
        else:
            return False

    if game_type == 6:
        for item in mill_dict_6[str(move)]:
             if state[move] == state[item[0]] == state[item[1]]:
                return True
        else:
            return False


    if game_type == 9:
        for item in mill_dict_9[str(move)]:
        	if state[move] == state[item[0]] == state[item[1]]:
                return True
        else:
            return False


    if game_type == 12:
        for item in mill_dict_12[str(move)]:
             if state[move] == state[item[0]] == state[item[1]]:
                return True
        else:
            return False



def free_space_finder(state):
    free_space = []
    for i in range(len(state)):
        if state[i] == 0:
            free_space.append(i)

    return free_space


def game_play(player1,player2,game_type):#
    winner = 0
    move_no = 0
    game_states = []
    player1_piece_list = []
    player2_piece_list = []
    if game_type == 3:
        state = [0,0,0,0,0,0,0,0,0]
    elif game_type == 6:
        state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    else:
        state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    printboard(game_type,state)
    free_space = free_space_finder(state)
    while winner == 0:
        if move_no < game_type * 2:
            if move_no % 2 == 0:
                move = player1.place(state,free_space,game_type,nodes)
                player1_piece_list.append(move)
            else:
                move = player2.place(state,free_space,game_type,nodes)
                player2_piece_list.append(move)
            state[move] = (move_no % 2) + 1
            free_space.remove(move)
            printboard(game_type,state)
            if det_mill(state, move, game_type):
                print('Mill Created')
                if move_no % 2 == 0:
                    removed_piece = player1.remove_piece(player2_piece_list,game_type,nodes)
                    print('P2 Plist = ' + str(player2_piece_list))
                    print('Removed piece = ' + str(removed_piece))
                    state[removed_piece] = 0
                    player2_piece_list.remove(removed_piece)
                else:
                    removed_piece = player2.remove_piece(player1_piece_list,game_type,nodes)
                    print('P1 Plist = ' + str(player1_piece_list))
                    print('Removed piece = ' + str(removed_piece))
                    state[removed_piece] = 0
                    player1_piece_list.remove(removed_piece)
                free_space.append(removed_piece)
                printboard(game_type,state)
#            winner = end_game(state)
        else:
            if move_no == game_type * 2:
                winner = end_game(state)
                if winner != 0:
                    return winner, game_states

            print('Free Space = ' + str(free_space))
            print('P1 PList = ' + str(player1_piece_list))
            print('P2 PList = ' + str(player2_piece_list))
            if move_no % 2 == 0:
                prev_pos, move = player1.move(state,game_type,free_space,player1_piece_list,nodes)
                if move == [9,9]:
                    return winner ,game_states
                player1_piece_list.append(move)
                player1_piece_list.remove(prev_pos)
                print('Player1 moves' + str(move))
                print('From ' + str(prev_pos))
            else:
                prev_pos, move = player2.move(state,game_type,free_space,player2_piece_list,nodes)
                if move == [9,9]:
                    return winner ,game_states
                player2_piece_list.append(move)
                player2_piece_list.remove(prev_pos)
                print('Player2 moves' + str(move))
                print('From ' + str(prev_pos))
            state[move] = (move_no % 2) + 1
            state[prev_pos] = 0
            free_space.remove(move)
            free_space.append(prev_pos)
            printboard(game_type,state)
            print(len(free_space))
            if det_mill(state, move, game_type):
                print('Mill Created')
                if move_no % 2 == 0:
                    removed_piece = player1.remove_piece(player2_piece_list,nodes)
                    print('Removed piece = ' + str(removed_piece))
                    state[removed_piece] = 0
                    player2_piece_list.remove(removed_piece)
                else:
                    removed_piece = player2.remove_piece(player1_piece_list,nodes)
                    print('Removed piece = ' + str(removed_piece))
                    state[removed_piece] = 0
                    player1_piece_list.remove(removed_piece)
                free_space.append(removed_piece)
                printboard(game_type,state)
                winner = end_game(state)
        game_states.append(state)
        move_no += 1





    return winner, game_states

human_player = Human_Player()
random_player = Random_Player()
learned_player = Learned_Player()
#gameboard = define_board(6)
#nx.draw(gameboard)
#plt.show()

for i in range(1):
    winner, game_states = game_play(human_player1,random_player2, 3)
print(winner)
