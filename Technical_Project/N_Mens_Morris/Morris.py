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
"[0, 0]": [[0,1], [1,0], [1,1]],
"[0, 1]": [[0,0], [0,2], [1,1]],
"[0, 2]": [[0,1], [1,1], [1,2]],
"[1, 0]": [[0,0], [1,1], [2,0]],
"[1, 1]": [[0,0], [0,1], [0,2], [1,0], [1,2], [2,0], [2,1], [2,2]],
"[1, 2]": [[0,2], [1,1], [2,2]],
"[2, 0]": [[2,1], [1,0], [1,1]],
"[2, 1]": [[2,0], [2,2], [1,1]],
"[2, 2]": [[2,1], [1,1], [1,2]],
}

mill_dict_3 = {
"[0, 0]": [[[0,1], [0,2]], [[1,0], [2,0]], [[1,1], [2,2]]],
"[0, 1]": [[[0,0], [0,2]], [[1,1], [2,1]]],
"[0, 2]": [[[1,2], [2,2]], [[1,1], [2,0]], [[0,0], [0,1]]],
"[1, 0]": [[[0,0], [2,0]], [[1,1], [1,2]]],
"[1, 1]": [[[0,0], [2,2]], [[0,1], [2,1]], [[0,2], [2,0]], [[1,0], [1,2]]],
"[1, 2]": [[[0,2], [2,2]], [[1,1], [1,0]]],
"[2, 0]": [[[2,1], [2,2]], [[1,1], [0,2]], [[0,0], [1,0]]],
"[2, 1]": [[[2,0], [2,2]], [[1,1], [0,1]]],
"[2, 2]": [[[2,1], [2,0]], [[1,2], [0,2]], [[0,0], [1,1]]]
}

adj_dict_6 = {
"[0, 0]": [[0,1], [2,0]],
"[0, 1]": [[0,0], [0,2], [1,1]],
"[0, 2]": [[0,1], [2,3]],
"[1, 0]": [[1,1], [2,1]],
"[1, 1]": [[0,1], [1,0], [1,2]],
"[1, 2]": [[1,1], [2,2]],
"[2, 0]": [[2,1], [0,0], [4,0]],
"[2, 1]": [[3,0], [1,0], [2,0]],
"[2, 2]": [[3,2], [1,2], [2,3]],
"[2, 3]": [[0,2], [2,2], [4,2]],
"[3, 0]": [[2,1], [3,1]],
"[3, 1]": [[3,2], [4,1], [3,0]],
"[3, 2]": [[2,2], [3,1]],
"[4, 0]": [[4,1], [2,0]],
"[4, 1]": [[4,0], [4,2], [3,1]],
"[4, 2]": [[2,3], [4,1]]
}

mill_dict_6 = {
"[0, 0]": [[[0,1], [0,2]], [[4,0], [2,0]]],
"[0, 1]": [[[0,0], [0,2]]],
"[0, 2]": [[[2,3], [4,2]], [[0,0], [0,1]]],
"[1, 0]": [[[3,0], [2,1]], [[1,1], [1,2]]],
"[1, 1]": [[[1,0], [1,2]]],
"[1, 2]": [[[3,2], [2,2]], [[1,1], [1,0]]],
"[2, 0]": [[[0,0], [4,0]]],
"[2, 1]": [[[3,0], [1,0]]],
"[2, 2]": [[[1,2], [3,2]]],
"[2, 3]": [[[0,2], [4,2]]],
"[3, 0]": [[[1,0], [2,1]], [[3,1], [3,2]]],
"[3, 1]": [[[3,0], [3,2]]],
"[3, 2]": [[[3,1], [3,0]], [[1,2], [2,2]]],
"[4, 0]": [[[2,0], [0,0]], [[4,1], [4,2]]],
"[4, 1]": [[[4,0], [4,2]]],
"[4, 2]": [[[2,3], [0,2]], [[4,0], [4,1]]]
}

adj_dict_9 = {
"[0, 0]": [[0,1], [3,0]],
"[0, 1]": [[0,0], [0,2], [1,1]],
"[0, 2]": [[0,1], [3,5]],
"[1, 0]": [[1,1], [3,1]],
"[1, 1]": [[0,1], [1,0], [1,2], [2,1]],
"[1, 2]": [[1,1], [3,4]],
"[2, 0]": [[2,1], [3,2]],
"[2, 1]": [[1,1], [2,0], [2,2]],
"[2, 2]": [[3,3], [2,1]],
"[3, 0]": [[0,0], [6,0], [3,1]],
"[3, 1]": [[3,2], [5,0], [1,0], [3,0]],
"[3, 2]": [[2,0], [4,0], [3,1]],
"[3, 3]": [[2,2], [4,2], [3,4], [3,5]],
"[3, 4]": [[3,5], [5,2], [1,2]],
"[3, 5]": [[0,2], [6,2], [3,4]],
"[4, 0]": [[3,2], [4,0]],
"[4, 1]": [[4,0], [4,2], [5,1]],
"[4, 2]": [[3,3], [4,1]],
"[5, 0]": [[3,1], [5,1]],
"[5, 1]": [[5,0], [5,2], [3,1], [6,1]],
"[5, 2]": [[5,1], [3,4]],
"[6, 0]": [[3,0], [6,1]],
"[6, 1]": [[6,0], [6,2], [5,1]],
"[6, 2]": [[6,1], [3,5]]
}

mill_dict_9 = {
"[0, 0]": [[[0,1], [0,2]], [[3,0], [6,0]]],
"[0, 1]": [[[0,0], [0,2]], [[2,1], [1,1]]],
"[0, 2]": [[[0,0], [0,1]], [[3,5], [6,2]]],
"[1, 0]": [[[5,0], [3,1]], [[1,1], [1,2]]],
"[1, 1]": [[[1,0], [1,2]], [[0,1], [2,1]]],
"[1, 2]": [[[3,4], [5,2]], [[1,1], [1,0]]],
"[2, 0]": [[[3,2], [4,0]], [[2,1], [2,2]]],
"[2, 1]": [[[2,0], [2,2]], [[0,1], [1,1]]],
"[2, 2]": [[[3,3], [4,2]], [[2,0], [2,1]]],
"[3, 0]": [[[3,1], [3,2]], [[0,0], [6,0]]],
"[3, 1]": [[[3,0], [3,2]], [[1,0], [5,0]]],
"[3, 2]": [[[2,0], [4,0]], [[3,0], [3,2]]],
"[3, 3]": [[[2,2], [4,2]], [[3,4], [3,5]]],
"[3, 4]": [[[3,3], [3,5]], [[1,2], [5,2]]],
"[3, 5]": [[[0,2], [6,2]], [[3,3], [3,4]]],
"[4, 0]": [[[2,0], [3,2]], [[4,1], [4,2]]],
"[4, 1]": [[[4,0], [4,2]], [[5,1], [6,1]]],
"[4, 2]": [[[3,3], [2,2]], [[4,0], [4,1]]],
"[5, 0]": [[[3,1], [1,0]], [[5,1], [5,2]]],
"[5, 1]": [[[5,0], [5,2]], [[3,1], [1,0]]],
"[5, 2]": [[[5,1], [5,0]], [[1,2], [3,4]]],
"[6, 0]": [[[3,0], [0,0]], [[6,1], [6,2]]],
"[6, 1]": [[[6,0], [6,2]], [[4,1], [5,1]]],
"[6, 2]": [[[6,0], [6,1]], [[0,2], [3,5]]]
}

adj_dict_12 = {
"[0, 0]": [[0,1], [3,0], [1,0]],
"[0, 1]": [[0,0], [0,2], [1,1]],
"[0, 2]": [[0,1], [3,5], [1,2]],
"[1, 0]": [[1,1], [3,1], [0,0], [2,0]],
"[1, 1]": [[0,1], [1,0], [1,2], [2,1]],
"[1, 2]": [[1,1], [3,4], [0,2], [2,2]],
"[2, 0]": [[2,1], [3,2], [1,0]],
"[2, 1]": [[1,1], [2,0], [2,2]],
"[2, 2]": [[3,3], [2,1], [1,2]],
"[3, 0]": [[0,0], [6,0], [3,1]],
"[3, 1]": [[3,2], [5,0], [1,0]],
"[3, 2]": [[2,0], [4,0]],
"[3, 3]": [[2,2], [4,2]],
"[3, 4]": [[3,5], [5,2], [1,2]],
"[3, 5]": [[0,2], [6,2], [3,4]],
"[4, 0]": [[3,2], [4,0], [5,0]],
"[4, 1]": [[4,0], [4,2], [5,1]],
"[4, 2]": [[3,3], [4,1], [5,2]],
"[5, 0]": [[3,1], [5,0], [4,0], [6,0]],
"[5, 1]": [[5,0], [5,2], [3,1], [6,1]],
"[5, 2]": [[5,1], [3,4], [4,2], [5,2]],
"[6, 0]": [[3,0], [6,1], [5,0]],
"[6, 1]": [[6,0], [6,2], [5,1]],
"[6, 2]": [[6,1], [3,5], [5,2]]
}

mill_dict_12 = {
"[0, 0]": [[[0,1], [0,2]], [[3,0], [6,0]], [[1,0], [2,0]]],
"[0, 1]": [[[0,0], [0,2]], [[2,1], [1,1]]],
"[0, 2]": [[[0,0], [0,1]], [[3,5], [6,2]], [[1,2], [2,2]]],
"[1, 0]": [[[5,0], [3,1]], [[1,1], [1,2]], [[0,0], [2,0]]],
"[1, 1]": [[[1,0], [1,2]], [[0,1], [2,1]]],
"[1, 2]": [[[3,4], [5,2]], [[1,1], [1,0]], [[0,2], [2,2]]],
"[2, 0]": [[[3,2], [4,0]], [[2,1], [2,2]], [[1,0], [0,0]]],
"[2, 1]": [[[2,0], [2,2]], [[0,1], [1,1]]],
"[2, 2]": [[[3,3], [4,2]], [[2,0], [2,1]], [[1,2], [0,2]]],
"[3, 0]": [[[3,1], [3,2]], [[0,0], [6,0]]],
"[3, 1]": [[[3,0], [3,2]], [[1,0], [5,0]]],
"[3, 2]": [[[2,0], [4,0]], [[3,0], [3,2]]],
"[3, 3]": [[[2,2], [4,2]], [[3,4], [3,5]]],
"[3, 4]": [[[3,3], [3,5]], [[1,2], [5,2]]],
"[3, 5]": [[[0,2], [6,2]], [[3,3], [3,4]]],
"[4, 0]": [[[2,0], [3,2]], [[4,1], [4,2]], [[5,0], [6,0]]],
"[4, 1]": [[[4,0], [4,2]], [[5,1], [6,1]]],
"[4, 2]": [[[3,3], [2,2]], [[4,0], [4,1]], [[5,2], [6,2]]],
"[5, 0]": [[[3,1], [1,0]], [[5,1], [5,2]], [[4,0], [6,0]]],
"[5, 1]": [[[5,0], [5,2]], [[3,1], [1,0]]],
"[5, 2]": [[[5,1], [5,0]], [[1,2], [3,4]], [[4,2], [6,2]]],
"[6, 0]": [[[3,0], [0,0]], [[6,1], [6,2]], [[4,0], [5,0]]],
"[6, 1]": [[[6,0], [6,2]], [[4,1], [5,1]]],
"[6, 2]": [[[6,0], [6,1]], [[0,2], [3,5]], [[4,2], [5,2]]]
}


def printboard(game_type,state):
#    gameboard = nx.graph()
    if game_type == 3:
        print(str(state[0][0])+'-'+str(state[0][1])+'-'+str(state[0][2]))
        print('|\|/|')
        print(str(state[1][0])+'-'+str(state[1][1])+'-'+str(state[1][2]))
        print('|/|\|')
        print(str(state[2][0])+'-'+str(state[2][1])+'-'+str(state[2][2]))

    if game_type == 6:
        print(str(state[0][0])+'---'+str(state[0][1])+'---'+str(state[0][2]))
        print('|   |   |')
        print('| '+str(state[1][0])+'-'+str(state[1][1])+'-'+str(state[1][2])+' |')
        print('| |   | |')
        print(str(state[2][0])+'-'+str(state[2][1])+'   '+str(state[2][2])+'-'+str(state[2][3]))
        print('| |   | |')
        print('| '+str(state[3][0])+'-'+str(state[3][1])+'-'+str(state[3][2])+' |')
        print('|   |   |')
        print(str(state[4][0])+'---'+str(state[4][1])+'---'+str(state[4][2]))


    if game_type == 9:
        print(str(state[0][0])+'-----'+str(state[0][1])+'-----'+str(state[0][2]))
        print('|     |     |')
        print('| '+str(state[1][0])+'---'+str(state[1][1])+'---'+str(state[1][2])+' |')
        print('| |   |   | |')
        print('| | '+str(state[2][0])+'-'+str(state[2][1])+'-'+str(state[2][2])+' | |')
        print('| | |   | | |')
        print(str(state[3][0])+'-'+str(state[3][1])+'-'+str(state[3][2])+'   '+str(state[3][3])+'-'+str(state[3][4])+'-'+str(state[3][5]))
        print('| | |   | | |')
        print('| | '+str(state[4][0])+'-'+str(state[4][1])+'-'+str(state[4][2])+' | |')
        print('| |   |   | |')
        print('| '+str(state[5][0])+'---'+str(state[5][1])+'---'+str(state[5][2])+' |')
        print('|     |     |')
        print(str(state[6][0])+'-----'+str(state[6][1])+'-----'+str(state[6][2]))

    if game_type == 12:
        print(str(state[0][0])+'-----'+str(state[0][1])+'-----'+str(state[0][2]))
        print('|\    |    /|')
        print('| '+str(state[1][0])+'---'+str(state[1][1])+'---'+str(state[1][2])+' |')
        print('| |\  |  /| |')
        print('| | '+str(state[2][0])+'-'+str(state[2][1])+'-'+str(state[2][2])+' | |')
        print('| | |   | | |')
        print(str(state[3][0])+'-'+str(state[3][1])+'-'+str(state[3][2])+'   '+str(state[3][3])+'-'+str(state[3][4])+'-'+str(state[3][5]))
        print('| | |   | | |')
        print('| | '+str(state[4][0])+'-'+str(state[4][1])+'-'+str(state[4][2])+' | |')
        print('| |/  |  \| |')
        print('| '+str(state[5][0])+'---'+str(state[5][1])+'---'+str(state[5][2])+' |')
        print('|/    |    \|')
        print(str(state[6][0])+'-----'+str(state[6][1])+'-----'+str(state[6][2]))



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
            if state[move[0]][move[1]] == state[item[0][0]][item[0][1]] == state[item[1][0]][item[1][1]]:
                return True
        else:
            return False

    if game_type == 6:
        for item in mill_dict_6[str(move)]:
            if state[move[0]][move[1]] == state[item[0][0]][item[0][1]] == state[item[1][0]][item[1][1]]:
                return True
        else:
            return False


    if game_type == 9:
        for item in mill_dict_9[str(move)]:
            if state[move[0]][move[1]] == state[item[0][0]][item[0][1]] == state[item[1][0]][item[1][1]]:
                return True
        else:
            return False


    if game_type == 12:
        for item in mill_dict_12[str(move)]:
            if state[move[0]][move[1]] == state[item[0][0]][item[0][1]] == state[item[1][0]][item[1][1]]:
                return True
        else:
            return False



def free_space_finder(state):
    free_space = []
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] == 0:
                free_space.append([i,j])

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
