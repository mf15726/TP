#Packages
import numpy as np
import pandas as pd
import random
from random import randint
from copy import deepcopy
import csv
import matplotlib
#import matplotlib.pyplot as plt
import tensorflow as tf
from math import log
#Classes
from learned_player import Learned_Player
from random_player import Random_Player
from human_player import Human_Player


def end_game(state):
    state_mod = deepcopy(state)
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] == 0:
                continue
            if i < 4:
                if state[i][j] == state[i+1][j] == state[i+2][j] == state[i+3][j]:
                    game_winner = state[i][j]
                    return game_winner
            if j < 3:
                if state[i][j] == state[i][j+1] == state[i][j+2] == state[i][j+3]:
                    game_winner = state[i][j]
                    return game_winner
            if i < 4 and j < 3:
                if state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3]:
                    game_winner = state[i][j]
                    return game_winner
                if state[i][j+3] == state[i+1][j+2] == state[i+2][j+1] == state[i+3][j]:
                    game_winner = state[i][j+3]
                    return game_winner
    else:
        game_winner = 0
        return game_winner


def printboard(state):
    for i in reversed(range(len(state[0]))):
        line_list = []
        for j in range(len(state)):
            line_list.append(state[j][i])
        line = '  '.join(str(x) for x in line_list)
        print(line)
    print('\n')

def game_play(player1, player2, print_board):
    state = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    game_states = []
    state_index = []
    if print_board:
        printboard(state)
    for i in range(42):
        if i % 2 == 0:
            move = player1.action(state)
        else:
            move = player2.action(state)
        if isinstance(player1, Learned_Player):
            player1.state_index.append((deepcopy(state),move))
        if isinstance(player2, Learned_Player):
            player2.state_index.append((deepcopy(state),move))
        state[move[0]][move[1]] = (i % 2) + 1
        if print_board:
            printboard(state)
        new_state = deepcopy(state)
        game_states.append(new_state)
        winner = end_game(state)
        if winner != 0:
            break
    return winner, game_states

def play_and_learn(iterations, player1, player2, test_at, test_no, print_board):
    winner1_plt_list = []
    winner2_plt_list = []
    for i in range(iterations):
        winner, game_states = game_play(player1,player2,print_board=print_board)
#        print 'Winner = ' + str(winner)
        if isinstance(player1, Learned_Player):
            if i != 0:
                player1.learn(winner)
            if i % test_at == 0:
                print 'iterations = ' + str(i)
                winner1_per = test_against(test_no,player1,random_player2)
                winner1_plt_list.append([winner1_per,i])
                print 'Player 1 winrate vs Random = ' + str(winner1_per)
        if isinstance(player2, Learned_Player):
            if i != 0:
                player2.learn(winner)
            if i % test_at == 0:
                winner2_per = test_against(test_no,random_player1,player2)
                winner2_plt_list.append([winner2_per,i])
                print 'Player 2 winrate vs Random = ' + str(winner2_per)
    return winner1_plt_list, winner2_plt_list

def test_against(iterations,test_agent,opponent):
    winner_list = []
    if test_agent.player == 1:
        for i in range(iterations):
            winner,_ = game_play(test_agent,opponent,print_board=False)
            winner_list.append((winner + test_agent.player) % 3)
    else:
        for i in range(iterations):
            winner,_ = game_play(opponent,test_agent,print_board=False)
            winner_list.append(((2 * winner) + 1) % 3)
    winner_per = float(sum(winner_list))/float(len(winner_list))
    return winner_per

#def graph_writer(input_list,test_at):
#    plt.title('Average Score obtained by the Learned Player')
#    plt.x_label('Number of Learning Iterations')
#    plt.y_label('Average Score')
#    temp = np.linspace[0,len(input_list)*test_at,len(input_list)]
#    temp = list(temp)
#    print(input_list)
#    plt.plot(input_list,temp)

#    plt.x_label('Number of Iterations')
#    plt.y_label('Score against Random Player')
#    plt.show()

def save_csv(input_list):
    writer.writerow(['Winner Percentage', 'Iteration Number'])
    for i in range(len(input_list)):
        printable = [input_list[i][0],input_list[i][1]]
        writer.writerow(printable)

random_player1 = Random_Player(player=1)
random_player2 = Random_Player(player=2)

human_player1 = Human_Player(player=1)
human_player2 = Human_Player(player=2)

learned_player1 = Learned_Player(player=1,alpha=0.9,epsilon=0.9,gamma=0.9)
learned_player2 = Learned_Player(player=2,alpha=0.9,epsilon=0.9,gamma=0.9)
learned_player1.sess.run(tf.global_variables_initializer())
learned_player2.sess.run(tf.global_variables_initializer())

test_at = 100
test_no = 100

#winner, _ = game_play(random_player1, learned_player2)
winner1_test, winner2_test = play_and_learn(4000000, learned_player1, learned_player2,  test_at, test_no, print_board=False)
with open('Connect-4_data1.csv', 'wb') as file:
    writer = csv.writer(file)
    save_csv(winner1_test)
with open('Connect-4_data2.csv', 'wb') as file:
    writer = csv.writer(file)
    save_csv(winner2_test)
#graph_writer(winner1_test, test_at)
#    winner,_ = game_play(human_player1, human_player2,print_board=True)
