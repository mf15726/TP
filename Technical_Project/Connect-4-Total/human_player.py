from random import randint
import numpy as np
import pandas as pd
import os
import os.path

class Human_Player(object):
    def __init__(self, player):
        self.player = player

    def action(self, state):
        index = input('Choose Column (Counting from 0) : ')
        move = [index,state[index].index(0)]
        return move
#        return (int(move.split(',')[0]),int(move.split(',')[1]))
