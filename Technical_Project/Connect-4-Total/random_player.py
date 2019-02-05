import random

class Random_Player(object):
    def __init__(self, player):
        self.player = player

    def free_space_finder(self, state):
        free_space = []
        if len(free_space) == 7:
            return free_space
        for i in range(len(state)):
                if 0 in state[i]:
                    free_space.append((i,state[i].index(0)))
        return free_space

    def action(self,state):
        free_space = self.free_space_finder(state)
        temp = random.randint(0, len(free_space) - 1)
        if 7 in free_space:
            ind = free_space.index(7)
            del free_space[ind]
        return free_space[temp]
