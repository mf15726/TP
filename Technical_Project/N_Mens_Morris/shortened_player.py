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



class Shortened_Player(object):

	def __init__(self, epsilon, alpha, gamma, limit):

		self.sess = tf.Session()
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.limit = limit
		
		self.to_index = [(None, None, None)] * self.limit
		self.from_index = [(None, None, None)] * (self.limit - 6)
		self.remove_index = [(None, None, None)] * 19
		
		self.to_future_index = [None] * self.limit
		self.from_future_index = [None] * (self.limit - 6)
		self.remove_future_index = [None] * 19
