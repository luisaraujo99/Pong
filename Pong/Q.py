
# state = (x_ball,y_ball,x_pad)

import numpy as np
import pickle
import random
import math


class Q_AI:

    def __init__(self, learning_rate, discount_rate, X_Pad_dim, X_Grid_dim, Y_Grid_Dim, learning_decay):
        self.q_matrix = np.zeros(
            (X_Pad_dim, Y_Grid_Dim, X_Grid_dim, 3))
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = 1
        self.learning_decay = learning_decay
        self.q_matrix_counter = np.zeros(
            (X_Pad_dim, Y_Grid_Dim, X_Grid_dim), dtype=np.int64)

    def q(self, action, reward, state, new_state):
        arg_max_action = np.argmax(self.q_matrix[new_state])
        new_expected_val = self.q_matrix[new_state][arg_max_action]
        old_val = self.q_matrix[state][action]

        self.q_matrix[state][action] = (
            1-self.learning_rate)*old_val + self.learning_rate*(reward+new_expected_val*self.discount_rate)

        return self.q_matrix[state][action]

    def v(self, state):
        return np.max(self.q_matrix[state])

    def v_min(self, state):
        return np.min(self.q_matrix[state])

    def v_mid(self, state):
        return np.sort(self.q_matrix[state])[-2]

    def softmax(self, state):
        res = np.exp(self.q_matrix[state])/sum(np.exp(self.q_matrix[state]))
        return (np.max(res), np.min(res))

    def greedy(self, state):

        return np.argmax(self.q_matrix[state])

    def epsilon_greedy(self, state):
        r = random.random()
        if r < self.exploration_rate:
            return random.choice([0, 1, 2])
        else:
            return np.argmax(self.q_matrix[state])

    # For values of A below 0.5, agent would be spending less time exploring and more time exploiting
    # B decides the slope of transition region between Exploration to Exploitation zone
    # C controls the steepness of left and right tail of the graph
    def exploration_rate_decay(self, time, EPISODES, A=0.4, B=0.5, C=0.1):
        standardized_time = (time-A*EPISODES)/(B*EPISODES)
        cosh = np.cosh(math.exp(-standardized_time))
        expr = 1.3-(1/cosh+(time*C/EPISODES))
        if expr < 0:
            self.exploration_rate = 0.001
        else:
            self.exploration_rate = expr

    def exploration_rate_decay2(self, val):
        self.exploration_rate = val

    def learning_rate_decay(self, time):
        self.learning_rate = math.exp(-self.learning_decay*time)

    def save_state(self, filename='last_state.txt'):
        file = open(filename, 'wb')
        pickle.dump(self, file)
        file.close()

    def matrix_ratio(self):
        return np.count_nonzero(self.q_matrix)/self.q_matrix.size

    def load_file(self, filename="last_state.txt"):
        try:
            file = open(filename, 'rb')
            last_saved_object = pickle.load(file)
            self.q_matrix = last_saved_object.q_matrix
            self.learning_rate = last_saved_object.learning_rate
            # self.discount_rate = last_saved_object.discount_rate
            # self.exploit_rate = last_saved_object.exploit_rate
            self.q_matrix_counter = last_saved_object.q_matrix_counter
            file.close()
        except IOError:
            print("Could not read file")
            self.q_state_counter()

    def q_state_counter(self, state=None):
        if state != None:
            self.q_matrix_counter[state] += 1
