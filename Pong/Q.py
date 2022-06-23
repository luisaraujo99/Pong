
# state = (x_ball,y_ball,x_pad)

from os import environ
import numpy as np
import pickle
import math

GREEDY, EPS_GREEDY, STATE_LOC_GREEDY, WIND_LOC_GREEDY = 1, 2, 3, 4


class Q_AI:

    # default seed
    rng_fixed_seed = np.random.default_rng(2022)

    def __init__(self, learning_rate=1, discount_rate=0.97, X_Pad_dim=None, X_Grid_dim=None, Y_Grid_Dim=None, learning_decay=None, exploration_rate=1):
        self.q_matrix = np.zeros(
            (X_Pad_dim, Y_Grid_Dim, X_Grid_dim, 3))
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.learning_decay = learning_decay
        self.q_matrix_counter = np.zeros(
            (X_Pad_dim, Y_Grid_Dim, X_Grid_dim), dtype=np.int64)

    def q(self, action, reward, state, new_state, negative_propagation=True):
        arg_max_action = np.argmax(self.q_matrix[new_state])
        new_expected_val = self.q_matrix[new_state][arg_max_action]
        old_val = self.q_matrix[state][action]
        if negative_propagation:
            self.q_matrix[state][action] = old_val + \
                self.learning_rate*(reward+self.discount_rate *
                                    new_expected_val-old_val)
        else:
            self.q_matrix[state][action] = (1-self.learning_rate)*old_val\
                + self.learning_rate * \
                (reward+new_expected_val*self.discount_rate)

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
        r = rng_fixed_seed.random()
        if r < self.exploration_rate:
            return rng_fixed_seed.choice([0, 1, 2])
        else:
            return np.argmax(self.q_matrix[state])

    def state_local_epsilon_greedy(self, state, exploit_threshold=3):

        if self.q_matrix_counter[state] < exploit_threshold:
            return rng_fixed_seed.choice([0, 1, 2])
        else:
            return np.argmax(self.q_matrix[state])

    def neighbors(self, state, radius):
        _, Y, X = self.q_matrix_counter.shape
        x_pad, y, x = state
        y_min = 0 if y-radius < 0 else y-radius
        y_max = Y if y+radius+1 > Y else y+radius+1
        x_min = 0 if x-radius < 0 else x-radius
        x_max = X if x+radius+1 > X else x+radius+1
        return self.q_matrix_counter[x_pad, y_min:y_max, x_min:x_max]

    def radius_local_epsilon_greedy(self, state, exploit_threshold=3, radius=2):

        if np.mean(self.neighbors(state, radius)) < exploit_threshold:
            return rng_fixed_seed.choice([0, 1, 2])
        else:
            return np.argmax(self.q_matrix[state])

    def action_chooser_method(self, state, method, exploit):
        if method == EPS_GREEDY:
            return self.epsilon_greedy(state)
        elif method == GREEDY:
            return self.greedy(state)
        elif method == STATE_LOC_GREEDY:
            return self.state_local_epsilon_greedy(state, exploit_threshold=exploit)
        elif method == WIND_LOC_GREEDY:
            return self.radius_local_epsilon_greedy(state, exploit_threshold=exploit)

    # For values of A below 0.5, agent would be spending less time exploring and more time exploiting
    # B decides the slope of transition region between Exploration to Exploitation zone
    # C controls the steepness of left and right tail of the graph

    def exploration_rate_decay(self, time, EPISODES, A=0.5, B=0.1, C=0.1):
        standardized_time = (time-A*EPISODES)/(B*EPISODES)
        cosh = np.cosh(math.exp(-standardized_time))
        expr = 1.3-(1/cosh+(time*C/EPISODES))
        if expr < 0:
            self.exploration_rate = 0.001
        else:
            self.exploration_rate = expr

    def set_exploration_rate_decay(self, val):
        self.exploration_rate = val

    def learning_rate_decay(self, t, omega):
        self.learning_rate = 1/(t**omega)

    def full_matrix_softmax(self):
        X, Y, Z, _ = self.q_matrix.shape
        return [self.softmax((x, y, z)) for x in range(X)
                for y in range(Y) for z in range(Z)]

    def save_state(self, filename='last_state.txt'):
        file = open(filename, 'wb')
        pickle.dump(self, file)
        file.close()

    def matrix_ratio(self):
        return np.count_nonzero(self.q_matrix)/self.q_matrix.size

    def fitness_score(self, rewards, softmax):
        environment_knowledge = (
            self.q_matrix_counter > 20).sum()/self.q_matrix_counter.size
        rewards_ratio = (np.array(rewards) == 1).sum() / \
            (abs(np.array(rewards))).sum()
        max_min_distance = sum(
            [abs(max-min) for max, min in softmax])/self.q_matrix_counter.size
        return environment_knowledge*0.25+rewards_ratio*0.25+0.5*max_min_distance

    def load_file(self, filename="last_state.txt"):
        try:
            file = open(filename, 'rb')
            last_saved_object = pickle.load(file)
            self.q_matrix = last_saved_object.q_matrix
            self.learning_rate = last_saved_object.learning_rate
            self.exploration_rate = last_saved_object.exploration_rate
            self.q_matrix_counter = last_saved_object.q_matrix_counter
            file.close()
        except IOError:
            print("Could not read file")
            self.q_state_counter()

    def q_state_counter(self, state=None):
        if state != None:
            self.q_matrix_counter[state] += 1
