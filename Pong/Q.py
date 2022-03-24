
# state = (x_ball,y_ball,x_pad)

import numpy as np
import pickle


class Q_AI:

    def __init__(self, learning_rate, discount_rate, Ndim,exploit_rate,epochs,episodes):
        self.q_matrix = np.zeros((Ndim+2, Ndim+2, Ndim+2, 3))
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploit_rate = exploit_rate
        self.exploit_inc = episodes*epochs
        

    def q(self, action, reward, state, new_state):
        arg_max_action = np.argmax(self.q_matrix[new_state])
        new_expected_val = self.q_matrix[new_state][arg_max_action]
        old_val = self.q_matrix[state][action]
        self.q_matrix[state][action] = (
            1-self.learning_rate)*old_val + self.learning_rate*(reward+new_expected_val*self.discount_rate)
      
        
        return self.q_matrix[state][action]

    def prob_action(self, state):

        prob_array = np.power(self.exploit_rate, self.q_matrix[state])/np.sum(
            np.power(self.exploit_rate, self.q_matrix[state]))

        if self.exploit_rate < 1:
            self.exploit_rate += (1-self.exploit_rate)/self.exploit_inc
        else:
            self.exploit_rate =1

        
        return np.random.choice([0, 1, 2], p=prob_array)

    def save_state(self):
        file = open('last_state.txt', 'wb')
        pickle.dump(self, file)
        file.close()
        
    def matrix_ratio(self):
        return np.count_nonzero(self.q_matrix)/((40**3)*3)

    def load_file(self):
        try:
            file = open('last_state.txt', 'rb')
            last_saved_object = pickle.load(file)
            self.q_matrix = last_saved_object.q_matrix
            self.learning_rate = last_saved_object.learning_rate
            self.discount_rate = last_saved_object.discount_rate
            #self.exploit_rate = last_saved_object.exploit_rate
            file.close()
        except IOError:
            print("Could not read file")
