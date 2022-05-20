
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import numpy as np
from Pong import DoublePadPong
from Pong import Q_AI
import pygame
from alive_progress import alive_bar
from itertools import count
import matplotlib

from Pong.doublePadPong import DoublePadPong
matplotlib.rc('xtick', labelsize=7)
matplotlib.rc('ytick', labelsize=7)


WIDTH_SCALE, HEIGHT_SCALE = 16, 14
GAME_DIM_X, GAME_DIM_Y = 50, 60
PAD_SIZE = 8
WIDTH, HEIGHT = GAME_DIM_X*WIDTH_SCALE, GAME_DIM_Y*HEIGHT_SCALE
X_PAD_DIM = GAME_DIM_X-(PAD_SIZE-1)


class PongGame:
    def __init__(self, window, width, height):
        self.game = DoublePadPong(
            window, width, height, WIDTH_SCALE, HEIGHT_SCALE, PAD_SIZE, GAME_DIM_X, GAME_DIM_Y)
        self.ball = self.game.ball
        self.paddle1 = self.game.paddle1
        self.paddle2 = self.game.paddle2

    def reward(self, initial_score, end_score):
        return end_score - initial_score

    def plot_v(self, matrix_range, epochs, vmax, vmin, vmid, softmax, rewards, exploration_rates, states_visited_ratio, learning_rate_evol):

        xvals = [i for i in range(epochs)]
        x2vals = [i for i in range(matrix_range)]
        xvals3 = [i for i in range(len(rewards))]
        fig, axs = plt.subplots(4)
        axs[0].set_title("Matrix values Mean per epoch",
                         fontdict={'fontsize': 10})
        axs[0].plot(xvals, vmax, color='r', linewidth=1)
        axs[0].plot(xvals, vmin, color='g', linewidth=1)
        axs[0].plot(xvals, vmid, color='b', linewidth=1)
        axs[0].legend(['vmax', "vmin", "vmid"])
        axs[1].set_title(
            "Softmax of Matrix values after training", fontdict={'fontsize': 10})
        axs[1].plot(x2vals, [Max for (Max, Min) in softmax],
                    color='r', linewidth=1)
        axs[1].plot(x2vals, [Min for (Max, Min) in softmax],
                    color='b', linewidth=1)
        axs[1].legend(["max", "min"])
        axs[2].set_title(
            "Rewards ratio per epoch and exploration rate", fontdict={'fontsize': 10})
        axs[2].plot(xvals, exploration_rates, linewidth=1)
        axs[2].plot(xvals, states_visited_ratio, linewidth=1)
        axs[2].plot(xvals, learning_rate_evol, linewidth=1)
        axs[2].legend(["explor. rate",
                      "stat. visited", "learning rate"], fontsize=8)
        axs[3].set_title(
            "Total rewards in last 20 episodes", fontdict={'fontsize': 10})
        axs[3].plot(xvals3, rewards, linewidth=1)

        plt.show()

    def plot_matrix_state_counter(self, matrix_counter):
        pad_dim, y_dim, x_dim = matrix_counter.shape
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle('State visits counter', fontsize=13)
        for p_pos in range(pad_dim):
            axes = axs[p_pos // 3, p_pos % 3]
            axes.set_title('paddle x: {}'.format(p_pos), fontsize=10)
            axes.matshow(matrix_counter[p_pos], cmap=plt.cm.Blues)
            for y in range(y_dim):
                for x in range(x_dim):
                    axes.text(
                        x, y, str(matrix_counter[(p_pos, y, x)]), va='center', ha='center', fontsize=5)

            axes.grid(False)
        fig.tight_layout()
        plt.show()

    def enqueue(self, rewards_queue, r):
        """ Function to help creating a sliding window to save game status"""
        if len(rewards_queue) == 5:
            rewards_queue.pop(0)
        rewards_queue.append(0 if r == -1 else 1)

    def Q_learning_algorithm(self, epochs=2000, episodes=1000, show_v_plot=True, render=False):
        clock = pygame.time.Clock()
        run = True

        ###### Declaring Q_AI Instances ######
        q_ai_1 = Q_AI(learning_rate=1, discount_rate=0.97, X_Pad_dim=GAME_DIM_X-(PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1,
                      learning_decay=1/1200)

        q_ai_2 = Q_AI(learning_rate=1, discount_rate=0.97, X_Pad_dim=GAME_DIM_X-(PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1,
                      learning_decay=1/1200)

        filename1 = "LVL2_P1_x={x}_y={y}.txt".format(
            x=GAME_DIM_X, y=GAME_DIM_Y)
        filename2 = "LVL2_P2_x={x}_y={y}.txt".format(
            x=GAME_DIM_X, y=GAME_DIM_Y)

        q_ai_1.load_file(filename=filename1)
        q_ai_2.load_file(filename=filename2)

        # plots
        plt.style.use('fivethirtyeight')

        # aux arrays
        v_max_mean1, v_max_mean2 = np.zeros(epochs), np.zeros(epochs)
        v_mid_mean1, v_mid_mean2 = np.zeros(epochs), np.zeros(epochs)
        v_min_mean1, v_min_mean2 = np.zeros(epochs), np.zeros(epochs)
        exploration_rates1, exploration_rates2 = np.zeros(
            epochs), np.zeros(epochs)
        states_visited_ratio1, states_visited_ratio2 = np.zeros(
            epochs), np.zeros(epochs)
        learning_rate_evol1, learning_rate_evol2 = np.zeros(
            epochs), np.zeros(epochs)
        rewards1, rewards2 = [], []
        rewards_queue1, rewards_queue2 = [], []

        # variables
        time, epoch, rewards_in_a_row1, rewards_in_a_row2 = 0, 0, 0, 0

        while epoch < epochs:

            episode = 0
            game_info = self.game.loop()

            v_max1, v_max2, v_mid1, v_mid2, v_min1, v_min2 = np.zeros(
                episodes), np.zeros(episodes), np.zeros(episodes), np.zeros(episodes), np.zeros(episodes), np.zeros(episodes)

            with alive_bar(episodes, bar='blocks', title=f'Epoch {epoch}', spinner='arrows') as bar:
                while episode < episodes and run:

                    if render:
                        clock.tick(20)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                run = False
                                break

                    init_score = game_info.score

                    state_p1 = ((self.paddle1.x//WIDTH_SCALE), (self.ball.y //
                                                                HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

                    state_p2 = ((self.paddle2.x//WIDTH_SCALE), (self.ball.y //
                                                                HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

                    action_p1 = q_ai_1.epsilon_greedy(state_p1)
                    action_p2 = q_ai_2.epsilon_greedy(state_p2)
                    #action = q_ai.greedy(state)

                    ##########################################
                    ################# PAD 1 ##################
                    # right
                    if action_p1 == 2:
                        self.game.paddle1.move(
                            False, True, window_width=WIDTH)
                    # left
                    elif action_p1 == 1:
                        self.game.paddle1.move(
                            False, False, window_width=WIDTH)

                    ################# PAD 2 ##################
                    # right
                    if action_p2 == 2:
                        self.game.paddle2.move(
                            False, True, window_width=WIDTH)
                    # left
                    elif action_p2 == 1:
                        self.game.paddle2.move(
                            False, False, window_width=WIDTH)
                    ##########################################
                    ##########################################

                    if render:
                        self.game.draw()
                        pygame.display.update()

                    game_info = self.game.loop()

                    end_score = self.game.score

                    r = self.reward(init_score, end_score)

                    new_state_p1 = ((self.paddle1.x//WIDTH_SCALE), (self.ball.y //
                                                                    HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

                    new_state_p2 = ((self.paddle1.x//WIDTH_SCALE), (self.ball.y //
                                                                    HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

                    q_ai_1.q(action_p1, r, state_p1, new_state_p1)
                    q_ai_2.q(action_p2, r, state_p2, new_state_p2)

                    if r == 1:
                        rewards_in_a_row1 += 1
                        rewards_in_a_row2 += 1
                        if rewards_in_a_row1 == 15 or rewards_in_a_row2 == 15:
                            self.game.ball.reset()
                            rewards_in_a_row1 = 0
                            rewards_in_a_row2 = 0

                    if abs(r) == 1:
                        self.enqueue(rewards_queue1, r)
                        self.enqueue(rewards_queue2, r)
                        rewards1.append(np.mean(rewards_queue1))
                        rewards2.append(np.mean(rewards_queue2))

                    

                    #################### SAVE TRAINING DATA ####################
                    # pad1
                    v_max1[episode] = q_ai_1.v(state_p1)
                    v_min1[episode] = q_ai_1.v_min(state_p1)
                    v_mid1[episode] = q_ai_1.v_mid(state_p1)
                    q_ai_1.q_state_counter(state=state_p1)
                    # pad2
                    v_max2[episode] = q_ai_2.v(state_p2)
                    v_min2[episode] = q_ai_2.v_min(state_p2)
                    v_mid2[episode] = q_ai_2.v_mid(state_p2)
                    q_ai_2.q_state_counter(state=state_p2)
                    # ############################################################

                    #######----- EXPLORATION RATE -----#######
                    #q_ai_1.exploration_rate_decay(time, episodes*epochs)
                    #q_ai_2.exploration_rate_decay(time, episodes*epochs)
                    q_ai_1.exploration_rate_decay2(
                        ((q_ai_1.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]) < 2).sum()/(q_ai_1.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]).size)

                    q_ai_2.exploration_rate_decay2(
                        ((q_ai_2.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]) < 2).sum()/(q_ai_1.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]).size)

                    # iteration
                    episode += 1
                    time += 1

                    # show training evolution
                    bar.text(
                        f'\n-> Exploration rate: {q_ai_1.exploration_rate}')
                    bar()

            #################### SAVE TRAINING DATA ####################
            exploration_rates1[epoch] = q_ai_1.exploration_rate
            v_max_mean1[epoch] = np.mean(v_max1)
            v_min_mean1[epoch] = np.mean(v_min1)
            v_mid_mean1[epoch] = np.mean(v_mid1)
            states_visited_ratio1[epoch] = (
                q_ai_1.q_matrix_counter < 1).sum()/q_ai_1.q_matrix_counter.size
            learning_rate_evol1[epoch] = q_ai_1.learning_rate
            # pad 2 stats
            exploration_rates2[epoch] = q_ai_2.exploration_rate
            v_max_mean2[epoch] = np.mean(v_max2)
            v_min_mean2[epoch] = np.mean(v_min2)
            v_mid_mean2[epoch] = np.mean(v_mid2)
            states_visited_ratio2[epoch] = (
                q_ai_2.q_matrix_counter < 1).sum()/q_ai_2.q_matrix_counter.size
            learning_rate_evol2[epoch] = q_ai_2.learning_rate
            ############################################################

            # learning rate evolution
            q_ai_1.learning_rate_decay(epoch)
            q_ai_2.learning_rate_decay(epoch)

            # iteration
            epoch += 1

            # save Q state
            q_ai_1.save_state(filename=filename1)
            q_ai_2.save_state(filename=filename2)

        X, Y, Z, _ = q_ai_1.q_matrix.shape
        softmax1 = [q_ai_1.softmax((x, y, z)) for x in range(X)
                    for y in range(Y) for z in range(Z)]
        softmax2 = [q_ai_2.softmax((x, y, z)) for x in range(X)
                    for y in range(Y) for z in range(Z)]

        if show_v_plot:
            self.plot_v(X*Y*Z, epochs, v_max_mean1,
                        v_min_mean1, v_mid_mean1, softmax1, rewards1, exploration_rates1, states_visited_ratio1, learning_rate_evol1)
            self.plot_v(X*Y*Z, epochs, v_max_mean2,
                        v_min_mean2, v_mid_mean2, softmax2, rewards2, exploration_rates2, states_visited_ratio2, learning_rate_evol2)

        # close pygame env
        pygame.quit()


def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Double Pad Pong")
    pong = PongGame(win, WIDTH, HEIGHT)
    plt.show()
    pong.Q_learning_algorithm()


main()
