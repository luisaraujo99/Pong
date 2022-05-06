
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


WIDTH_SCALE, HEIGHT_SCALE = 16, 15
GAME_DIM = 50
PAD_SIZE = 8
WIDTH, HEIGHT = GAME_DIM*WIDTH_SCALE, GAME_DIM*HEIGHT_SCALE


class PongGame:
    def __init__(self, window, width, height):
        self.game = DoublePadPong(
            window, width, height, WIDTH_SCALE, HEIGHT_SCALE, PAD_SIZE, GAME_DIM)
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

    def plot_color_action_matrix(self, q_matrix):

        pad_dim, y_dim, x_dim, action = q_matrix.shape
        max_matrix = np.zeros((pad_dim, y_dim, x_dim))

        for pad in range(pad_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    max_matrix[(pad, y, x)] = np.argmax(
                        q_matrix[(pad, y, x)])

        fig, axs = plt.subplots(3, 3, sharey=True, figsize=(10, 10))
        fig.suptitle('Action per state', fontsize=13)
        cmap = colors.ListedColormap(['red', 'green', 'blue'])
        for p_pos in range(pad_dim):
            axes = axs[p_pos // 3, p_pos % 3]
            axes.set_title('paddle x: {}'.format(p_pos), fontsize=10)
            axes.matshow(
                max_matrix[p_pos], origin='lower', cmap=cmap)
            axes.grid(False)
        axs[0, 0].invert_yaxis()
        fig.tight_layout()
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

    def plot_max_val_gradient(self, matrix):
        pad_dim, y_dim, x_dim, action = matrix.shape
        max_matrix = np.zeros((pad_dim, y_dim, x_dim))

        for pad in range(pad_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    max_matrix[(pad, y, x)] = np.max(
                        matrix[(pad, y, x)])

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle('Max value per state', fontsize=13)
        for p_pos in range(pad_dim):
            axes = axs[p_pos // 3, p_pos % 3]
            axes.set_title('paddle x: {}'.format(p_pos), fontsize=10)
            axes.matshow(
                preprocessing.MinMaxScaler().fit_transform(max_matrix[p_pos]), cmap=plt.cm.Blues)
            for y in range(y_dim):
                for x in range(x_dim):
                    axes.text(
                        x, y, str(round(np.max(matrix[(p_pos, y, x)]), 1)), va='center', ha='center', fontsize=5)

            axes.grid(False)
        fig.tight_layout()
        plt.show()

    def enqueue(self, rewards_queue, r):
        if len(rewards_queue) == 5:
            rewards_queue.pop(0)
        rewards_queue.append(0 if r == -1 else 1)

    def Q_learning_algorithm(self, epochs=10, episodes=5000, show_v_plot=True, render=True):
        clock = pygame.time.Clock()
        run = True
        q_ai_1 = Q_AI(learning_rate=1, discount_rate=0.97, Ndim=GAME_DIM, exploration_rate=1,
                      learning_decay=1/300, paddle_scale_len=PAD_SIZE)

        q_ai_2 = Q_AI(learning_rate=1, discount_rate=0.97, Ndim=GAME_DIM, exploration_rate=1,
                      learning_decay=1/300, paddle_scale_len=PAD_SIZE)

        q_ai_1.load_file(filename="paddle1_last_state.txt")
        q_ai_2.load_file(filename="paddle2_last_state.txt")

        epoch = 0

        # plots
        plt.style.use('fivethirtyeight')

        v_max_mean = []
        v_mid_mean = []
        v_min_mean = []
        exploration_rates = []
        states_visited_ratio = []
        learning_rate_evol = []
        rewards_in_a_row = 0
        time = 0
        rewards = []
        rewards_queue = []

        while epoch < epochs:

            episode = 0
            game_info = self.game.loop()

            v_max = []
            v_mid = []
            v_min = []

            with alive_bar(episodes, bar='blocks', title=f'Epoch {epoch}', spinner='arrows') as bar:
                while episode < episodes and run:

                    if render:
                        clock.tick(25)
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

                    # right
                    if action_p1 == 2:
                        self.game.paddle1.move(
                            False, True, window_width=WIDTH)
                    # left
                    elif action_p1 == 1:
                        self.game.paddle1.move(
                            False, False, window_width=WIDTH)

                    # right
                    if action_p2 == 2:
                        self.game.paddle2.move(
                            False, True, window_width=WIDTH)
                    # left
                    elif action_p2 == 1:
                        self.game.paddle2.move(
                            False, False, window_width=WIDTH)

                    if render:
                        self.game.draw()
                        pygame.display.update()

                    game_info = self.game.loop()

                    end_score = self.game.score

                    r = self.reward(init_score, end_score)

                    if r == 1:
                        rewards_in_a_row += 1
                        if rewards_in_a_row == 5:
                            self.game.ball.reset()
                            rewards_in_a_row = 0

                    if abs(r) == 1:
                        self.enqueue(rewards_queue, r)
                        rewards.append(np.mean(rewards_queue))

                    new_state_p1 = ((self.paddle1.x//WIDTH_SCALE), (self.ball.y //
                                                                    HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

                    new_state_p2 = ((self.paddle1.x//WIDTH_SCALE), (self.ball.y //
                                                                    HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

                    q_ai_1.q(action_p1, r, state_p1, new_state_p1)
                    q_ai_2.q(action_p2, r, state_p2, new_state_p2)

                    #################### SAVE TRAINING DATA ####################
                    # v_max.append(q_ai_1.v(state_p1))
                    # v_min.append(q_ai_1.v_min(state_p1))
                    # v_mid.append(q_ai_1.v_mid(state_p1))
                    q_ai_1.q_state_counter(state=state_p1)
                    q_ai_2.q_state_counter(state=state_p2)
                    # ############################################################

                    #######----- EXPLORATION RATE -----#######
                    #q_ai.exploration_rate_decay(time, episodes*epochs)
                    q_ai_1.exploration_rate_decay2(
                        (q_ai_1.q_matrix_counter < 2).sum()/q_ai_1.q_matrix_counter.size)

                    q_ai_2.exploration_rate_decay2(
                        (q_ai_2.q_matrix_counter < 2).sum()/q_ai_2.q_matrix_counter.size)

                    # iteration
                    episode += 1
                    time += 1

                    # show training evolution
                    bar.text(
                        f'\n-> Exploration rate: {q_ai_1.exploration_rate}')
                    bar()

            #################### SAVE TRAINING DATA ####################
            # exploration_rates.append(q_ai_1.exploration_rate)
            # v_max_mean.append(np.mean(v_max))
            # v_min_mean.append(np.mean(v_min))
            # v_mid_mean.append(np.mean(v_mid))
            # states_visited_ratio.append(
            #     (q_ai_1.q_matrix_counter < 5).sum()/q_ai_1.q_matrix_counter.size)
            # learning_rate_evol.append(q_ai_1.learning_rate)
            ############################################################

            # learning rate evolution
            q_ai_1.learning_rate_decay(epoch)
            q_ai_2.learning_rate_decay(epoch)

            # iteration
            epoch += 1

            # save Q state
            q_ai_1.save_state(filename="paddle1_last_state.txt")
            q_ai_2.save_state(filename="paddle2_last_state.txt")

        X, Y, Z, _ = q_ai_1.q_matrix.shape
        softmax = [q_ai_1.softmax((x, y, z)) for x in range(X)
                   for y in range(Y) for z in range(Z)]

        # if show_v_plot:
        #     self.plot_v(X*Y*Z, epochs, v_max_mean,
        #                 v_min_mean, v_mid_mean, softmax, rewards, exploration_rates, states_visited_ratio, learning_rate_evol)
        #     if GAME_DIM == 10:
        #         self.plot_color_action_matrix(q_ai_1.q_matrix)
        #         self.plot_matrix_state_counter(q_ai_1.q_matrix_counter)
        #         self.plot_max_val_gradient(q_ai_1.q_matrix)

        # close pygame env
        pygame.quit()


def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Single Pad Pong")
    pong = PongGame(win, WIDTH, HEIGHT)
    plt.show()
    pong.Q_learning_algorithm()


main()
