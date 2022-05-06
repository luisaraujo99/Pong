
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import numpy as np
from Pong import SinglePadPong
from Pong import Q_AI
import pygame
from alive_progress import alive_bar
from itertools import count
import matplotlib
matplotlib.rc('xtick', labelsize=7)
matplotlib.rc('ytick', labelsize=7)


WIDTH_SCALE, HEIGHT_SCALE = 120, 80
GAME_DIM = 10
PAD_SIZE = 2
WIDTH, HEIGHT = GAME_DIM*WIDTH_SCALE, GAME_DIM*HEIGHT_SCALE


class PongGame:
    def __init__(self, window, width, height):
        self.game = SinglePadPong(
            window, width, height, WIDTH_SCALE, HEIGHT_SCALE, PAD_SIZE, GAME_DIM)
        self.ball = self.game.ball
        self.paddle = self.game.paddle

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
                        x, y, str(round(np.max(matrix[(p_pos, y, x)]), 1)), va='center', ha='center', fontsize=7)

            axes.grid(False)
        fig.tight_layout()
        plt.show()

    def enqueue(self, rewards_queue, r):
        if len(rewards_queue) == 5:
            rewards_queue.pop(0)
        rewards_queue.append(0 if r == -1 else 1)

    def Q_learning_algorithm(self, epochs=10, episodes=100, show_v_plot=True, render=True):
        clock = pygame.time.Clock()
        run = True
        q_ai = Q_AI(learning_rate=1, discount_rate=0.97, Ndim=GAME_DIM, exploration_rate=1,
                    learning_decay=1/800, paddle_scale_len=PAD_SIZE)
        q_ai.load_file()

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
                        clock.tick(20)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                run = False
                                break

                    init_score = game_info.score

                    state = ((self.paddle.x//WIDTH_SCALE), (self.ball.y //
                                                            HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

                    #action = q_ai.epsilon_greedy(state)
                    action = q_ai.greedy(state)

                    # right
                    if action == 2:
                        self.game.paddle.move(
                            False, True, window_width=WIDTH)
                    # left
                    elif action == 1:
                        self.game.paddle.move(
                            False, False, window_width=WIDTH)

                    if render:
                        self.game.draw()
                        pygame.display.update()

                    game_info = self.game.loop()

                    end_score = self.game.score

                    r = self.reward(init_score, end_score)

                    if r == 1:
                        rewards_in_a_row += 1
                        if rewards_in_a_row == 7:
                            self.game.ball.reset()
                            rewards_in_a_row = 0

                    if abs(r) == 1:
                        self.enqueue(rewards_queue, r)
                        rewards.append(np.mean(rewards_queue))

                    new_state = ((self.paddle.x//WIDTH_SCALE), (self.ball.y //
                                                                HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

                    q_ai.q(action, r, state, new_state)

                    #################### SAVE TRAINING DATA ####################
                    v_max.append(q_ai.v(state))
                    v_min.append(q_ai.v_min(state))
                    v_mid.append(q_ai.v_mid(state))
                    q_ai.q_state_counter(state=state)
                    ############################################################

                    #######----- EXPLORATION RATE -----#######
                    #q_ai.exploration_rate_decay(time, episodes*epochs)
                    # q_ai.exploration_rate_decay2(
                    #     (q_ai.q_matrix_counter < 7).sum()/q_ai.q_matrix_counter.size)

                    # iteration
                    episode += 1
                    time += 1

                    # show training evolution
                    bar.text(
                        f'\n-> Exploration rate: {q_ai.exploration_rate}')
                    bar()

            #################### SAVE TRAINING DATA ####################
            exploration_rates.append(q_ai.exploration_rate)
            v_max_mean.append(np.mean(v_max))
            v_min_mean.append(np.mean(v_min))
            v_mid_mean.append(np.mean(v_mid))
            states_visited_ratio.append(
                (q_ai.q_matrix_counter < 7).sum()/q_ai.q_matrix_counter.size)
            learning_rate_evol.append(q_ai.learning_rate)
            ############################################################

            # learning rate evolution
            q_ai.learning_rate_decay(epoch)

            # iteration
            epoch += 1

            # save Q state
            q_ai.save_state()

        X, Y, Z, _ = q_ai.q_matrix.shape
        softmax = [q_ai.softmax((x, y, z)) for x in range(X)
                   for y in range(Y) for z in range(Z)]

        if show_v_plot:
            self.plot_v(X*Y*Z, epochs, v_max_mean,
                        v_min_mean, v_mid_mean, softmax, rewards, exploration_rates, states_visited_ratio, learning_rate_evol)
            if GAME_DIM == 10:
                self.plot_color_action_matrix(q_ai.q_matrix)
                self.plot_matrix_state_counter(q_ai.q_matrix_counter)
                self.plot_max_val_gradient(q_ai.q_matrix)

        # close pygame env
        pygame.quit()


def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Single Pad Pong")
    pong = PongGame(win, WIDTH, HEIGHT)
    plt.show()
    pong.Q_learning_algorithm()


main()
