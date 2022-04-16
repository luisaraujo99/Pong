
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


WIDTH_SCALE, HEIGHT_SCALE = 130, 70
GAME_DIM = 10
PAD_SIZE = 2
WIDTH, HEIGHT = GAME_DIM*WIDTH_SCALE, GAME_DIM*HEIGHT_SCALE


class PongGame:
    def __init__(self, window, width, height):
        self.game = SinglePadPong(
            window, width, height, WIDTH_SCALE, HEIGHT_SCALE)
        self.ball = self.game.ball
        self.paddle = self.game.paddle

    def reward(self, initial_score, end_score):
        return end_score - initial_score

    def plot_v(self, x_vals, y_vals, plot_index, state, q_ai):
        x_vals.append(next(plot_index))
        y_vals.append(q_ai.v(state))
        plt.cla()
        plt.plot(x_vals, y_vals)
        plt.pause(1e-10)

    def plot_color_action_matrix(self, q_matrix):

        pad_dim, y_dim, x_dim, action = q_matrix.shape
        max_matrix = np.zeros((pad_dim, y_dim, x_dim))

        for pad in range(pad_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    max_matrix[(pad, y, x)] = np.argmax(
                        q_matrix[(pad, y, x)])

        fig, axs = plt.subplots(3, 3, sharey=True)
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
        fig, axs = plt.subplots(3, 3)
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

        fig, axs = plt.subplots(3, 3)
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

    def Q_learning_algorithm(self, epochs=1, episodes=100, show_v_plot=True):
        clock = pygame.time.Clock()
        run = True
        q_ai = Q_AI(learning_rate=0.8, discount_rate=0.97, Ndim=GAME_DIM, exploit_rate=0.01,
                    epochs=epochs, episodes=episodes, paddle_scale_len=PAD_SIZE)
        q_ai.load_file()

        epoch = 0

        # plots
        plt.style.use('fivethirtyeight')
        plot_index = count()
        x_vals = []
        y_vals = []

        while epoch < epochs:
            initial_nonzeros_ratio = q_ai.matrix_ratio()
            episode = 0
            game_info = self.game.loop()

            with alive_bar(episodes, bar='blocks', title=f'Epoch {epoch}', spinner='arrows') as bar:
                while episode < episodes and run:

                    clock.tick()

                    init_score = game_info.score

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            run = False
                            break

                    state = ((self.paddle.x//WIDTH_SCALE), (self.ball.y //
                             HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

                    action = q_ai.prob_action(state)

                    # direita
                    if action == 2:
                        self.game.paddle.move(False, True, window_width=WIDTH)
                    # esquerda
                    elif action == 1:
                        self.game.paddle.move(False, False, window_width=WIDTH)

                    self.game.draw()
                    pygame.display.update()

                    game_info = self.game.loop()

                    end_score = self.game.score

                    r = self.reward(init_score, end_score)

                    new_state = ((self.paddle.x//WIDTH_SCALE), (self.ball.y //
                                                                HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))
                    q_ai.q(action, r, state, new_state)

                    # plots
                    if show_v_plot:
                        self.plot_v(x_vals, y_vals, plot_index, state, q_ai)
                    q_ai.q_state_counter(state=state)

                    episode += 1
                    bar.text(
                        f'\n-> Non zero ratio: {initial_nonzeros_ratio}')
                    bar()

            epoch += 1
            q_ai.inc_exploit_rate()
            print("exploitation rate: "+str(q_ai.exploit_rate))
            q_ai.save_state()

        if show_v_plot:
            self.plot_color_action_matrix(q_ai.q_matrix)
            self.plot_matrix_state_counter(q_ai.q_matrix_counter)
            self.plot_max_val_gradient(q_ai.q_matrix)
        pygame.quit()


def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong")
    pong = PongGame(win, WIDTH, HEIGHT)
    plt.show()
    pong.Q_learning_algorithm()


main()
