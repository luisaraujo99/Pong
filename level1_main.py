
import matplotlib.pyplot as plt
import numpy as np
from Pong import SinglePadPong
from Pong import Q_AI
import pygame
from alive_progress import alive_bar
from matplotlib.animation import FuncAnimation
from itertools import count


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

    def plot(self, ax, episodes, values, episode, val):

        episodes.append(episode)
        values.append(val)
        ax.plot(episodes, values,
                linewidth=2,
                color='red')
        return

    def Q_learning_algorithm(self, epochs=1000, episodes=500):
        clock = pygame.time.Clock()
        run = True
        q_ai = Q_AI(0.02, 0.97, GAME_DIM, 0.6, epochs, episodes, PAD_SIZE)
        q_ai.load_file()
        q_ai.q_matrix_to_max_by_state()

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

                    state = ((self.ball.x//WIDTH_SCALE), (self.ball.y //
                             HEIGHT_SCALE), (self.paddle.x//WIDTH_SCALE))

                    action = q_ai.prob_action(state)

                    if action == 2:
                        self.game.paddle.move(False, True, window_width=WIDTH)
                    elif action == 1:
                        self.game.paddle.move(False, False, window_width=WIDTH)

                    self.game.draw()
                    pygame.display.update()

                    game_info = self.game.loop()

                    end_score = self.game.score

                    r = self.reward(init_score, end_score)

                    new_state = ((self.ball.x//WIDTH_SCALE), (self.ball.y //
                                                              HEIGHT_SCALE), (self.paddle.x//WIDTH_SCALE))
                    q_ai.q(action, r, state, new_state)

                    x_vals.append(next(plot_index))
                    y_vals.append(q_ai.v(state))
                    plt.cla()
                    plt.plot(x_vals, y_vals)
                    plt.pause(0.1)

                    episode += 1
                    bar.text(
                        f'\n-> Non zero ratio: {initial_nonzeros_ratio}')
                    bar()

            epoch += 1
            q_ai.inc_exploit_rate()
            q_ai.save_state()

        pygame.quit()


def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong")
    pong = PongGame(win, WIDTH, HEIGHT)
    plt.show()
    pong.Q_learning_algorithm()


main()
