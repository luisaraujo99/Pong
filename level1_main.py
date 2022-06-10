import matplotlib.pyplot as plt
import numpy as np
from Pong import SinglePadPong
from Pong import Q_AI
from Pong.PlotTool import *
import pygame
from alive_progress import alive_bar
from itertools import count
import os

WIDTH_SCALE, HEIGHT_SCALE = 24, 22
GAME_DIM_X, GAME_DIM_Y = 40, 40
PAD_SIZE = 8
WIDTH, HEIGHT = GAME_DIM_X*WIDTH_SCALE, GAME_DIM_Y*HEIGHT_SCALE
EPS_GREEDY, GREEDY, STATE_LOC_GREEDY, WIND_LOC_GREEDY = 1, 2, 3, 4

plt.style.use('fivethirtyeight')
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)


class PongGame:
    def __init__(self, window, width, height):
        self.game = SinglePadPong(
            window, width, height, WIDTH_SCALE, HEIGHT_SCALE, PAD_SIZE, GAME_DIM_X, GAME_DIM_Y)
        self.ball = self.game.ball
        self.paddle = self.game.paddle

    def reward(self, initial_score, end_score):
        return end_score - initial_score

    def enqueue(self, rewards_queue, r):
        if len(rewards_queue) == 30:
            rewards_queue.pop(0)
        rewards_queue.append(0 if r == -1 else 1)

    def Q_learning_algorithm(self, epochs=200, episodes=5000, show_v_plot=True, render=True, Action_method=EPS_GREEDY, exploration_rate=0, visits_threshold=20, reset_on=10):
        clock = pygame.time.Clock()
        run = True
        q_ai = Q_AI(exploration_rate=exploration_rate, X_Pad_dim=GAME_DIM_X-(PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1,
                    learning_decay=1/4000)
        filename = "./level1_results/LVL1_x={x}_y={y}_Method={m}_epochs={e}_vt={vt}_reseton={reset_on}.txt".format(
            x=GAME_DIM_X, y=GAME_DIM_Y, m=Action_method, e=epochs, vt=visits_threshold, reset_on=reset_on)
        q_ai.load_file(filename=filename)

        epoch = 0

        # plots

        v_max_mean = np.zeros(epochs)
        v_mid_mean = np.zeros(epochs)
        v_min_mean = np.zeros(epochs)
        exploration_rates = np.zeros(epochs)
        states_visited_ratio = np.zeros(epochs)
        learning_rate_evol = np.zeros(epochs)
        rewards_in_a_row = 0
        time = 0
        rewards = []
        rewards_seq = []
        rewards_queue = []

        if not render:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        while epoch < epochs:

            episode = 0
            game_info = self.game.loop()

            v_max = np.zeros(episodes)
            v_mid = np.zeros(episodes)
            v_min = np.zeros(episodes)

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

                    action = q_ai.action_chooser_method(
                        state, Action_method)

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

                    new_state = ((self.paddle.x//WIDTH_SCALE), (self.ball.y //
                                                                HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

                    q_ai.q(action, r, state, new_state)

                    if abs(r) == 1:
                        self.enqueue(rewards_queue, r)
                        rewards.append(np.mean(rewards_queue))
                        rewards_seq.append(r)

                    if r == 1:
                        rewards_in_a_row += 1
                        if rewards_in_a_row == reset_on:
                            self.game.ball.reset()
                            rewards_in_a_row = 0

                    #################### SAVE TRAINING DATA ####################
                    v_max[episode] = q_ai.v(state)
                    v_min[episode] = q_ai.v_min(state)
                    v_mid[episode] = q_ai.v_mid(state)
                    q_ai.q_state_counter(state=state)
                    ############################################################

                    #######----- EXPLORATION RATE -----#######
                    #q_ai.exploration_rate_decay(time, episodes*epochs)
                    q_ai.exploration_rate_decay2(
                        (q_ai.q_matrix_counter < visits_threshold).sum()/q_ai.q_matrix_counter.size)

                    # iteration
                    episode += 1
                    time += 1

                    # show training evolution
                    bar.text(
                        f'\n-> Exploration rate: {q_ai.exploration_rate}')
                    bar()

            #################### SAVE TRAINING DATA ####################
            exploration_rates[epoch] = q_ai.exploration_rate
            v_max_mean[epoch] = np.mean(v_max)
            v_min_mean[epoch] = np.mean(v_min)
            v_mid_mean[epoch] = np.mean(v_mid)
            states_visited_ratio[epoch] = (
                q_ai.q_matrix_counter < 1).sum()/q_ai.q_matrix_counter.size
            learning_rate_evol[epoch] = q_ai.learning_rate
            ############################################################

            # learning rate evolution
            q_ai.learning_rate_decay(epoch)

            # iteration
            epoch += 1

            # save Q state
            q_ai.save_state(filename=filename)

        X, Y, Z, _ = q_ai.q_matrix.shape
        softmax = [q_ai.softmax((x, y, z)) for x in range(X)
                   for y in range(Y) for z in range(Z)]

        if show_v_plot:
            plot_v(X*Y*Z, epochs, v_max_mean,
                   v_min_mean, v_mid_mean, softmax, rewards, exploration_rates, states_visited_ratio, learning_rate_evol, filename.replace('txt', 'png'))
            if GAME_DIM_Y == 10:
                plot_color_action_matrix(q_ai.q_matrix)
                plot_matrix_state_counter(q_ai.q_matrix_counter)
                plot_max_val_gradient(q_ai.q_matrix)

        # close pygame env
        pygame.quit()
        return q_ai.fitness_score(rewards_seq, softmax)


def genetic_algorithm():
    agents_fitness = {}
    win = pygame.display.set_mode((1, 1))
    pong = PongGame(win, WIDTH, HEIGHT)
    parameters_by_agent = [(2, 10, WIND_LOC_GREEDY), (7, 10, WIND_LOC_GREEDY), (6, 10, WIND_LOC_GREEDY),
                           (4, 15, WIND_LOC_GREEDY), (5, 10, WIND_LOC_GREEDY), (2, 3, WIND_LOC_GREEDY)]
    for generation in range(2, 4, 2):
        for params in parameters_by_agent:
            vt, reset_on, method = params
            agents_fitness[params] = pong.Q_learning_algorithm(
                render=False, Action_method=method, epochs=generation, visits_threshold=vt, reset_on=reset_on)
        # pop the two worst agents
        parameters_by_agent = sorted(parameters_by_agent,
                                     reverse=True, key=lambda p: agents_fitness[p])[:-2]

    print(agents_fitness)


def main():
    win = pygame.display.set_mode((1, 1))
    pygame.display.set_caption("Single Pad Pong")
    pong = PongGame(win, WIDTH, HEIGHT)
    # for method in [EPS_GREEDY, GREEDY, STATE_LOC_GREEDY, WIND_LOC_GREEDY]:
    #     pong.Q_learning_algorithm(Action_method=method
    pong.Q_learning_algorithm(
        render=False, Action_method=WIND_LOC_GREEDY, epochs=50, visits_threshold=8, reset_on=18)
    # pong.Q_learning_algorithm(exploration_rate=0.25)


# main()
genetic_algorithm()
