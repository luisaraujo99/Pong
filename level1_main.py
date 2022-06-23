from pickle import FALSE, TRUE
from tkinter import W
import matplotlib.pyplot as plt
import numpy as np
from Pong import SinglePadPong
from Pong import Q_AI
from Pong.PlotTool import *
import pygame
from alive_progress import alive_bar
from itertools import count
import os

WIDTH_SCALE, HEIGHT_SCALE = 10, 10
GAME_DIM_X, GAME_DIM_Y = 40, 40
PAD_SIZE = 8
WIDTH, HEIGHT = GAME_DIM_X*WIDTH_SCALE, GAME_DIM_Y*HEIGHT_SCALE
GREEDY, EPS_GREEDY, STATE_LOC_GREEDY, WIND_LOC_GREEDY = 1, 2, 3, 4


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

    def method2str(self, method):
        if method == GREEDY:
            return "GREEDY/"
        elif method == EPS_GREEDY:
            return 'EPS_GREEDY/'
        elif method == STATE_LOC_GREEDY:
            return 'STATE_LOC_GREEDY/'
        elif method == WIND_LOC_GREEDY:
            return 'WIND_LOC_GREEDY/'

    def enqueue(self, rewards_queue, r):
        if len(rewards_queue) == 500:
            rewards_queue.pop(0)
        rewards_queue.append(0 if r == -1 else 1)

    def Q_learning_algorithm(self, epochs=200, episodes=5000, show_v_plot=True, render=True,
                             negative_propagation=False, Action_method=EPS_GREEDY, discount_rate=0.97, lr=1,
                             exploration_rate=1, visits_threshold=20, reset_on=10):
        clock = pygame.time.Clock()
        run = True

        # create AI agent
        q_ai = Q_AI(learning_rate=lr, discount_rate=discount_rate, exploration_rate=exploration_rate,
                    X_Pad_dim=GAME_DIM_X-(PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1)

        # check if the path exists
        if not os.path.isdir('./level1_results'):
            os.mkdir('./level1_results')
        path_name = "./level1_results/X={x}Y={y}".format(
            x=GAME_DIM_X, y=GAME_DIM_Y)
        if not os.path.isdir(path_name):
            os.mkdir(path_name)
        # method path
        method_path_name = "./level1_results/X={x}Y={y}/{method}".format(
            x=GAME_DIM_X, y=GAME_DIM_Y, method=self.method2str(Action_method))
        if not os.path.isdir(method_path_name):
            os.mkdir(method_path_name)

        # if possible, load previous AI state
        filename = "./level1_results/X={x}Y={y}/{method}epochs={e}_vt={vt}_reseton={reset_on}_lr={lr}_dr={dr}_negprop={np}.txt".format(
            x=GAME_DIM_X, y=GAME_DIM_Y, m=Action_method, e=epochs,
            vt=visits_threshold, reset_on=reset_on, lr=lr,
            dr=discount_rate, np=negative_propagation, method=self.method2str(Action_method))
        q_ai.load_file(filename=filename)

        # Aux vars and arrays

        v_max_mean = np.zeros(epochs)
        v_mid_mean = np.zeros(epochs)
        v_min_mean = np.zeros(epochs)
        exploration_rates = np.zeros(epochs)
        states_visited_ratio = np.zeros(epochs)
        fitness_scores = np.zeros(epochs)
        rewards_in_a_row = 0
        time, epoch = 0, 0
        rewards = []
        rewards_seq = []
        rewards_queue = []

        if not render:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        with alive_bar(epochs, bar='blocks', title=f'Trainig evolution', spinner='arrows') as bar:
            while epoch < epochs:

                episode = 0
                game_info = self.game.loop()

                v_max = np.zeros(episodes)
                v_mid = np.zeros(episodes)
                v_min = np.zeros(episodes)

                while episode < episodes and run:
                    if render:
                        clock.tick(15)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                run = False
                                break

                    init_score = game_info.score
                    state = ((self.paddle.x//WIDTH_SCALE), (self.ball.y //
                                                            HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

                    action = q_ai.action_chooser_method(
                        state, Action_method, visits_threshold)
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

                    q_ai.q(action, r, state, new_state,
                           negative_propagation=negative_propagation)
                    if abs(r) == 1:
                        self.enqueue(rewards_queue, r)
                        rewards.append(np.mean(rewards_queue))
                        rewards_seq.append(r)
                    if r == 1:
                        rewards_in_a_row += 1
                        if rewards_in_a_row == reset_on:
                            self.game.ball.reset()
                            rewards_in_a_row = 0
                    elif r == -1:
                        rewards_in_a_row = 0
                    #################### SAVE TRAINING DATA ####################
                    v_max[episode] = q_ai.v(state)
                    v_min[episode] = q_ai.v_min(state)
                    v_mid[episode] = q_ai.v_mid(state)
                    q_ai.q_state_counter(state=state)
                    ############################################################
                    #######----- EXPLORATION RATE -----#######
                    #q_ai.exploration_rate_decay(time, episodes*epochs)
                    q_ai.set_exploration_rate_decay(
                        (q_ai.q_matrix_counter < visits_threshold).sum()/q_ai.q_matrix_counter.size)

                    # iteration
                    episode += 1
                    time += 1

                #################### SAVE TRAINING DATA ####################
                exploration_rates[epoch] = q_ai.exploration_rate
                v_max_mean[epoch] = np.mean(v_max)
                v_min_mean[epoch] = np.mean(v_min)
                v_mid_mean[epoch] = np.mean(v_mid)
                states_visited_ratio[epoch] = (
                    q_ai.q_matrix_counter < 1).sum()/q_ai.q_matrix_counter.size
                # fitness_scores[epoch] = q_ai.fitness_score(
                #    rewards_seq, q_ai.full_matrix_softmax())
                fitness_scores[epoch] = np.max(q_ai.q_matrix)
                ############################################################

                # iteration
                epoch += 1

                # save Q state
                q_ai.save_state(filename=filename)

                # show training evolution
                bar.text(
                    f'\n-> Exploration rate: {q_ai.exploration_rate}')
                bar()

        softmax = q_ai.full_matrix_softmax()

        if show_v_plot:
            plot_v(np.prod(q_ai.q_matrix.shape[:-1]), epochs, v_max_mean,
                   v_min_mean, v_mid_mean, softmax, rewards, exploration_rates, states_visited_ratio, fitness_scores, filename.replace('txt', 'png'))
            if GAME_DIM_Y == 10:
                plot_color_action_matrix(
                    q_ai.q_matrix, filename.replace('txt', 'png'))
                plot_matrix_state_counter(
                    q_ai.q_matrix_counter, filename.replace('txt', 'png'))
                plot_max_val_gradient(
                    q_ai.q_matrix, filename.replace('txt', 'png'))

        # close pygame env
        print(np.max(q_ai.q_matrix))
        pygame.quit()


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
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Single Pad Pong")
    pong = PongGame(win, WIDTH, HEIGHT)

    for m in [4]:
        for reseton in [15]:
            for visits in [2]:
                for lr in [0.9]:
                    for neg in [False]:
                        pong.Q_learning_algorithm(
                            epochs=300, episodes=20000, discount_rate=0.97, lr=lr,
                            negative_propagation=neg, visits_threshold=visits,
                            reset_on=reseton, render=True, Action_method=m, exploration_rate=1)


main()
# genetic_algorithm()
