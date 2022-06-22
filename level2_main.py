import matplotlib.pyplot as plt
import numpy as np
from Pong import DoublePadPong
from Pong import Q_AI
from Pong.PlotTool import *
import pygame
from alive_progress import alive_bar
import operator
import os

from Pong.doublePadPong import DoublePadPong
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)


WIDTH_SCALE, HEIGHT_SCALE = 16, 14
GAME_DIM_X, GAME_DIM_Y = 50, 60
PAD_SIZE = 8
WIDTH, HEIGHT = GAME_DIM_X*WIDTH_SCALE, GAME_DIM_Y*HEIGHT_SCALE
X_PAD_DIM = GAME_DIM_X-(PAD_SIZE-1)
EPS_GREEDY, GREEDY, STATE_LOC_GREEDY, WIND_LOC_GREEDY = 1, 2, 3, 4


class PongGame:
    def __init__(self, window, width, height):
        self.game = DoublePadPong(
            window, width, height, WIDTH_SCALE, HEIGHT_SCALE, PAD_SIZE, GAME_DIM_X, GAME_DIM_Y)
        self.ball = self.game.ball
        self.paddle1 = self.game.paddle1
        self.paddle2 = self.game.paddle2

    def reward(self, initial_score, end_score):
        return tuple(
            map(operator.sub, end_score, initial_score))

    def enqueue(self, rewards_queue, r):
        """ Function to help creating a sliding window to save game status"""
        if len(rewards_queue) == 50:
            rewards_queue.pop(0)
        rewards_queue.append(0 if r == -1 else 1)

    def Q_learning_algorithm(self, epochs=200, episodes=5000, show_v_plot=True, render=True,
                             negative_propagation=True, Action_method=(EPS_GREEDY, EPS_GREEDY), discount_rate=0.97,
                             lr_omega=1/2, exploration_rate=1, visits_threshold=20, reset_on=10):

        clock = pygame.time.Clock()
        run = True

        ###### Declaring Q_AI Instances ######
        q_ai_1 = Q_AI(learning_rate=1, discount_rate=0.97, X_Pad_dim=GAME_DIM_X -
                      (PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1)

        q_ai_2 = Q_AI(learning_rate=1, discount_rate=0.97, X_Pad_dim=GAME_DIM_X -
                      (PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1)

        # if possible, load previous AI state
        filename1 = "./level2_results/LVL2_P1_x={x}_y={y}_Method={m}_epochs={e}_er={er}.txt".format(
            x=GAME_DIM_X, y=GAME_DIM_Y, m=Action_method, e=epochs, er=exploration_rate)
        filename2 = "./level2_results/LVL2_P2_x={x}_y={y}_Method={m}_epochs={e}_er={er}.txt".format(
            x=GAME_DIM_X, y=GAME_DIM_Y, m=Action_method, e=epochs, er=exploration_rate)

        # check if the path exists
        path_name = "./level1_results/X={x}Y={y}".format(
            x=GAME_DIM_X, y=GAME_DIM_Y)
        if not os.path.isdir(path_name):
            os.mkdir(path_name)
        # if possible, load previous AI state
        filename1 = "./level1_results/X={x}Y={y}/Method={m}_epochs={e}_vt={vt}_reseton={reset_on}_lromega={lromega}_dr={dr}_negprop={np}.txt".format(
            x=GAME_DIM_X, y=GAME_DIM_Y, m=Action_method, e=epochs, vt=visits_threshold, reset_on=reset_on, lromega=lr_omega, dr=discount_rate, np=negative_propagation)

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
        rewards1, rewards2 = [], []
        rewards_queue1, rewards_queue2 = [], []

        # variables
        time, epoch, rewards_in_a_row1, rewards_in_a_row2 = 0, 0, 0, 0
        with alive_bar(epochs, bar='blocks', title=f'Trainig evolution', spinner='arrows') as bar:
            while epoch < epochs:

                episode = 0
                game_info = self.game.loop()

                v_max1, v_max2, v_mid1, v_mid2, v_min1, v_min2 = np.zeros(
                    episodes), np.zeros(episodes), np.zeros(episodes), np.zeros(episodes), np.zeros(episodes), np.zeros(episodes)

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
                    action_p1 = q_ai_1.action_chooser_method(
                        state_p1, Action_method[0])
                    action_p2 = q_ai_2.action_chooser_method(
                        state_p2, Action_method[1])
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
                    r1, r2 = self.reward(init_score, end_score)
                    new_state_p1 = ((self.paddle1.x//WIDTH_SCALE), (self.ball.y //
                                                                    HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))
                    new_state_p2 = ((self.paddle1.x//WIDTH_SCALE), (self.ball.y //
                                                                    HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))
                    q_ai_1.q(action_p1, r1, state_p1, new_state_p1)
                    q_ai_2.q(action_p2, r2, state_p2, new_state_p2)
                    if r1 == 1:
                        rewards_in_a_row1 += 1
                        if rewards_in_a_row1 == 15:
                            self.game.ball.reset()
                            rewards_in_a_row1 = 0
                    if abs(r1) == 1:
                        self.enqueue(rewards_queue1, r1)
                        rewards1.append(np.mean(rewards_queue1))
                    if r2 == 1:
                        rewards_in_a_row2 += 1
                        if rewards_in_a_row2 == 15:
                            self.game.ball.reset()
                            rewards_in_a_row2 = 0
                    if abs(r2) == 1:
                        self.enqueue(rewards_queue2, r2)
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
                        ((q_ai_1.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]) < visits_threshold).sum()/(q_ai_1.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]).size)
                    q_ai_2.exploration_rate_decay2(
                        ((q_ai_2.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]) < visits_threshold).sum()/(q_ai_1.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]).size)
                    # iteration
                    episode += 1
                    time += 1

                #################### SAVE TRAINING DATA ####################
                exploration_rates1[epoch] = q_ai_1.exploration_rate
                v_max_mean1[epoch] = np.mean(v_max1)
                v_min_mean1[epoch] = np.mean(v_min1)
                v_mid_mean1[epoch] = np.mean(v_mid1)
                states_visited_ratio1[epoch] = (
                    q_ai_1.q_matrix_counter < 1).sum()/q_ai_1.q_matrix_counter.size
                # pad 2 stats
                exploration_rates2[epoch] = q_ai_2.exploration_rate
                v_max_mean2[epoch] = np.mean(v_max2)
                v_min_mean2[epoch] = np.mean(v_min2)
                v_mid_mean2[epoch] = np.mean(v_mid2)
                states_visited_ratio2[epoch] = (
                    q_ai_2.q_matrix_counter < 1).sum()/q_ai_2.q_matrix_counter.size
                ############################################################

                # iteration
                epoch += 1

                # save Q state
                q_ai_1.save_state(filename=filename1)
                q_ai_2.save_state(filename=filename2)

                # show training evolution
                bar.text(
                    f'\n-> Exploration rate: {q_ai_1.exploration_rate}')
                bar()

        X, Y, Z, _ = q_ai_1.q_matrix.shape
        softmax1 = [q_ai_1.softmax((x, y, z)) for x in range(X)
                    for y in range(Y) for z in range(Z)]
        softmax2 = [q_ai_2.softmax((x, y, z)) for x in range(X)
                    for y in range(Y) for z in range(Z)]

        if show_v_plot:
            plot_v(X*Y*Z, epochs, v_max_mean1,
                   v_min_mean1, v_mid_mean1, softmax1, rewards1, exploration_rates1, states_visited_ratio1, filename1.replace('txt', 'png'))
            plot_v(X*Y*Z, epochs, v_max_mean2,
                   v_min_mean2, v_mid_mean2, softmax2, rewards2, exploration_rates2, states_visited_ratio2, filename2.replace('txt', 'png'))

        # close pygame env
        pygame.quit()


def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Double Pad Pong")
    pong = PongGame(win, WIDTH, HEIGHT)
    plt.show()
    for i in range(5, 21, 5):
        pong.Q_learning_algorithm(visits_threshold=i)


main()
