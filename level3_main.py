import matplotlib.pyplot as plt
import numpy as np
from Pong import FourPadPong
from Pong import Q_AI
from Pong.PlotTool import *
import pygame
from alive_progress import alive_bar
import operator
import os
from pygame_recorder import ScreenRecorder
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)
os.environ['SDL_VIDEODRIVER'] = 'dummy'

WIDTH_SCALE, HEIGHT_SCALE = 20,20
GAME_DIM_X, GAME_DIM_Y =30,30
PAD_SIZE = 6
WIDTH, HEIGHT = GAME_DIM_X*WIDTH_SCALE, GAME_DIM_Y*HEIGHT_SCALE
X_PAD_DIM = GAME_DIM_X-(PAD_SIZE-1)
GREEDY, EPS_GREEDY, STATE_LOC_GREEDY, WIND_LOC_GREEDY = 1, 2, 3, 4
FPS = 50



class PongGame:
    def __init__(self, window, width, height):
        self.game = FourPadPong(
            window, width, height, WIDTH_SCALE, HEIGHT_SCALE, PAD_SIZE, GAME_DIM_X, GAME_DIM_Y)
        self.ball = self.game.ball
        self.paddle1 = self.game.paddle1
        self.paddle2 = self.game.paddle2
        self.paddle3 = self.game.paddle3
        self.paddle4 = self.game.paddle4

    def method2str(self, method):
        if method == GREEDY:
            return "GREEDY/"
        elif method == EPS_GREEDY:
            return 'EPS_GREEDY/'
        elif method == STATE_LOC_GREEDY:
            return 'STATE_LOC_GREEDY/'
        elif method == WIND_LOC_GREEDY:
            return 'WIND_LOC_GREEDY/'

    def reward(self, initial_score, end_score):
        return tuple(
            map(operator.sub, end_score, initial_score))

    def create_paths(self, Action_method):
        # check if the path exists
        if not os.path.isdir('./level3_results'):
            os.mkdir('./level3_results')
        path_name = "./level3_results/X={x}Y={y}".format(
            x=GAME_DIM_X, y=GAME_DIM_Y)
        if not os.path.isdir(path_name):
            os.mkdir(path_name)
        # method path
        method_path_name = "./level3_results/X={x}Y={y}/{method}".format(
            x=GAME_DIM_X, y=GAME_DIM_Y, method=self.method2str(Action_method))
        if not os.path.isdir(method_path_name):
            os.mkdir(method_path_name)

    def enqueue(self, rewards_queue, r):
        """ Function to help creating a sliding window to save game status"""
        if len(rewards_queue) == 500:
            rewards_queue.pop(0)
        rewards_queue.append(0 if r == -1 else 1)

    def return_states(self):
        state_p1 = ((self.paddle1.x//WIDTH_SCALE), (self.ball.y //
                                                    HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

        state_p2 = ((self.paddle2.x//WIDTH_SCALE), (self.ball.y //
                                                    HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

        state_p3 = ((self.paddle3.y//WIDTH_SCALE), (self.ball.y //
                                                    HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))

        state_p4 = ((self.paddle4.y//WIDTH_SCALE), (self.ball.y //
                                                    HEIGHT_SCALE), (self.ball.x//WIDTH_SCALE))
        return (state_p1, state_p2, state_p3, state_p4)

    def init_np_arrays(self, length, n):
        ''' return n numpy arrays of a given length'''
        return [np.ones(length)]*n

    def Q_learning_algorithm(self, epochs=200, episodes=5000, show_v_plot=True, render=True,
                             negative_propagation=False, Action_method=(EPS_GREEDY, EPS_GREEDY, EPS_GREEDY, EPS_GREEDY), discount_rate=0.97, lr=1,
                             exploration_rate=1, visits_threshold=20, reset_on=10):

        clock = pygame.time.Clock()
        run = True
        if render:
            recorder = ScreenRecorder(WIDTH, HEIGHT, FPS)

        ###### Declaring Q_AI Instances ######
        q_ai_1 = Q_AI(learning_rate=lr, discount_rate=0.97, X_Pad_dim=GAME_DIM_X -
                      (PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1, seed=np.random.default_rng(12022))

        q_ai_2 = Q_AI(learning_rate=lr, discount_rate=0.97, X_Pad_dim=GAME_DIM_X -
                      (PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1, seed=np.random.default_rng(22022))

        q_ai_3 = Q_AI(learning_rate=lr, discount_rate=0.97, X_Pad_dim=GAME_DIM_X -
                      (PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1, seed=np.random.default_rng(32022))

        q_ai_4 = Q_AI(learning_rate=lr, discount_rate=0.97, X_Pad_dim=GAME_DIM_X -
                      (PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1, seed=np.random.default_rng(42022))

        # create paths to save files
        filenames = []
        for ai in range(4):
            self.create_paths(Action_method[ai])
            filenames.append("./level3_results/X={x}Y={y}/{method}P{player}_epochs={e}_vt={vt}_reseton={reset_on}_lr={lr}_dr={dr}_negprop={np}.txt".format(
                x=GAME_DIM_X, y=GAME_DIM_Y, m=Action_method, e=epochs,
                vt=visits_threshold, reset_on=reset_on, lr=lr,
                dr=discount_rate, np=negative_propagation, method=self.method2str(Action_method[ai]), player=ai))

        q_ai_1.load_file(filename=filenames[0])
        q_ai_2.load_file(filename=filenames[1])
        q_ai_3.load_file(filename=filenames[2])
        q_ai_4.load_file(filename=filenames[3])

        # plots
        plt.style.use('fivethirtyeight')

        # aux arrays
        v_max_mean1, v_max_mean2, v_max_mean3, v_max_mean4 = self.init_np_arrays(
            epochs, 4)
        v_mid_mean1, v_mid_mean2, v_mid_mean3, v_mid_mean4 = self.init_np_arrays(
            epochs, 4)
        v_min_mean1, v_min_mean2, v_min_mean3, v_min_mean4 = self.init_np_arrays(
            epochs, 4)
        exploration_rates1, exploration_rates2, exploration_rates3, exploration_rates4 = self.init_np_arrays(
            epochs, 4)
        states_visited_ratio1, states_visited_ratio2, states_visited_ratio3, states_visited_ratio4 = self.init_np_arrays(
            epochs, 4)
        learning_rate_evol1, learning_rate_evol2, learning_rate_evol3, learning_rate_evol4 = self.init_np_arrays(
            epochs, 4)
        maximum_rec_val1, maximum_rec_val2, maximum_rec_val3, maximum_rec_val4 = self.init_np_arrays(
            epochs, 4)
        rewards1, rewards2, rewards3, rewards4 = [], [], [], []
        rewards_queue1, rewards_queue2, rewards_queue3, rewards_queue4 = [], [], [], []

        # variables
        time, epoch = 0, 0
        rewards_in_a_row = 0
        with alive_bar(epochs, bar='blocks', title=f'Trainig evolution', spinner='arrows') as bar:
            while epoch < epochs:

                episode = 0
                self.game.loop()

                v_max1, v_max2, v_mid1, v_mid2, v_min1, v_min2 = self.init_np_arrays(
                    episodes, 6)
                v_max3, v_max4, v_mid3, v_mid4, v_min3, v_min4 = self.init_np_arrays(
                    episodes, 6)

                while episode < episodes and run:
                    if render:
                        clock.tick(FPS)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                run = False
                                break

                    state_p1, state_p2, state_p3, state_p4 = self.return_states()

                    # let each AI choose an action
                    action_p1 = q_ai_1.action_chooser_method(
                        state_p1, Action_method[0], visits_threshold)
                    action_p2 = q_ai_2.action_chooser_method(
                        state_p2, Action_method[1], visits_threshold)
                    action_p3 = q_ai_3.action_chooser_method(
                        state_p3, Action_method[2], visits_threshold)
                    action_p4 = q_ai_4.action_chooser_method(
                        state_p4, Action_method[3], visits_threshold)

                    ##########################################
                    ################# PAD 1 ##################
                    # right
                    if action_p1 == 2:
                        self.game.paddle1.move(
                            False, True, window_width=WIDTH, window_height=HEIGHT)
                    # left
                    elif action_p1 == 1:
                        self.game.paddle1.move(
                            False, False, window_width=WIDTH, window_height=HEIGHT)
                    ################# PAD 2 ##################
                    # right
                    if action_p2 == 2:
                        self.game.paddle2.move(
                            True, True, window_width=WIDTH, window_height=HEIGHT)
                    # left
                    elif action_p2 == 1:
                        self.game.paddle2.move(
                            True, False, window_width=WIDTH, window_height=HEIGHT)
                    ################# PAD 3 ##################
                    # right
                    if action_p3 == 2:
                        self.game.paddle3.move(
                            True, True, window_width=WIDTH, window_height=HEIGHT)
                    # left
                    elif action_p3 == 1:
                        self.game.paddle3.move(
                            True, False, window_width=WIDTH, window_height=HEIGHT)
                    ################# PAD 4 ##################
                    # right
                    if action_p4 == 2:
                        self.game.paddle4.move(
                            True, True, window_width=WIDTH, window_height=HEIGHT)
                    # left
                    elif action_p4 == 1:
                        self.game.paddle4.move(
                            True, False, window_width=WIDTH, window_height=HEIGHT)
                    ##########################################
                    ##########################################
                    if render:
                        self.game.draw()
                        pygame.display.update()
                        recorder.capture_frame(self.game.window)

                    ############################################################
                    ######################## REWARD HANDLING ###################
                    ############################################################

                    r1, r2, r3, r4, winner = self.game.loop()
                    new_state_p1, new_state_p2, new_state_p3, new_state_p4 = self.return_states()
                    q_ai_1.q(action_p1, r1, state_p1, new_state_p1)
                    q_ai_2.q(action_p2, r2, state_p2, new_state_p2)
                    q_ai_3.q(action_p3, r3, state_p3, new_state_p3)
                    q_ai_4.q(action_p4, r4, state_p4, new_state_p4)
                    # reset on
                    if winner > 0:
                        rewards_in_a_row += 1
                        if rewards_in_a_row == reset_on:
                            self.game.ball.reset()
                            rewards_in_a_row = 0
                    elif winner < 0:
                        rewards_in_a_row = 0
                    # pad1
                    if abs(r1) > 0:
                        self.enqueue(rewards_queue1, r1)
                        rewards1.append(np.mean(rewards_queue1))
                    # pad 2
                    if abs(r2) > 0:
                        self.enqueue(rewards_queue2, r2)
                        rewards2.append(np.mean(rewards_queue2))
                    # pad 3
                    if abs(r3) > 0:
                        self.enqueue(rewards_queue3, r3)
                        rewards3.append(np.mean(rewards_queue3))
                    # pad 4
                    if abs(r4) > 0:
                        self.enqueue(rewards_queue4, r4)
                        rewards4.append(np.mean(rewards_queue4))
                    ############################################################
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
                    # pad3
                    v_max3[episode] = q_ai_3.v(state_p3)
                    v_min3[episode] = q_ai_3.v_min(state_p3)
                    v_mid3[episode] = q_ai_3.v_mid(state_p3)
                    q_ai_3.q_state_counter(state=state_p3)
                    # pad4
                    v_max4[episode] = q_ai_4.v(state_p4)
                    v_min4[episode] = q_ai_4.v_min(state_p4)
                    v_mid4[episode] = q_ai_4.v_mid(state_p4)
                    q_ai_4.q_state_counter(state=state_p4)
                    # ############################################################
                    #######----- EXPLORATION RATE -----#######
                    # q_ai_1.exploration_rate_decay(time, episodes*epochs)
                    # q_ai_2.exploration_rate_decay(time, episodes*epochs)
                    q_ai_1.set_exploration_rate_decay(
                        ((q_ai_1.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]) < visits_threshold).sum()/(q_ai_1.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]).size)
                    q_ai_2.set_exploration_rate_decay(
                        ((q_ai_2.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]) < visits_threshold).sum()/(q_ai_2.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]).size)
                    q_ai_3.set_exploration_rate_decay(
                        ((q_ai_3.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]) < visits_threshold).sum()/(q_ai_3.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]).size)
                    q_ai_4.set_exploration_rate_decay(
                        ((q_ai_4.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]) < visits_threshold).sum()/(q_ai_4.q_matrix_counter[:, 1:GAME_DIM_Y-1, :]).size)
                    # iteration
                    episode += 1
                    time += 1

                #################### SAVE TRAINING DATA ####################
                exploration_rates1[epoch] = q_ai_1.exploration_rate
                v_max_mean1[epoch] = np.mean(v_max1)
                v_min_mean1[epoch] = np.mean(v_min1)
                v_mid_mean1[epoch] = np.mean(v_mid1)
                maximum_rec_val1[epoch] = np.max(q_ai_1.q_matrix)
                states_visited_ratio1[epoch] = (
                    q_ai_1.q_matrix_counter < 1).sum()/q_ai_1.q_matrix_counter.size
                learning_rate_evol1[epoch] = q_ai_1.learning_rate
                # pad 2 stats
                v_max_mean2[epoch] = np.mean(v_max2)
                v_min_mean2[epoch] = np.mean(v_min2)
                v_mid_mean2[epoch] = np.mean(v_mid2)
                maximum_rec_val2[epoch] = np.max(q_ai_2.q_matrix)
                states_visited_ratio2[epoch] = (
                    q_ai_2.q_matrix_counter < 1).sum()/q_ai_2.q_matrix_counter.size
                learning_rate_evol2[epoch] = q_ai_2.learning_rate
                exploration_rates2[epoch] = q_ai_2.exploration_rate
                # pad 3 stats
                v_max_mean3[epoch] = np.mean(v_max3)
                v_min_mean3[epoch] = np.mean(v_min3)
                v_mid_mean3[epoch] = np.mean(v_mid3)
                maximum_rec_val3[epoch] = np.max(q_ai_3.q_matrix)
                states_visited_ratio3[epoch] = (
                    q_ai_3.q_matrix_counter < 1).sum()/q_ai_3.q_matrix_counter.size
                learning_rate_evol3[epoch] = q_ai_3.learning_rate
                exploration_rates3[epoch] = q_ai_3.exploration_rate
                # pad 4 stats
                v_max_mean4[epoch] = np.mean(v_max4)
                v_min_mean4[epoch] = np.mean(v_min4)
                v_mid_mean4[epoch] = np.mean(v_mid4)
                maximum_rec_val4[epoch] = np.max(q_ai_4.q_matrix)
                states_visited_ratio4[epoch] = (
                    q_ai_4.q_matrix_counter < 1).sum()/q_ai_4.q_matrix_counter.size
                learning_rate_evol4[epoch] = q_ai_4.learning_rate
                exploration_rates4[epoch] = q_ai_4.exploration_rate
                ############################################################

                # iteration
                epoch += 1
                if epoch > 15:
                    q_ai_1.learning_rate_decay(epoch,0.1)
                    q_ai_2.learning_rate_decay(epoch,0.1)
                    q_ai_3.learning_rate_decay(epoch,0.1)
                    q_ai_4.learning_rate_decay(epoch,0.1)

                # save Q state
                q_ai_1.save_state(filename=filenames[0])
                q_ai_2.save_state(filename=filenames[1])
                q_ai_3.save_state(filename=filenames[2])
                q_ai_4.save_state(filename=filenames[3])

                # show training evolution
                bar.text(
                    f'\n-> ERs: {round(q_ai_1.exploration_rate,2)},{round(q_ai_2.exploration_rate,2)},{round(q_ai_3.exploration_rate,2)},{round(q_ai_4.exploration_rate,2)}')
                bar()

        X, Y, Z, _ = q_ai_1.q_matrix.shape
        softmax1 = [q_ai_1.softmax((x, y, z)) for x in range(X)
                    for y in range(Y) for z in range(Z)]
        softmax2 = [q_ai_2.softmax((x, y, z)) for x in range(X)
                    for y in range(Y) for z in range(Z)]
        softmax3 = [q_ai_3.softmax((x, y, z)) for x in range(X)
                    for y in range(Y) for z in range(Z)]
        softmax4 = [q_ai_4.softmax((x, y, z)) for x in range(X)
                    for y in range(Y) for z in range(Z)]

        if show_v_plot:
            plot_v(X*Y*Z, epochs, v_max_mean1,
                   v_min_mean1, v_mid_mean1, softmax1, rewards1, exploration_rates1, states_visited_ratio1, maximum_rec_val1, filenames[0].replace('txt', 'png'))
            plot_v(X*Y*Z, epochs, v_max_mean2,
                   v_min_mean2, v_mid_mean2, softmax2, rewards2, exploration_rates2, states_visited_ratio2, maximum_rec_val2, filenames[1].replace('txt', 'png'))
            plot_v(X*Y*Z, epochs, v_max_mean3,
                   v_min_mean3, v_mid_mean3, softmax3, rewards3, exploration_rates3, states_visited_ratio3, maximum_rec_val3, filenames[2].replace('txt', 'png'))
            plot_v(X*Y*Z, epochs, v_max_mean4,
                   v_min_mean4, v_mid_mean4, softmax4, rewards4, exploration_rates4, states_visited_ratio4, maximum_rec_val4, filenames[3].replace('txt', 'png'))

        # close pygame env
        if render:
            recorder.end_recording()
        pygame.quit()


def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Four Pad Pong")
    pong = PongGame(win, WIDTH, HEIGHT)

    for m in [(4, 4, 4, 4)]:
        for reseton in [8]:
            for visits in [18]:
                for lr in [1]:
                    for neg in [False]:
                        pong.Q_learning_algorithm(
                            epochs=80, episodes=50000, discount_rate=0.97, lr=lr,
                            negative_propagation=neg, visits_threshold=visits,
                            reset_on=reseton, render=False, Action_method=m, exploration_rate=1)


main()
