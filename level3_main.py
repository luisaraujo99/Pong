import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import numpy as np
from Pong import FourPadPong
from Pong import Q_AI
import pygame
from alive_progress import alive_bar
from itertools import count
import matplotlib
import operator

matplotlib.rc('xtick', labelsize=7)
matplotlib.rc('ytick', labelsize=7)


WIDTH_SCALE, HEIGHT_SCALE = 13, 12
GAME_DIM_X, GAME_DIM_Y = 70, 70
PAD_SIZE = 14
WIDTH, HEIGHT = GAME_DIM_X*WIDTH_SCALE, GAME_DIM_Y*HEIGHT_SCALE
X_PAD_DIM = GAME_DIM_X-(PAD_SIZE-1)


class PongGame:
    def __init__(self, window, width, height):
        self.game = FourPadPong(
            window, width, height, WIDTH_SCALE, HEIGHT_SCALE, PAD_SIZE, GAME_DIM_X, GAME_DIM_Y)
        self.ball = self.game.ball
        self.paddle1 = self.game.paddle1
        self.paddle2 = self.game.paddle2
        self.paddle3 = self.game.paddle3
        self.paddle4 = self.game.paddle4

    def reward(self, initial_score, end_score):
        return tuple(
            map(operator.sub, end_score, initial_score))

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

    def Q_learning_algorithm(self, epochs=5000, episodes=1000, show_v_plot=True, render=True):
        clock = pygame.time.Clock()
        run = True

        ###### Declaring Q_AI Instances ######
        q_ai_1 = Q_AI(learning_rate=1, discount_rate=0.97, X_Pad_dim=GAME_DIM_X-(PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1,
                      learning_decay=1/1200)

        q_ai_2 = Q_AI(learning_rate=1, discount_rate=0.97, X_Pad_dim=GAME_DIM_X-(PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1,
                      learning_decay=1/1200)

        q_ai_3 = Q_AI(learning_rate=1, discount_rate=0.97, X_Pad_dim=GAME_DIM_X-(PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1,
                      learning_decay=1/1200)

        q_ai_4 = Q_AI(learning_rate=1, discount_rate=0.97, X_Pad_dim=GAME_DIM_X-(PAD_SIZE-1), X_Grid_dim=GAME_DIM_X+1, Y_Grid_Dim=GAME_DIM_Y-1,
                      learning_decay=1/1200)

        filename1 = "LVL3_P1_x={x}_y={y}.txt".format(
            x=GAME_DIM_X, y=GAME_DIM_Y)
        filename2 = "LVL3_P2_x={x}_y={y}.txt".format(
            x=GAME_DIM_X, y=GAME_DIM_Y)
        filename3 = "LVL3_P3_x={x}_y={y}.txt".format(
            x=GAME_DIM_X, y=GAME_DIM_Y)
        filename4 = "LVL3_P4_x={x}_y={y}.txt".format(
            x=GAME_DIM_X, y=GAME_DIM_Y)

        q_ai_1.load_file(filename=filename1)
        q_ai_2.load_file(filename=filename2)
        q_ai_3.load_file(filename=filename3)
        q_ai_4.load_file(filename=filename4)

        # plots
        plt.style.use('fivethirtyeight')

        # aux arrays
        v_max_mean1, v_max_mean2, v_max_mean3, v_max_mean4 = [], [], [], []
        v_mid_mean1, v_mid_mean2, v_mid_mean3, v_mid_mean4 = [], [], [], []
        v_min_mean1, v_min_mean2, v_min_mean3, v_min_mean4 = [], [], [], []
        exploration_rates1, exploration_rates2, exploration_rates3, exploration_rates4 = [], [], [], []
        states_visited_ratio1, states_visited_ratio2, states_visited_ratio3, states_visited_ratio4 = [], [], [], []
        learning_rate_evol1, learning_rate_evol2, learning_rate_evol3, learning_rate_evol4 = [], [], [], []
        rewards1, rewards2, rewards3, rewards4 = [], [], [], []
        rewards_queue1, rewards_queue2, rewards_queue3, rewards_queue4 = [], [], [], []

        # variables
        time, epoch = 0, 0
        rewards_in_a_row1, rewards_in_a_row2, rewards_in_a_row3, rewards_in_a_row4 = 0, 0, 0, 0

        while epoch < epochs:

            episode = 0
            game_info = self.game.loop()

            v_max1, v_max2, v_mid1, v_mid2, v_min1, v_min2 = [], [], [], [], [], []
            v_max3, v_max4, v_mid3, v_mid4, v_min3, v_min4 = [], [], [], [], [], []

            with alive_bar(episodes, bar='blocks', title=f'Epoch {epoch}', spinner='arrows') as bar:
                while episode < episodes and run:

                    if render:
                        clock.tick(20)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                run = False
                                break

                    init_score = game_info.score

                    state_p1, state_p2, state_p3, state_p4 = self.return_states()

                    action_p1 = q_ai_1.epsilon_greedy(state_p1)
                    action_p2 = q_ai_2.epsilon_greedy(state_p2)
                    action_p3 = q_ai_3.epsilon_greedy(state_p3)
                    action_p4 = q_ai_4.epsilon_greedy(state_p4)

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

                    game_info = self.game.loop()

                    ############################################################
                    ######################## REWARD HANDLING ###################
                    end_score = self.game.score

                    r1, r2, r3, r4 = self.reward(init_score, end_score)

                    # pad 1
                    if r1 >= 1:
                        rewards_in_a_row1 += 1
                        if rewards_in_a_row1 == 15:
                            self.game.ball.reset()
                            rewards_in_a_row1 = 0

                    if abs(r1) == 1:
                        self.enqueue(rewards_queue1, r1)
                        rewards1.append(np.mean(rewards_queue1))
                    # pad 2
                    if r2 >= 1:
                        rewards_in_a_row2 += 1
                        if rewards_in_a_row2 == 15:
                            self.game.ball.reset()
                            rewards_in_a_row2 = 0

                    if abs(r2) == 1:
                        self.enqueue(rewards_queue2, r2)
                        rewards3.append(np.mean(rewards_queue2))
                    # pad 3
                    if r3 >= 1:
                        rewards_in_a_row3 += 1
                        if rewards_in_a_row3 == 15:
                            self.game.ball.reset()
                            rewards_in_a_row3 = 0

                    if abs(r3) == 1:
                        self.enqueue(rewards_queue3, r3)
                        rewards3.append(np.mean(rewards_queue3))
                    # pad 4
                    if r4 >= 1:
                        rewards_in_a_row4 += 1
                        if rewards_in_a_row4 == 15:
                            self.game.ball.reset()
                            rewards_in_a_row4 = 0

                    if abs(r4) == 1:
                        self.enqueue(rewards_queue4, r4)
                        rewards4.append(np.mean(rewards_queue4))

                    new_state_p1, new_state_p2, new_state_p3, new_state_p4 = self.return_states()

                    q_ai_1.q(action_p1, r1, state_p1, new_state_p1)
                    q_ai_2.q(action_p2, r2, state_p2, new_state_p2)
                    q_ai_3.q(action_p3, r3, state_p3, new_state_p3)
                    q_ai_4.q(action_p4, r4, state_p4, new_state_p4)

                    ############################################################
                    #################### SAVE TRAINING DATA ####################
                    # pad1
                    v_max1.append(q_ai_1.v(state_p1))
                    v_min1.append(q_ai_1.v_min(state_p1))
                    v_mid1.append(q_ai_1.v_mid(state_p1))
                    q_ai_1.q_state_counter(state=state_p1)
                    # pad2
                    v_max2.append(q_ai_2.v(state_p2))
                    v_min2.append(q_ai_2.v_min(state_p2))
                    v_mid2.append(q_ai_2.v_mid(state_p2))
                    q_ai_2.q_state_counter(state=state_p2)
                    # pad3
                    v_max3.append(q_ai_3.v(state_p3))
                    v_min3.append(q_ai_3.v_min(state_p3))
                    v_mid3.append(q_ai_3.v_mid(state_p3))
                    q_ai_3.q_state_counter(state=state_p3)
                    # pad4
                    v_max4.append(q_ai_4.v(state_p4))
                    v_min4.append(q_ai_4.v_min(state_p4))
                    v_mid4.append(q_ai_4.v_mid(state_p4))
                    q_ai_4.q_state_counter(state=state_p4)
                    # ############################################################

                    #######----- EXPLORATION RATE -----#######
                    # q_ai_1.exploration_rate_decay(time, episodes*epochs)
                    # q_ai_2.exploration_rate_decay(time, episodes*epochs)
                    q_ai_1.exploration_rate_decay2(
                        ((q_ai_1.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]) < 4).sum()/(q_ai_1.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]).size)

                    q_ai_2.exploration_rate_decay2(
                        ((q_ai_2.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]) < 4).sum()/(q_ai_1.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]).size)

                    q_ai_3.exploration_rate_decay2(
                        ((q_ai_3.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]) < 4).sum()/(q_ai_3.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]).size)

                    q_ai_4.exploration_rate_decay2(
                        ((q_ai_4.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]) < 4).sum()/(q_ai_4.q_matrix_counter[0:, 1:GAME_DIM_Y-1, 0:]).size)

                    # iteration
                    episode += 1
                    time += 1

                    # show training evolution
                    bar.text(
                        f'\n-> Exploration rate: {q_ai_1.exploration_rate}')
                    bar()

            #################### SAVE TRAINING DATA ####################
            exploration_rates1.append(q_ai_1.exploration_rate)
            v_max_mean1.append(np.mean(v_max1))
            v_min_mean1.append(np.mean(v_min1))
            v_mid_mean1.append(np.mean(v_mid1))
            states_visited_ratio1.append(
                (q_ai_1.q_matrix_counter < 1).sum()/q_ai_1.q_matrix_counter.size)
            learning_rate_evol1.append(q_ai_1.learning_rate)
            # pad 2 stats
            v_max_mean2.append(np.mean(v_max2))
            v_min_mean2.append(np.mean(v_min2))
            v_mid_mean2.append(np.mean(v_mid2))
            states_visited_ratio2.append(
                (q_ai_2.q_matrix_counter < 1).sum()/q_ai_2.q_matrix_counter.size)
            learning_rate_evol2.append(q_ai_2.learning_rate)
            exploration_rates2.append(q_ai_2.exploration_rate)
            ############################################################

            # learning rate evolution
            q_ai_1.learning_rate_decay(epoch)
            q_ai_2.learning_rate_decay(epoch)
            q_ai_3.learning_rate_decay(epoch)
            q_ai_4.learning_rate_decay(epoch)

            # iteration
            epoch += 1

            # save Q state
            q_ai_1.save_state(filename=filename1)
            q_ai_2.save_state(filename=filename2)
            q_ai_3.save_state(filename=filename3)
            q_ai_4.save_state(filename=filename4)

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
    pygame.display.set_caption("Four Pad Pong")
    pong = PongGame(win, WIDTH, HEIGHT)
    plt.show()
    pong.Q_learning_algorithm()


main()
