
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import count
from matplotlib import colors
import numpy as np

plt.style.use('fivethirtyeight')


def plot_v(matrix_range, epochs, vmax, vmin, vmid, softmax, rewards, exploration_rates, states_visited_ratio, fitness_scores, filename):
    xvals = [i for i in range(epochs)]
    x2vals = [i for i in range(matrix_range)]
    xvals3 = [i for i in range(len(rewards))]
    fig, axs = plt.subplots(5)
    # v function plot
    axs[0].set_title("Matrix values Mean per epoch",
                     fontdict={'fontsize': 9})
    axs[0].plot(xvals, vmax, color='r', linewidth=1)
    axs[0].plot(xvals, vmin, color='g', linewidth=1)
    axs[0].plot(xvals, vmid, color='b', linewidth=1)
    axs[0].legend(['vmax', "vmin", "vmid"], fontsize=7)
    # final full matrix softmax
    axs[1].set_title(
        "Softmax of Matrix values after training", fontdict={'fontsize': 8})
    axs[1].plot(x2vals, [Max for (Max, Min) in softmax],
                color='r', linewidth=1)
    axs[1].plot(x2vals, [Min for (Max, Min) in softmax],
                color='b', linewidth=1)
    axs[1].legend(["max", "min"], fontsize=7)
    # rewards, explor. rate, l rate
    axs[2].set_title(
        "Rewards ratio per epoch and exploration rate", fontdict={'fontsize': 8})
    axs[2].plot(xvals, exploration_rates, linewidth=1)
    axs[2].plot(xvals, states_visited_ratio, linewidth=1)
    axs[2].legend(["explor. rate",
                  "stat. visited"], fontsize=7)
    axs[3].set_title(
        "Rewards Moving average", fontdict={'fontsize': 8})
    axs[3].plot(xvals3, rewards, linewidth=1)
    # fitness
    axs[4].set_title(
        "Maximum Q Values recorded", fontdict={'fontsize': 8})
    axs[4].plot(xvals, fitness_scores, linewidth=1)
    plt.subplots_adjust(hspace=0.9)
    plt.savefig(filename, dpi=900)


def plot_color_action_matrix(q_matrix, filename):
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
    plt.savefig(filename[:-4]+"_action.png", dpi=900)


def plot_matrix_state_counter(matrix_counter, filename):
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
    plt.savefig(filename[:-4]+"_states.png", dpi=900)


def plot_max_val_gradient(matrix, filename):
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
    plt.savefig(filename[:-4]+"_max.png", dpi=900)
