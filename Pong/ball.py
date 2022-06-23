import pygame
import numpy as np

WHITE = (255, 255, 255)


class Ball:
    MAX_VEL = 3
    COLOR = WHITE

    def __init__(self, x, y, width_scale, height_scale, window_width, window_height, GAME_DIM, radius=8):
        self.x = self.original_x = x
        self.y = self.original_y = y

        self.x_vel = self.original_x_vel = width_scale
        self.y_vel = self.original_y_vel = height_scale
        self.window_width = window_width
        self.window_height = window_height
        self.deviation_possibilities = [
            i for i in range(-((GAME_DIM//4)), ((GAME_DIM//4)+1))]
        self.radius = radius
        self.rng_fixed_seed = np.random.default_rng(2022)

    def draw(self, win):
        'Method responsible for displaying the ball in pygame window.'
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.radius)

    def move(self):
        'Method that updates the ball position according to currend ball speed.'
        new_x = self.x+self.x_vel
        new_y = self.y+self.y_vel
        return (new_x, new_y)

    def reset(self, isFourPadPong=False):
        'Method to reset the ball position. '

        random_dir = self.rng_fixed_seed.choice([1, -1])

        if (not isFourPadPong):
            random_x_deviation = self.rng_fixed_seed.choice(
                self.deviation_possibilities)
            # choose the direction to be vertical with 30% probability
            null_vel = self.rng_fixed_seed.choice([0, 1], p=[0.7, 0.3])
            self.x = self.original_x+random_x_deviation*self.original_x_vel
            self.y = self.original_y
            self.x_vel = (1-null_vel)*self.original_x_vel*random_dir
            self.y_vel = self.original_y_vel

        else:
            up_or_down = self.rng_fixed_seed.choice([1, -1])
            null_vel = self.rng_fixed_seed.choice([0, 1])
            self.x = self.original_x
            self.y = self.original_y
            self.x_vel = null_vel*self.original_x_vel*random_dir
            self.y_vel = (1-null_vel)*self.original_y_vel*up_or_down
