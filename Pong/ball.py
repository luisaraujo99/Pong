import pygame
import numpy as np
import random

WHITE = (255, 255, 255)


class Ball:
    MAX_VEL = 3
    COLOR = WHITE
    RADIUS = 8

    def __init__(self, x, y, width_scale, height_scale, window_width, window_height, GAME_DIM):
        self.x = self.original_x = x
        self.y = self.original_y = y

        self.x_vel = self.original_x_vel = width_scale
        self.y_vel = self.original_y_vel = height_scale
        self.window_width = window_width
        self.window_height = window_height
        self.deviation_possibilities = [
            i for i in range(-((GAME_DIM//2)-2), ((GAME_DIM//2)-2))]

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.RADIUS)

    def move(self):
        new_x = self.x+self.x_vel
        new_y = self.y+self.y_vel
        return (new_x, new_y)

    def reset(self):

        random_x_deviation = random.choice(self.deviation_possibilities)

        self.x = self.original_x+random_x_deviation*self.original_x_vel
        self.y = self.original_y

        random_dir = random.choice([1, -1])
        up_or_down = random.choice([1, -1])

        self.x_vel = self.original_x_vel*random_dir
        self.y_vel = self.original_y_vel*up_or_down
