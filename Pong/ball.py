import pygame
import math
import random

WHITE = (255, 255, 255)


class Ball:
    MIN_VEL = 30
    MAX_VEL = 4
    INIT_VEL = 40
    COLOR = WHITE
    RADIUS = 8

    def __init__(self, x, y, width_scale, height_scale):
        self.x = self.original_x = x
        self.y = self.original_y = y

        random_dir = random.choice([1, -1])
        self.x_vel = self.original_x_vel = width_scale*random_dir
        self.y_vel = self.original_y_vel = height_scale*2

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.RADIUS)

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y

        random_dir = random.choice([1, -1])

        self.x_vel = self.original_x_vel*random_dir
        self.y_vel = self.original_y_vel
