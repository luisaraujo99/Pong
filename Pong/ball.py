import pygame
import numpy as np
import random

WHITE = (255, 255, 255)


class Ball:
    MAX_VEL = 3
    COLOR = WHITE
    RADIUS = 8

    def __init__(self, x, y, width_scale, height_scale, window_width, window_height):
        self.x = self.original_x = x
        self.y = self.original_y = y

        self.x_vel = self.original_x_vel = width_scale
        self.y_vel = self.original_y_vel = height_scale
        self.window_width = window_width
        self.window_height = window_height

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.RADIUS)

    def move(self):

        new_x = self.x+self.x_vel
        new_y = self.y+self.y_vel

        LambdaUP = (-self.y)/self.y_vel
        LambdaLEFT = (-self.x)/self.x_vel
        LambdaRIGHT = (-self.x+self.window_width)/self.x_vel

        lambdas = [LambdaUP, LambdaLEFT, LambdaRIGHT]
        ind = np.argmin([np.linalg.norm(l*np.array([self.x_vel, self.y_vel]))
                         for l in lambdas])

        if((LambdaUP == LambdaRIGHT and (new_y < 0 and new_x > self.window_width)) or (LambdaUP == LambdaLEFT and (new_y < 0 and new_x < 0))):
            self.y_vel *= -1
            self.x_vel *= -1
            self.x += self.x_vel
            self.y += self.y_vel

        # upper wall
        if new_y < 0 and (ind == 0):
            Lambda = (-self.y)/self.y_vel
            x_sim = self.x+Lambda*self.x_vel
            self.y_vel *= -1
            self.x = int(x_sim+Lambda*self.x_vel)
        # left wall
        if new_x < 0 and (ind == 1):
            Lambda = -self.x/self.x_vel
            y_sim = self.y+Lambda*self.y_vel
            self.x_vel *= -1
            self.y = int(y_sim+Lambda*self.y_vel)
        # right wall
        if new_x > self.window_width and (ind == 2):
            Lambda = (-self.x+self.window_width)/self.x_vel
            y_sim = self.y+Lambda*self.y_vel
            self.x_vel *= -1
            self.y = int(y_sim+Lambda*self.y_vel)

        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):

        random_x_deviation = random.choice(
            [5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

        self.x = self.original_x+random_x_deviation*self.original_x_vel
        self.y = self.original_y

        random_dir = random.choice([1, -1])
        up_or_down = random.choice([1, -1])

        self.x_vel = self.original_x_vel*random_dir
        self.y_vel = self.original_y_vel*up_or_down
