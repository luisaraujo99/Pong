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

    def __init__(self, x, y, width_scale, height_scale,window_width,window_height):
        self.x = self.original_x = x
        self.y = self.original_y = y

        random_dir = random.choice([1, -1])
        self.x_vel = self.original_x_vel = width_scale*random_dir
        self.y_vel = self.original_y_vel = height_scale*2
        self.window_width = window_width
        self.window_height = window_height

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.RADIUS)

    def move(self):
        
        new_x = self.x+self.x_vel 
        new_y = self.y+self.y_vel
        
        
        
        #upper wall
        if new_y <= 0:
            Lambda = (-self.y)/self.y_vel
            x_sim=self.x+Lambda*self.x_vel
            self.y_vel *=-1
            self.x = int(x_sim+Lambda*self.x_vel)
        # left wall 
        if new_x <= 0 :
            Lambda = -self.x/self.x_vel
            y_sim=self.y+Lambda*self.y_vel
            self.x_vel *=-1
            self.y = int(y_sim+Lambda*self.y_vel)
        #right wall
        if new_x >= self.window_width:
            Lambda = (-self.x+self.window_width)/self.x_vel
            y_sim=self.y+Lambda*self.y_vel
            self.x_vel *=-1
            self.y = int(y_sim+Lambda*self.y_vel)
        
        if new_x>0 and new_y>0 and new_x < self.window_width:
            self.x += self.x_vel 
            self.y += self.y_vel
            
            

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y

        random_dir = random.choice([1, -1])

        self.x_vel = self.original_x_vel*random_dir
        self.y_vel = self.original_y_vel
