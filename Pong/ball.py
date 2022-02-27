import pygame
import math
import random

WHITE = (255,255,255)
ANGLES = [math.radians(alpha) for alpha in range(50,130,1) if alpha != 0]

class Ball:
    MAX_VEL = 5
    COLOR = WHITE
    RADIUS = 7
    
    def __init__(self,x,y):
        self.x = self.original_x = x
        self.y = self.original_y = y
        
        random_angle = random.choice(ANGLES)
        
        self.x_vel = math.cos(random_angle) * self.MAX_VEL
        self.y_vel = math.sin(random_angle) * self.MAX_VEL
        
    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x,self.y), self.RADIUS)
        
    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel
        
    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        
        random_angle = random.choice(ANGLES)
        
        self.x_vel = math.cos(random_angle) * self.MAX_VEL
        self.y_vel = math.sin(random_angle) * self.MAX_VEL