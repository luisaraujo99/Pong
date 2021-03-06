from .paddle import Paddle
from .ball import Ball
import pygame
import operator
import numpy as np

COOPERATION = 1
TEAM_COOPERATION = 2
COMP = 3
PERSONALITY_COOPERATION = 4


class DoublePadPong:
    pygame.init()

    WHITE = (255, 255, 255)
    GREY = (128, 128, 128)
    CYAN = (0, 255, 255)
    GREEN = (124, 252, 0)
    BLACK = (0, 0, 0)
    SCORE_FONT = pygame.font.SysFont("comicsans", 50)

    def __init__(self, window, window_width, window_height, width_scale, height_scale, PAD_SIZE, GAME_DIM_X, GAME_DIM_Y):
        self.window_width = window_width
        self.window_height = window_height
        self.width_scale = width_scale
        self.height_scale = height_scale

        self.paddle1 = Paddle(width_scale*(GAME_DIM_X//2), height_scale,
                              width_scale*PAD_SIZE, height_scale, False, width_scale)

        self.paddle2 = Paddle(width_scale*(GAME_DIM_X//2), self.window_height -
                              2*height_scale, width_scale*PAD_SIZE, height_scale, False, width_scale)

        self.ball = Ball(width_scale*(GAME_DIM_X//2), height_scale*(GAME_DIM_Y//2),
                         self.width_scale, self.height_scale, window_width, window_height, GAME_DIM_X, radius=5)

        self.score = (0, 0)
        self.window = window

    def draw_score(self):
        score_text1 = self.SCORE_FONT.render(f"{self.score[0]}", 1, self.CYAN)
        score_text2 = self.SCORE_FONT.render(f"{self.score[1]}", 1, self.GREEN)
        self.window.blit(score_text1, (15, 60))
        self.window.blit(score_text2, (15, self.window_height-90))

    def handle_ball_paddle_collision(self, ball_x, paddle_number):

        if paddle_number == 1:
            x_paddle_center = self.paddle1.x + self.paddle1.width//2
        elif paddle_number == 2:
            x_paddle_center = self.paddle2.x + self.paddle2.width//2

        ball_distance_paddle_x = (
            ball_x - x_paddle_center)//self.width_scale

        x_vel = self.ball.x_vel
        y_vel = self.ball.y_vel

        if ball_distance_paddle_x < 0:
            if x_vel == 0:
                x_vel = self.ball.original_x_vel
            x_vel = abs(x_vel)*(int(ball_distance_paddle_x))
            y_vel = -y_vel*(int(ball_distance_paddle_x))
        if ball_distance_paddle_x > 0:
            if x_vel == 0:
                x_vel = self.ball.original_x_vel
            x_vel = abs(x_vel)*(int(ball_distance_paddle_x))
            y_vel = y_vel*(int(ball_distance_paddle_x))
        if ball_distance_paddle_x == 0:
            x_vel = 0
        if abs(x_vel) / self.width_scale > Ball.MAX_VEL_X:
            x_vel = - Ball.MAX_VEL_X * \
                self.width_scale if x_vel < 0 else Ball.MAX_VEL_X * self.width_scale
        if abs(y_vel) / self.width_scale > Ball.MAX_VEL_Y:
            y_vel = - Ball.MAX_VEL_Y * \
                self.height_scale if y_vel < 0 else Ball.MAX_VEL_Y * self.height_scale

        self.ball.x_vel = x_vel
        self.ball.y_vel = y_vel

    def reward_type(self, paddle_reward, type=COMP):
        ''' function used to create a reward to manipulate the way the paddles behave '''
        ''' paddle_reward = 1 means Paddle1 caught the ball  '''
        ''' paddle_reward = -1 means Paddle1 did not catch the ball  '''
        if type == COMP:
            if paddle_reward in [-2]:
                return (0, -1, paddle_reward)
            if paddle_reward in [2]:
                return (-1, 0, paddle_reward)
            else:
                return (0, 0, paddle_reward)
        if type == COOPERATION:
            if paddle_reward == 1:
                return (10, 1, paddle_reward)
            if paddle_reward == 2:
                return (1, 10, paddle_reward)
            if paddle_reward == -1:
                return (-10, -1, paddle_reward)
            if paddle_reward == -2:
                return (-1, -10, paddle_reward)
        if type == PERSONALITY_COOPERATION:
            if paddle_reward == 1 or paddle_reward == 2:
                return (1, 1, paddle_reward)
            if paddle_reward == -1 or paddle_reward == -2:
                return (-10, -10, paddle_reward)

    def handle_collision(self):
        (ball_x, ball_y) = self.ball.move()
        reward = (0, 0, 0)

        ####################################################################
        ########################### TOP   PAD ##############################
        ####################################################################

        # TOP LEFT (PAD)
        if ball_x < 0 and ball_y < self.paddle1.y and self.paddle1.x == 0:
            lambda_x = (-self.ball.x)/self.ball.x_vel
            lambda_y = (self.ball.y-(self.paddle1.y +
                        self.paddle1.height))/self.ball.y_vel

            if lambda_x <= lambda_y:
                self.ball.x_vel *= -1
                ball_x = (1-lambda_x)*self.ball.x_vel
                ball_y = (self.ball.y+lambda_x*self.ball.y_vel) + \
                    (1-lambda_x)*self.ball.y_vel
            else:  # PADD COLLISION
                self.ball.y_vel *= -1
                self.handle_ball_paddle_collision(ball_x, 1)
                ball_x = ball_x+(1-lambda_y)*self.ball.x_vel
                ball_y = self.paddle1.y + (1-lambda_y)*self.ball.y_vel
                reward = self.reward_type(1)

        # TOP RIGHT (PAD)
        elif ball_x > self.window_width and ball_y < self.paddle1.y and self.paddle1.x + self.paddle1.width == self.window_width:
            lambda_x = (self.window_width-self.ball.x)/self.ball.x_vel
            lambda_y = (self.ball.y-(self.paddle1.y +
                        self.paddle1.height))/self.ball.y_vel

            if lambda_x <= lambda_y:
                self.ball.x_vel *= -1
                ball_x = self.window_width + (1-lambda_x)*self.ball.x_vel
                ball_y = (self.ball.y+lambda_x*self.ball.y_vel) + \
                    (1-lambda_x)*self.ball.y_vel
            else:  # PADD COLLISION
                self.ball.y_vel *= -1
                self.handle_ball_paddle_collision(ball_x, 1)
                ball_x = ball_x+(1-lambda_y)*self.ball.x_vel
                ball_y = self.paddle1.y + (1-lambda_y)*self.ball.y_vel
                reward = self.reward_type(1)

        ####################################################################
        ########################### BOTTOM  PAD ############################
        ####################################################################

        # BOTTOM LEFT (PAD)
        if ball_x < 0 and ball_y > self.paddle2.y and self.paddle2.x == 0:
            lambda_x = (-self.ball.x)/self.ball.x_vel
            lambda_y = (self.paddle2.y - self.ball.y)/self.ball.y_vel

            if lambda_x <= lambda_y:
                self.ball.x_vel *= -1
                ball_x = (1-lambda_x)*self.ball.x_vel
                ball_y = (self.ball.y+lambda_x*self.ball.y_vel) + \
                    (1-lambda_x)*self.ball.y_vel
            else:  # PADD COLLISION
                self.ball.y_vel *= -1
                self.handle_ball_paddle_collision(ball_x, 2)
                ball_x = ball_x+(1-lambda_y)*self.ball.x_vel
                ball_y = self.paddle2.y + (1-lambda_y)*self.ball.y_vel
                reward = self.reward_type(2)

        # BOTTOM RIGHT (PAD)
        elif ball_x > self.window_width and ball_y > self.paddle2.y and self.paddle2.x + self.paddle2.width == self.window_width:
            lambda_x = (self.window_width-self.ball.x)/self.ball.x_vel
            lambda_y = (self.paddle2.y - self.ball.y)/self.ball.y_vel

            if lambda_x <= lambda_y:
                self.ball.x_vel *= -1
                ball_x = self.window_width + (1-lambda_x)*self.ball.x_vel
                ball_y = (self.ball.y+lambda_x*self.ball.y_vel) + \
                    (1-lambda_x)*self.ball.y_vel
            else:  # PADD COLLISION
                self.ball.y_vel *= -1
                self.handle_ball_paddle_collision(ball_x, 2)
                ball_x = ball_x+(1-lambda_y)*self.ball.x_vel
                ball_y = self.paddle2.y + (1-lambda_y)*self.ball.y_vel
                reward = self.reward_type(2)

        ##################################################################
        ##################################################################
        ##################################################################

        # LEFT
        if ball_x < 0:
            lambda_x = (-self.ball.x)/self.ball.x_vel
            self.ball.x_vel *= -1
            ball_x = (1-lambda_x)*self.ball.x_vel
            ball_y = (self.ball.y+lambda_x*self.ball.y_vel) + \
                (1-lambda_x)*self.ball.y_vel

        # RIGHT
        if ball_x > self.window_width:
            lambda_x = (self.window_width-self.ball.x)/self.ball.x_vel
            self.ball.x_vel *= -1
            ball_x = self.window_width + (1-lambda_x)*self.ball.x_vel
            ball_y = (self.ball.y+lambda_x*self.ball.y_vel) + \
                (1-lambda_x)*self.ball.y_vel

        # BOTTOM PAD LINE
        if ball_y > self.paddle2.y:

            lambda_y = (self.paddle2.y - self.ball.y)/self.ball.y_vel
            ball_x = self.ball.x+lambda_y*self.ball.x_vel
            ball_y = self.paddle2.y
            if ball_x >= self.paddle2.x and ball_x <= self.paddle2.x+self.paddle2.width:
                self.ball.y_vel *= -1
                self.handle_ball_paddle_collision(ball_x, 2)
                ball_x = ball_x+(1-lambda_y)*self.ball.x_vel
                ball_y = self.paddle2.y + (1-lambda_y)*self.ball.y_vel
                reward = self.reward_type(2)
            else:
                self.ball.reset(isDoublePadPong=True)
                return self.reward_type(-2)

        # TOP PAD LINE
        if ball_y < self.paddle1.y+self.paddle1.height:

            lambda_y = (self.ball.y-(self.paddle1.y +
                        self.paddle1.height))/self.ball.y_vel
            ball_x = self.ball.x+lambda_y*self.ball.x_vel
            ball_y = self.paddle1.y+self.paddle1.height
            if ball_x >= self.paddle1.x and ball_x <= self.paddle1.x+self.paddle1.width:
                self.ball.y_vel *= -1
                self.handle_ball_paddle_collision(ball_x, 1)
                ball_x = ball_x+(1-lambda_y)*self.ball.x_vel
                ball_y = ball_y + (1-lambda_y)*self.ball.y_vel
                reward = self.reward_type(1)
            else:
                self.ball.reset(isDoublePadPong=True)
                return self.reward_type(-1)

        ball_x = int(ball_x)
        ball_y = int(ball_y)

        # NEW POSITION IS VALID
        if (ball_x >= 0 and ball_x <= self.window_width) and (ball_y >= 0 and ball_y <= self.window_height):
            self.ball.x = ball_x
            self.ball.y = ball_y
            return reward

        # NEW POSITION IS OUT OF BOUNDs
        else:
            if ball_x < 0:
                self.ball.x = 0
                self.ball.y = ball_y
                return reward
            elif ball_x > self.window_width:
                self.ball.x = self.window_width
                self.ball.y = ball_y
                return reward
            elif ball_y < 0:
                self.ball.x = ball_x
                self.ball.y = 0
                return reward

    def drawGrid(self):
        for x in range(0, self.window_width, self.width_scale):
            pygame.draw.rect(self.window, self.GREY,
                             (x, 0, 1, self.window_height))
        for y in range(0, self.window_height, self.height_scale):
            pygame.draw.rect(self.window, self.GREY,
                             (0, y, self.window_width, 1))

    def drawMidLine(self):
        pygame.draw.rect(self.window, self.CYAN,
                         (0, self.window_height/2, self.window_width, 1))

        pygame.draw.circle(self.window, self.CYAN,
                           (self.window_width/2, self.window_height/2), 8)

    def draw(self):
        self.window.fill(self.BLACK)
        self.draw_score()
        self.paddle1.draw(self.window, self.CYAN)
        self.paddle2.draw(self.window, self.GREEN)
        self.ball.draw(self.window)
        self.drawMidLine()
        # self.drawGrid()

    def loop(self):
        reward = self.handle_collision()
        # competetive
        if reward[2] == -1:
            self.score = tuple(
                map(operator.add, self.score, (0, 1)))
        elif reward[2] == -2:
            self.score = tuple(
                map(operator.add, self.score, (1, 0)))
        return reward

    def reset(self):
        """Resets the entire game."""
        self.ball.reset(isDoublePadPong=True)
        self.paddle.reset()
        self.score = 0
