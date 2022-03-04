from .paddle import Paddle
from .ball import Ball
import pygame


class GameInformation:
    def __init__(self, score):
        self.score = score


class DoublePadPong:
    pygame.init()

    WHITE = (255, 255, 255)
    CYAN = (0,255,255)
    BLACK = (0, 0, 0)
    SCORE_FONT = pygame.font.SysFont("comicsans", 50)
    HORIZ_PAD_WIDTH = 100
    HORIZ_PAD_HEIGHT = 20

    def __init__(self, window, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

        self.paddle_H1 = Paddle(self.window_width//2 -
                             self.HORIZ_PAD_WIDTH//2, self.window_height-30, self.HORIZ_PAD_WIDTH, self.HORIZ_PAD_HEIGHT, False)
        
        self.paddle_H2 = Paddle(self.window_width//2 -
                             self.HORIZ_PAD_WIDTH//2, 10, self.HORIZ_PAD_WIDTH, self.HORIZ_PAD_HEIGHT, False)
        
        self.ball = Ball(self.window_width // 2, self.window_height // 2)

        self.score = 0
        self.window = window

    def draw_score(self):
        score_text = self.SCORE_FONT.render(f"{self.score}", 1, self.WHITE)
        self.window.blit(score_text, (15, 20))
        

    def handle_collision(self):
        # upper wall
        if self.ball.y - self.ball.RADIUS <= 0:
            self.ball.reset()
            return -1
        # right wall
        elif self.ball.x + self.ball.RADIUS >= self.window_width-4:
            self.ball.x_vel *= -1
            return 0
        # left wall
        elif self.ball.x - self.ball.RADIUS <= 4:
            self.ball.x_vel *= -1
            return 0
        elif self.ball.y > self.window_height:
            self.ball.reset()
            return -1

        # check if the ball hits the paddle H1
        if self.ball.y_vel > 0:
            if self.ball.x >= self.paddle_H1.x and self.ball.x <= self.paddle_H1.x + self.HORIZ_PAD_WIDTH:
                if self.ball.y+self.ball.RADIUS >= self.paddle_H1.y:
                    self.ball.y_vel *= -1
                    middle_x = self.paddle_H1.x + self.HORIZ_PAD_WIDTH/2
                    diff_ball_pad_center = middle_x - self.ball.x
                    reduction_factor = (
                        self.HORIZ_PAD_WIDTH / 2)/self.ball.MAX_VEL
                    x_vel = diff_ball_pad_center / reduction_factor
                    if abs(x_vel) < self.ball.MIN_VEL:
                        x_vel = -self.ball.MIN_VEL if x_vel < 0 else self.ball.MIN_VEL
                    self.ball.x_vel = -x_vel
                    return 1

        # check if the ball hits the paddle H2
        if self.ball.y_vel < 0:
            if self.ball.x >= self.paddle_H2.x and self.ball.x <= self.paddle_H2.x + self.HORIZ_PAD_WIDTH:
                if self.ball.y-self.ball.RADIUS-self.HORIZ_PAD_HEIGHT <= self.paddle_H2.y:
                    self.ball.y_vel *= -1
                    middle_x = self.paddle_H2.x + self.HORIZ_PAD_WIDTH/2
                    diff_ball_pad_center = middle_x - self.ball.x
                    reduction_factor = (
                        self.HORIZ_PAD_WIDTH / 2)/self.ball.MAX_VEL
                    x_vel = diff_ball_pad_center / reduction_factor
                    if abs(x_vel) < self.ball.MIN_VEL:
                        x_vel = -self.ball.MIN_VEL if x_vel < 0 else self.ball.MIN_VEL
                    self.ball.x_vel = -x_vel
                    return 1
        return 0
    
    def draw_line(self):
        for i in range(10, self.window_width-10, self.window_width//25):
            if i % 2 == 1:
                continue
            pygame.draw.rect(
                self.window, self.CYAN, (i, self.window_height//2, 10, 2))


    def draw(self):
        self.window.fill(self.BLACK)
        pygame.draw.rect(self.window,self.CYAN,(0,0,5,self.window_height))
        pygame.draw.rect(self.window,self.CYAN,(self.window_width-5,0,5,self.window_height))
        self.draw_line()
        self.draw_score()
        self.paddle_H1.draw(self.window)
        self.paddle_H2.draw(self.window)        
        self.ball.draw(self.window)
        

    def loop(self):
        self.ball.move()
        self.score += self.handle_collision()

        game_info = GameInformation(self.score)

        return game_info

    def reset(self):
        """Resets the entire game."""
        self.ball.reset()
        self.paddle_H1.reset()
        self.score = 0
