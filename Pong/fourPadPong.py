from .paddle import Paddle
from .ball import Ball
import pygame


class GameInformation:
    def __init__(self, score):
        self.score = score


class FourPadPong:
    pygame.init()

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    SCORE_FONT = pygame.font.SysFont("comicsans", 50)
    HORIZ_PAD_WIDTH = 100
    HORIZ_PAD_HEIGHT = 20
    VERT_PAD_WIDTH = 20
    VERT_PAD_HEIGHT = 100

    def __init__(self, window, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

        self.paddle_H1 = Paddle(self.window_width//2 -
                                self.HORIZ_PAD_WIDTH//2, self.window_height-30, self.HORIZ_PAD_WIDTH, self.HORIZ_PAD_HEIGHT, False)

        self.paddle_H2 = Paddle(self.window_width//2 -
                                self.HORIZ_PAD_WIDTH//2, 10, self.HORIZ_PAD_WIDTH, self.HORIZ_PAD_HEIGHT, False)

        self.paddle_V1 = Paddle(10, self.window_height//2 -
                                self.HORIZ_PAD_HEIGHT, self.HORIZ_PAD_HEIGHT, self.HORIZ_PAD_WIDTH, True)

        self.paddle_V2 = Paddle(self.window_width - self.VERT_PAD_WIDTH - 10, self.window_height//2 -
                                self.HORIZ_PAD_HEIGHT, self.HORIZ_PAD_HEIGHT, self.HORIZ_PAD_WIDTH, True)

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
        elif self.ball.x + self.ball.RADIUS >= self.window_width:
            self.ball.reset()
            return -1
        # left wall
        elif self.ball.x - self.ball.RADIUS <= 0:
            self.ball.reset()
            return -1
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

        # check if the ball hits the paddle V2
        if self.ball.x_vel > 0:
            if self.ball.y >= self.paddle_V2.y and self.ball.y <= self.paddle_V2.y + self.VERT_PAD_HEIGHT:
                if self.ball.x+self.ball.RADIUS + self.VERT_PAD_WIDTH >= self.paddle_V2.x:
                    self.ball.x_vel *= -1
                    middle_y = self.paddle_V2.y + self.VERT_PAD_HEIGHT/2
                    diff_ball_pad_center = middle_y - self.ball.y
                    reduction_factor = (
                        self.VERT_PAD_HEIGHT / 2)/self.ball.MAX_VEL
                    y_vel = diff_ball_pad_center / reduction_factor
                    if abs(y_vel) < self.ball.MIN_VEL:
                        y_vel = -self.ball.MIN_VEL if y_vel < 0 else self.ball.MIN_VEL
                    self.ball.y_vel = -y_vel
                    return 1

        # check if the ball hits the paddle V1
        if self.ball.x_vel < 0:
            if self.ball.y >= self.paddle_V1.y and self.ball.y <= self.paddle_V1.y + self.VERT_PAD_HEIGHT:
                if self.ball.x-self.ball.RADIUS-self.VERT_PAD_WIDTH <= self.paddle_V1.x:
                    self.ball.x_vel *= -1
                    middle_y = self.paddle_V1.y + self.VERT_PAD_HEIGHT/2
                    diff_ball_pad_center = middle_y - self.ball.y
                    reduction_factor = (
                        self.VERT_PAD_HEIGHT / 2)/self.ball.MAX_VEL
                    y_vel = diff_ball_pad_center / reduction_factor
                    if abs(y_vel) < self.ball.MIN_VEL:
                        y_vel = -self.ball.MIN_VEL if y_vel < 0 else self.ball.MIN_VEL
                    self.ball.y_vel = -y_vel
                    return 1

        return 0

    def draw(self):
        self.window.fill(self.BLACK)
        self.draw_score()
        self.paddle_H1.draw(self.window)
        self.paddle_H2.draw(self.window)
        self.paddle_V1.draw(self.window)
        self.paddle_V2.draw(self.window)
        self.ball.draw(self.window)

    def loop(self):
        self.ball.move()
        self.score += self.handle_collision()

        game_info = GameInformation(self.score)

        return game_info

    def reset(self):
        self.ball.reset()
        self.paddle_H1.reset()
        self.score = 0
