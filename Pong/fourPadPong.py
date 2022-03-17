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
    GREY = (128, 128, 128)
    SCORE_FONT = pygame.font.SysFont("comicsans", 50)

    def __init__(self, window, window_width, window_height, width_scale, height_scale):
        self.window_width = window_width
        self.window_height = window_height
        self.width_scale = width_scale
        self.height_scale = height_scale

        self.paddle_H1 = Paddle(width_scale*11, self.window_height -
                                2*height_scale, width_scale*6, height_scale, False, width_scale)

        self.paddle_H2 = Paddle(
            width_scale*11, height_scale, width_scale*6, height_scale, False, width_scale)

        self.paddle_V1 = Paddle(
            width_scale, height_scale*20, width_scale, height_scale*6, True, height_scale)

        self.paddle_V2 = Paddle(
            window_width - 2*width_scale, height_scale*20, width_scale, height_scale*6, True, height_scale)

        self.ball = Ball(width_scale*20, height_scale*20,
                         self.width_scale, self.height_scale)

        self.score = 0
        self.window = window

    def draw_score(self):
        score_text = self.SCORE_FONT.render(f"{self.score}", 1, self.GREY)
        self.window.blit(
            score_text, (self.window_width//2, self.window_height//2))

    def handle_collision(self):
        # upper wall
        if self.ball.y <= 0:
            self.ball.reset()
            return -1
        # right wall
        if self.ball.x >= self.window_width:
            self.ball.reset()
            return -1
        # left wall
        if self.ball.x <= 0:
            self.ball.reset()
            return -1
        if self.ball.y > self.window_height:
            self.ball.reset()
            return -1

        # check if the ball hits the paddle H1
        if self.ball.y_vel > 0:
            if self.ball.x >= self.paddle_H1.x and self.ball.x <= self.paddle_H1.x + self.paddle_H1.width and self.ball.y >= self.paddle_H1.y:
                self.ball.y_vel *= -1
                ball_distance_paddle_x = (abs(
                    self.ball.x - self.paddle_H1.x) // self.width_scale)

                if(ball_distance_paddle_x == 0 or ball_distance_paddle_x == 6):
                    x_vel = self.ball.x_vel*3
                elif ball_distance_paddle_x == 1 or ball_distance_paddle_x == 5:
                    x_vel = self.ball.x_vel*2
                elif ball_distance_paddle_x == 2 or ball_distance_paddle_x == 4:
                    x_vel = self.ball.x_vel*2
                else:
                    x_vel = self.ball.x_vel

                if abs(x_vel) / self.width_scale > self.ball.MAX_VEL:
                    x_vel = - self.ball.MAX_VEL * \
                        self.width_scale if x_vel < 0 else self.ball.MAX_VEL * self.width_scale
                self.ball.x_vel = -x_vel
                return 1

        # check if the ball hits the paddle H2
        if self.ball.y_vel < 0:
            if self.ball.x >= self.paddle_H2.x and self.ball.x <= self.paddle_H2.x + self.paddle_H2.width and self.ball.y <= self.paddle_H2.y + self.paddle_H2.height:
                self.ball.y_vel *= -1
                ball_distance_paddle_x = (abs(
                    self.ball.x - self.paddle_H2.x) // self.width_scale)

                if(ball_distance_paddle_x == 0 or ball_distance_paddle_x == 6):
                    x_vel = self.ball.x_vel*3
                elif ball_distance_paddle_x == 1 or ball_distance_paddle_x == 5:
                    x_vel = self.ball.x_vel*2
                elif ball_distance_paddle_x == 2 or ball_distance_paddle_x == 4:
                    x_vel = self.ball.x_vel*2
                else:
                    x_vel = self.ball.x_vel

                if abs(x_vel) / self.width_scale > self.ball.MAX_VEL:
                    x_vel = - self.ball.MAX_VEL * \
                        self.width_scale if x_vel < 0 else self.ball.MAX_VEL * self.width_scale
                self.ball.x_vel = -x_vel
                return 1

        # check if the ball hits the paddle V2
        # direction: --->
        if self.ball.x_vel > 0:
            if self.ball.y >= self.paddle_V2.y and self.ball.y <= self.paddle_V2.y + self.paddle_V2.height:
                if self.ball.x >= self.paddle_V2.x:
                    self.ball.x_vel *= -1
                    ball_distance_paddle_y = (abs(
                        self.ball.y - self.paddle_V2.y) // self.height_scale)

                    if(ball_distance_paddle_y == 0 or ball_distance_paddle_y == 6):
                        y_vel = self.ball.y_vel*3
                    elif ball_distance_paddle_y == 1 or ball_distance_paddle_y == 5:
                        y_vel = self.ball.y_vel*2
                    elif ball_distance_paddle_y == 2 or ball_distance_paddle_y == 4:
                        y_vel = self.ball.y_vel*2
                    else:
                        y_vel = self.ball.y_vel

                    if abs(y_vel) / self.height_scale > self.ball.MAX_VEL:
                        y_vel = - self.ball.MAX_VEL * \
                            self.height_scale if y_vel < 0 else self.ball.MAX_VEL * self.height_scale
                    self.ball.y_vel = -y_vel
                    return 1

        # check if the ball hits the paddle V1
        # direction:  <---
        if self.ball.x_vel < 0:
            if self.ball.y >= self.paddle_V1.y and self.ball.y <= self.paddle_V1.y + self.paddle_V1.height and self.ball.x <= self.paddle_V1.x+self.paddle_V1.width:
                self.ball.x_vel *= -1
                ball_distance_paddle_y = (abs(
                    self.ball.y - self.paddle_V1.y) // self.height_scale)

                if(ball_distance_paddle_y == 0 or ball_distance_paddle_y == 6):
                    y_vel = self.ball.y_vel*3
                elif ball_distance_paddle_y == 1 or ball_distance_paddle_y == 5:
                    y_vel = self.ball.y_vel*2
                elif ball_distance_paddle_y == 2 or ball_distance_paddle_y == 4:
                    y_vel = self.ball.y_vel*2
                else:
                    y_vel = self.ball.y_vel

                if abs(y_vel) / self.height_scale > self.ball.MAX_VEL:
                    y_vel = - self.ball.MAX_VEL * \
                        self.height_scale if y_vel < 0 else self.ball.MAX_VEL * self.height_scale
                self.ball.y_vel = -y_vel
                return 1

        return 0

    def drawGrid(self):
        for x in range(0, self.window_width, self.width_scale):
            pygame.draw.rect(self.window, self.GREY,
                             (x, 0, 1, self.window_height))
        for y in range(0, self.window_width, self.height_scale):
            pygame.draw.rect(self.window, self.GREY,
                             (0, y, self.window_width, 1))

    def draw(self):
        self.window.fill(self.BLACK)
        self.draw_score()
        self.paddle_H1.draw(self.window)
        self.paddle_H2.draw(self.window)
        self.paddle_V1.draw(self.window)
        self.paddle_V2.draw(self.window)
        self.ball.draw(self.window)
        self.drawGrid()

    def loop(self):
        self.ball.move()
        self.score += self.handle_collision()

        game_info = GameInformation(self.score)

        return game_info

    def reset(self):
        self.ball.reset()
        self.paddle_H1.reset()
        self.score = 0
