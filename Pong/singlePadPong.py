from .paddle import Paddle
from .ball import Ball
import pygame


class GameInformation:
    def __init__(self, score):
        self.score = score


class SinglePadPong:
    pygame.init()

    WHITE = (255, 255, 255)
    GREY = (128, 128, 128)
    CYAN = (0, 255, 255)
    BLACK = (0, 0, 0)
    SCORE_FONT = pygame.font.SysFont("comicsans", 50)

    def __init__(self, window, window_width, window_height, width_scale, height_scale):
        self.window_width = window_width
        self.window_height = window_height
        self.width_scale = width_scale
        self.height_scale = height_scale

        self.paddle = Paddle(width_scale*11, self.window_height -
                             2*height_scale, width_scale*6, height_scale, False, width_scale)
        self.ball = Ball(width_scale*20, height_scale*12,
                         self.width_scale, self.height_scale,window_width,window_height)

        self.score = 0
        self.window = window

    def draw_score(self):
        score_text = self.SCORE_FONT.render(f"{self.score}", 1, self.WHITE)
        self.window.blit(score_text, (15, 20))

    def handle_collision(self):
       
        # didn't catch
        if self.ball.y >= self.window_height:
            self.ball.reset()
            return -1

        # check if the ball hits the paddle
        if self.ball.y_vel > 0:
            if self.ball.x >= self.paddle.x and self.ball.x <= self.paddle.x + self.paddle.width and self.ball.y >= self.paddle.y:
                self.ball.y_vel *= -1
                ball_distance_paddle_x = (abs(
                    self.ball.x - self.paddle.x) // self.width_scale)

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
        return 0

    def drawGrid(self):
        for x in range(0, self.window_width, self.width_scale):
            pygame.draw.rect(self.window, self.GREY,
                             (x, 0, 1, self.window_height))
        for y in range(0, self.window_width, self.height_scale):
            pygame.draw.rect(self.window, self.GREY,
                             (0, y, self.window_width, 1))

    def drawBallFlowGrid(self):
        for x in range(0, self.window_width, self.width_scale):
            pygame.draw.rect(self.window, self.GREY,
                             (x, 0, 1, self.window_height))
        for y in range(0, self.window_width, self.height_scale):
            pygame.draw.rect(self.window, self.GREY,
                             (0, y, self.window_width, 1))

    def draw(self):
        self.window.fill(self.BLACK)
        self.draw_score()
        self.paddle.draw(self.window)
        self.ball.draw(self.window)
        self.drawGrid()


    def loop(self):
        self.ball.move()
        self.score += self.handle_collision()

        game_info = GameInformation(self.score)

        return game_info

    def reset(self):
        """Resets the entire game."""
        self.ball.reset()
        self.paddle.reset()
        self.score = 0
