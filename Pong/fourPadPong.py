from matplotlib.ft2font import HORIZONTAL
from .paddle import Paddle
from .ball import Ball
import pygame
import operator


COOPERATION = 1
TEAM_COOPERATION = 2
ALL_VS_ALL = 3
PERSONALITY_COOPERATION = 4


class GameInformation:
    def __init__(self, score):
        self.score = score


class FourPadPong:
    pygame.init()

    WHITE = (255, 255, 255)
    GREY = (211, 211, 211)
    CYAN = (0, 255, 255)
    BLACK = (0, 0, 0)
    SCORE_FONT = pygame.font.SysFont("comicsans", 45)

    def __init__(self, window, window_width, window_height, width_scale, height_scale, PAD_SIZE, GAME_DIM_X, GAME_DIM_Y):
        self.window_width = window_width
        self.window_height = window_height
        self.width_scale = width_scale
        self.height_scale = height_scale

        self.paddle1 = Paddle(width_scale*(GAME_DIM_X//2), height_scale,
                              width_scale*PAD_SIZE, height_scale, False, width_scale)

        self.paddle2 = Paddle(width_scale*(GAME_DIM_X//2), self.window_height -
                              2*height_scale, width_scale*PAD_SIZE, height_scale, False, width_scale)

        self.paddle3 = Paddle(width_scale, height_scale*(GAME_DIM_Y//2),
                              width_scale, height_scale*PAD_SIZE, True, height_scale)

        self.paddle4 = Paddle(window_width-2*width_scale, height_scale*(GAME_DIM_Y//2),
                              width_scale, height_scale*PAD_SIZE, True, height_scale)

        self.ball = Ball(width_scale*(GAME_DIM_X//2), height_scale*(GAME_DIM_Y//2),
                         self.width_scale, self.height_scale, window_width, window_height, GAME_DIM_X, radius=5)

        self.score = (0, 0, 0, 0)
        self.window = window

    def draw_score(self):
        score_text1 = self.SCORE_FONT.render(f"{self.score[0]}", 1, self.CYAN)
        score_text2 = self.SCORE_FONT.render(f"{self.score[1]}", 1, self.CYAN)
        score_text3 = self.SCORE_FONT.render(f"{self.score[2]}", 1, self.CYAN)
        score_text4 = self.SCORE_FONT.render(f"{self.score[3]}", 1, self.CYAN)
        self.window.blit(score_text1, (15, 20))
        self.window.blit(score_text2, (self.window_width-50, 20))
        self.window.blit(score_text3, (15, self.window_height-50))
        self.window.blit(
            score_text4, (self.window_width-50, self.window_height-50))

    def handle_ball_paddle_collision(self, ball_x=None, ball_y=None, paddle_number=1):
        ''' function to handle ball speed and direction change when touches a paddle '''
        if paddle_number == 1:
            x_paddle_center = self.paddle1.x + self.paddle1.width//2
            ball_distance_paddle_x = (
                ball_x - x_paddle_center)//self.width_scale
        elif paddle_number == 2:
            x_paddle_center = self.paddle2.x + self.paddle2.width//2
            ball_distance_paddle_x = (
                ball_x - x_paddle_center)//self.width_scale
        elif paddle_number == 3:
            y_paddle_center = self.paddle3.y + self.paddle3.height//2
            ball_distance_paddle_y = (
                ball_y - y_paddle_center)//self.height_scale
        elif paddle_number == 4:
            y_paddle_center = self.paddle4.y + self.paddle4.height//2
            ball_distance_paddle_y = (
                ball_y - y_paddle_center)//self.height_scale

        x_vel = self.ball.x_vel
        y_vel = self.ball.y_vel

        # code for pad 1 and 2
        if paddle_number <= 2:
            if ball_distance_paddle_x < 0:
                if x_vel == 0:
                    x_vel = self.ball.original_x_vel
                x_vel = abs(x_vel)*(int(ball_distance_paddle_x))
            if ball_distance_paddle_x > 0:
                if x_vel == 0:
                    x_vel = self.ball.original_x_vel
                x_vel = abs(x_vel)*(int(ball_distance_paddle_x))
            if ball_distance_paddle_x == 0:
                x_vel = 0
                y_vel *= 2
        # code for pad 3 and 4
        if paddle_number >= 3:
            if ball_distance_paddle_y < 0:
                if y_vel == 0:
                    y_vel = self.ball.original_y_vel
                y_vel = abs(y_vel)*(int(ball_distance_paddle_y))
            if ball_distance_paddle_y > 0:
                if y_vel == 0:
                    y_vel = self.ball.original_y_vel
                y_vel = abs(y_vel)*(int(ball_distance_paddle_y))
            if ball_distance_paddle_y == 0:
                y_vel = 0
                x_vel *= 2

        # speed constraints
        if abs(x_vel) // self.width_scale > Ball.MAX_VEL_X:
            x_vel = - Ball.MAX_VEL_X * \
                self.width_scale if x_vel < 0 else Ball.MAX_VEL_X * self.width_scale
        if abs(y_vel) // self.width_scale > Ball.MAX_VEL_Y:
            y_vel = - Ball.MAX_VEL_Y * \
                self.height_scale if y_vel < 0 else Ball.MAX_VEL_Y * self.height_scale

        self.ball.x_vel = x_vel
        self.ball.y_vel = y_vel

    def reward_type(self, paddle_reward, type=COOPERATION):
        ''' function used to create a reward to manipulate the way the paddles behave '''
        ''' paddle_reward = 1 means Paddle1 caught the ball  '''
        ''' paddle_reward = -1 means Paddle1 did not catch the ball  '''
        if type == COOPERATION:
            if paddle_reward == 1 or paddle_reward == 2 or paddle_reward == 3 or paddle_reward == 4:
                return (1, 1, 1, 1)
            if paddle_reward == -3 or paddle_reward == -4 or paddle_reward == -1 or paddle_reward == -2:
                return (-1, -1, -1, -1)
        if type == TEAM_COOPERATION:
            if paddle_reward == 1 or paddle_reward == 2 or paddle_reward == -3 or paddle_reward == -4:
                return (1, 1, -1, -1)
            if paddle_reward == 3 or paddle_reward == 4 or paddle_reward == -1 or paddle_reward == -2:
                return (-1, -1, 1, 1)
        if type == PERSONALITY_COOPERATION:
            if paddle_reward == 1 or paddle_reward == 2 or paddle_reward == -3 or paddle_reward == -4:
                return (10, 10, -1, -1)
            if paddle_reward == 3 or paddle_reward == 4 or paddle_reward == -1 or paddle_reward == -2:
                return (-10, -10, 1, 1)

    def handle_collision(self):
        ''' function to handle the ball collision with the whole environment '''
        (ball_x, ball_y) = self.ball.move()
        reward = (0, 0, 0, 0)

        # LEFT PAD 3 LINE
        if ball_x < self.paddle3.x+self.paddle3.width:
            lambda_x = (self.ball.x-(self.paddle3.x +
                        self.paddle3.width))/self.ball.x_vel
            ball_y = self.ball.y+lambda_x*self.ball.y_vel
            ball_x = self.paddle3.x+self.paddle3.width
            if ball_y >= self.paddle3.y and ball_y <= self.paddle3.y+self.paddle3.height:
                self.ball.x_vel *= -1
                self.handle_ball_paddle_collision(
                    ball_y=ball_y, paddle_number=3)
                ball_y = ball_y+(1-lambda_x)*self.ball.y_vel
                ball_x = ball_x + (1-lambda_x)*self.ball.x_vel
                reward = self.reward_type(3)
            else:
                self.ball.reset(isFourPadPong=True)
                return self.reward_type(-3)

        # RIGHT PAD 4 LINE
        if ball_x > self.paddle4.x:
            lambda_x = (self.paddle4.x - self.ball.x)/self.ball.x_vel
            ball_y = self.ball.y+lambda_x*self.ball.y_vel
            ball_x = self.paddle4.x
            if ball_y >= self.paddle4.y and ball_y <= self.paddle4.y+self.paddle4.height:
                self.ball.x_vel *= -1
                self.handle_ball_paddle_collision(
                    ball_y=ball_y, paddle_number=4)
                ball_y = ball_y+(1-lambda_x)*self.ball.y_vel
                ball_x = ball_x + (1-lambda_x)*self.ball.x_vel
                reward = self.reward_type(4)
            else:
                self.ball.reset(isFourPadPong=True)
                return self.reward_type(-4)

        # BOTTOM PAD 2 LINE
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
                self.ball.reset(isFourPadPong=True)
                return self.reward_type(-2)

        # TOP PAD 1 LINE
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
                self.ball.reset(isFourPadPong=True)
                return self.reward_type(-1)

        ball_x = int(ball_x)
        ball_y = int(ball_y)

        HORIZONTAL_LIMIT = (ball_x >= self.paddle3.x +
                            self.paddle3.width and ball_x <= self.paddle4.x)
        VERTICAL_LIMIT = (ball_y >= self.paddle1.y +
                          self.paddle1.height and ball_y <= self.paddle2.y)

        # NEW POSITION IS VALID
        if HORIZONTAL_LIMIT and VERTICAL_LIMIT:
            self.ball.x = ball_x
            self.ball.y = ball_y
            return reward

        # NEW POSITION IS OUT OF BOUNDs
        else:
            if ball_x < self.paddle3.x+self.paddle3.width:
                self.ball.x = self.paddle3.x+self.paddle3.width
                self.ball.y = ball_y
                return reward
            elif ball_x > self.paddle4.x:
                self.ball.x = self.paddle4.x
                self.ball.y = ball_y
                return reward
            elif ball_y < self.paddle1.y+self.paddle4.height:
                self.ball.x = ball_x
                self.ball.y = self.paddle1.y+self.paddle4.height
                return reward
            elif ball_y > self.paddle2.y:
                self.ball.x = ball_x
                self.ball.y = self.paddle2.y
                return reward

    def drawGrid(self):
        for x in range(0, self.window_width, self.width_scale):
            pygame.draw.rect(self.window, self.GREY,
                             (x, 0, 1, self.window_height))
        for y in range(0, self.window_height, self.height_scale):
            pygame.draw.rect(self.window, self.GREY,
                             (0, y, self.window_width, 1))

    def draw(self):
        self.window.fill(self.BLACK)
        self.draw_score()
        self.paddle1.draw(self.window)
        self.paddle2.draw(self.window)
        self.paddle3.draw(self.window)
        self.paddle4.draw(self.window)
        self.ball.draw(self.window)
        # self.drawGrid()

    def loop(self):
        self.score = tuple(
            map(operator.add, self.score, self.handle_collision()))

        game_info = GameInformation(self.score)

        return game_info

    def reset(self):
        """Resets the entire game."""
        self.ball.reset()
        self.paddle.reset()
        self.score = 0
