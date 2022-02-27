from .paddle import Paddle
from .ball import Ball
import pygame




class GameInformation:
    def __init__(self, score):
        self.score=score
            
    
class SinglePadPong:
    pygame.init()
    
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    SCORE_FONT = pygame.font.SysFont("comicsans", 50)
    
    def __init__(self, window, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height

        self.paddle = Paddle(self.window_width//2 - Paddle.WIDTH//2,self.window_height-30)
        self.ball = Ball(self.window_width // 2, self.window_height // 5)

        self.score = 0
        self.window = window
    
    def draw_score(self):
        score_text = self.SCORE_FONT.render(f"{self.score}",1,self.WHITE)
        self.window.blit(score_text, (15,20))
        
    def handle_collision(self):
        #upper wall
        if self.ball.y - self.ball.RADIUS <= 0 :
            self.ball.y_vel *=-1
            return 0
        #right wall
        elif self.ball.x + self.ball.RADIUS >= self.window_width:
            self.ball.x_vel *=-1
            return 0
        #left wall
        elif self.ball.x - self.ball.RADIUS <= 0:
            self.ball.x_vel *=-1
            return 0
        elif self.ball.y > self.window_height:
            self.ball.reset()
            if self.score > 0:
                return -1
            else:
                return 0
            
           
        #check if the ball hits the paddle
        if self.ball.y_vel > 0:
            if self.ball.x >= self.paddle.x and self.ball.x <= self.paddle.x + Paddle.WIDTH:
                if self.ball.y+self.ball.RADIUS >= self.paddle.y:
                    self.ball.y_vel *=-1
                    middle_x = self.paddle.x + Paddle.WIDTH/2
                    diff_ball_pad_center = middle_x - self.ball.x
                    reduction_factor = (Paddle.WIDTH / 2)/self.ball.MAX_VEL
                    x_vel = diff_ball_pad_center / reduction_factor
                    self.ball.x_vel = -x_vel
                    return 1
        return 0
    
    def draw(self):
        self.window.fill(self.BLACK)
        self.draw_score()
        self.paddle.draw(self.window)
        self.ball.draw(self.window)
        
    def move_paddle(self,right=True):
        LEFT_THRESHOLD = not right and self.paddle.x - Paddle.VEL < 0
        RIGHT_THRESHOLD = right and self.paddle.x + Paddle.WIDTH + Paddle.VEL > self.window_width
        
        if LEFT_THRESHOLD:
            return False
        if RIGHT_THRESHOLD:
            return False
        
        self.paddle.move(right)
        return True
    
    def loop(self):
        """
        Executes a single game loop.
        :returns: GameInformation instance stating score 
                  and hits of each paddle.
        """
        self.ball.move()
        self.score += self.handle_collision()

        game_info = GameInformation(self.score)

        return game_info
    
    def reset(self):
        """Resets the entire game."""
        self.ball.reset()
        self.paddle.reset()
        self.score = 0