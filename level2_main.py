from Pong import DoublePadPong
import pygame

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)

class PongGame:
    def __init__(self, window, width, height):
        self.game = DoublePadPong(window, width, height)
        self.ball = self.game.ball
        self.paddle = self.game.paddle_H1
        
        
    def test_ai(self):
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(60)
            game_info = self.game.loop()
            

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT]:
                self.game.paddle_H1.move(False,True,window_width=WIDTH)
            elif keys[pygame.K_LEFT]:
                self.game.paddle_H1.move(False,False,window_width=WIDTH)

            if keys[pygame.K_d]:
                self.game.paddle_H2.move(False,True,window_width=WIDTH)
            elif keys[pygame.K_a]:
                self.game.paddle_H2.move(False,False,window_width=WIDTH)


            self.game.draw()
            pygame.display.update()
            if game_info.score == 10:
                self.game.reset()

        pygame.quit()


def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong")
    pong = PongGame(win, WIDTH, HEIGHT)
    pong.test_ai()


main()
