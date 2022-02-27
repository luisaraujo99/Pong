from Pong import SinglePadPong
import pygame


class PongGame:
    def __init__(self, window, width, height):
        self.game = SinglePadPong(window, width, height)
        self.ball = self.game.ball
        self.paddle = self.game.paddle

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
                self.game.move_paddle(right=True)
            elif keys[pygame.K_LEFT]:
                self.game.move_paddle(right=False)

            self.game.draw()
            pygame.display.update()
            if game_info.score == 10:
                self.game.reset()
        
        pygame.quit()


def main():
    width, height = 700, 500
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pong")
    pong = PongGame(win, width, height)
    pong.test_ai()

main()