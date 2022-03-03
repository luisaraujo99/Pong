from Pong import SinglePadPong
from Pong import DoublePadPong
from Pong import FourPadPong
import pygame

WIDTH, HEIGHT = 800, 600


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
                self.game.paddle_H1.move2(True, HEIGHT, WIDTH)
            elif keys[pygame.K_LEFT]:
                self.game.paddle_H1.move2(False, HEIGHT, WIDTH)

            if keys[pygame.K_d]:
                self.game.paddle_H2.move2(True, HEIGHT, WIDTH)
            elif keys[pygame.K_a]:
                self.game.paddle_H2.move2(False, HEIGHT, WIDTH)

            if keys[pygame.K_w]:
                self.game.paddle_V1.move2(False, HEIGHT, WIDTH)
            elif keys[pygame.K_s]:
                self.game.paddle_V1.move2(True, HEIGHT, WIDTH)

            if keys[pygame.K_UP]:
                self.game.paddle_V2.move2(False, HEIGHT, WIDTH)
            elif keys[pygame.K_DOWN]:
                self.game.paddle_V2.move2(True, HEIGHT, WIDTH)

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
