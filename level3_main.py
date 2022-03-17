from Pong import FourPadPong
import pygame

WIDTH_SCALE, HEIGHT_SCALE = 30, 20
WIDTH, HEIGHT = 40*WIDTH_SCALE, 40*HEIGHT_SCALE
WHITE = (255, 255, 255)


class PongGame:
    def __init__(self, window, width, height):
        self.game = FourPadPong(
            window, width, height, WIDTH_SCALE, HEIGHT_SCALE)
        self.ball = self.game.ball

    def test_ai(self):
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(5)
            game_info = self.game.loop()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT]:
                self.game.paddle_H1.move(
                    True, True, window_width=WIDTH, window_height=HEIGHT)
            elif keys[pygame.K_LEFT]:
                self.game.paddle_H1.move(
                    True, False, window_width=WIDTH, window_height=HEIGHT)

            if keys[pygame.K_d]:
                self.game.paddle_H2.move(
                    True, True, window_width=WIDTH, window_height=HEIGHT)
            elif keys[pygame.K_a]:
                self.game.paddle_H2.move(
                    True, False, window_width=WIDTH, window_height=HEIGHT)

            if keys[pygame.K_w]:
                self.game.paddle_V1.move(
                    True, False, window_width=WIDTH, window_height=HEIGHT)
            elif keys[pygame.K_s]:
                self.game.paddle_V1.move(
                    True, True, window_width=WIDTH, window_height=HEIGHT)

            if keys[pygame.K_UP]:
                self.game.paddle_V2.move(
                    True, False, window_width=WIDTH, window_height=HEIGHT)
            elif keys[pygame.K_DOWN]:
                self.game.paddle_V2.move(
                    True, True, window_width=WIDTH, window_height=HEIGHT)

            self.game.draw()
            pygame.display.update()

        pygame.quit()


def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong")
    pong = PongGame(win, WIDTH, HEIGHT)
    pong.test_ai()


main()
