from Pong import SinglePadPong
from Pong import Q_AI
import pygame
from alive_progress import alive_bar


WIDTH_SCALE, HEIGHT_SCALE = 30, 20
GAME_DIM = 39
WIDTH, HEIGHT = GAME_DIM*WIDTH_SCALE, GAME_DIM*HEIGHT_SCALE


class PongGame:
    def __init__(self, window, width, height):
        self.game = SinglePadPong(
            window, width, height, WIDTH_SCALE, HEIGHT_SCALE)
        self.ball = self.game.ball
        self.paddle = self.game.paddle

    def reward(self, initial_score, end_score):
        return end_score - initial_score

    def Q_learning_algorithm(self, epochs=1000, episodes=2000):
        clock = pygame.time.Clock()
        run = True
        q_ai = Q_AI(0.1, 0.97, GAME_DIM, 0.01, epochs, episodes)
        q_ai.load_file()

        epoch = 0
        while epoch < epochs:
            initial_nonzeros_ratio = q_ai.matrix_ratio()
            episode = 0
            game_info = self.game.loop()
            with alive_bar(episodes, bar='blocks', title=f'Epoch {epoch}', spinner='arrows') as bar:
                while episode < episodes and run:

                    clock.tick(700)

                    init_score = game_info.score

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            run = False
                            break

                    state = ((self.ball.x//WIDTH_SCALE), (self.ball.y //
                             HEIGHT_SCALE), (self.paddle.x//WIDTH_SCALE))

                    action = q_ai.prob_action(state)

                    if action == 2:
                        self.game.paddle.move(False, True, window_width=WIDTH)
                    elif action == 1:
                        self.game.paddle.move(False, False, window_width=WIDTH)

                    self.game.draw()
                    pygame.display.update()

                    game_info = self.game.loop()

                    end_score = self.game.score

                    r = self.reward(init_score, end_score)

                    new_state = ((self.ball.x//WIDTH_SCALE), (self.ball.y //
                                                              HEIGHT_SCALE), (self.paddle.x//WIDTH_SCALE))
                    q_ai.q(action, r, state, new_state)

                    episode += 1
                    bar.text(
                        f'\n-> Non zero ratio: {initial_nonzeros_ratio}')
                    bar()

            epoch += 1
            q_ai.save_state()

        pygame.quit()


def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong")
    pong = PongGame(win, WIDTH, HEIGHT)
    pong.Q_learning_algorithm()


main()
