import pygame

WHITE = (255, 255, 255)


class Paddle:

    def __init__(self, x, y, width, height, vertical, vel):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.height = height
        self.width = width
        self.vertical = vertical
        self.VERT_PAD_WIDTH = height
        self.VERT_PAD_HEIGHT = width
        self.VEL = vel

    def draw(self, win):
        pygame.draw.rect(
            win, WHITE, (self.x, self.y, self.width, self.height))

    def move(self, isFourPongPad=False, positive=True, **kwargs):

        window_width = kwargs['window_width']

        # Four pad logic
        if isFourPongPad:
            window_height = kwargs['window_height']
            if self.vertical == False:
                LEFT_THRESHOLD = not positive and self.x - \
                    Paddle.VEL < (self.VERT_PAD_WIDTH + 10)
                RIGHT_THRESHOLD = positive and self.x + \
                    self.width + Paddle.VEL > window_width - \
                    (self.VERT_PAD_WIDTH + 10)

                if LEFT_THRESHOLD:
                    return False
                if RIGHT_THRESHOLD:
                    return False

                if positive:
                    self.x += self.VEL
                else:
                    self.x -= self.VEL

            else:
                UPPER_THRESHOLD = not positive and self.y - \
                    self.VEL < self.VERT_PAD_WIDTH + 10
                BOTTOM_THRESHOLD = positive and self.y + \
                    self.VERT_PAD_HEIGHT + self.VEL > window_height - \
                    (self.VERT_PAD_WIDTH + 10)

                if UPPER_THRESHOLD:
                    return False
                if BOTTOM_THRESHOLD:
                    return False

                if positive:
                    self.y += self.VEL
                else:
                    self.y -= self.VEL
        # single pad or two pads logic
        else:
            LEFT_THRESHOLD = not positive and self.x - self.VEL < 0
            RIGHT_THRESHOLD = positive and self.x + \
                self.width + self.VEL > window_width

            if LEFT_THRESHOLD:
                return False
            if RIGHT_THRESHOLD:
                return False

            if positive:
                self.x += self.VEL
            else:
                self.x -= self.VEL

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
