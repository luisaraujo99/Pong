import pygame

WHITE = (255, 255, 255)


class Paddle:
    VEL = 4
    HORIZ_PAD_WIDTH = 100
    HORIZ_PAD_HEIGHT = 20
    VERT_PAD_WIDTH = 20
    VERT_PAD_HEIGHT = 100

    def __init__(self, x, y, width, height, vertical):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.height = height
        self.width = width
        self.vertical = vertical

    def draw(self, win):
        pygame.draw.rect(
            win, WHITE, (self.x, self.y, self.width, self.height))

    def move(self,isFourPongPad=False,positive=True, **kwargs):
        
        window_width=kwargs['window_width']
        
        
        #Four pad logic
        if isFourPongPad:
            window_height=kwargs['window_height'] 
            if self.vertical == False:
                LEFT_THRESHOLD = not positive and self.x - \
                    Paddle.VEL < (self.VERT_PAD_WIDTH + 10)
                RIGHT_THRESHOLD = positive and self.x + \
                    self.HORIZ_PAD_WIDTH + Paddle.VEL > window_width - \
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
                    Paddle.VEL < self.VERT_PAD_WIDTH + 10
                BOTTOM_THRESHOLD = positive and self.y + \
                    self.VERT_PAD_HEIGHT + Paddle.VEL > window_height - \
                    (self.VERT_PAD_WIDTH + 10)
    
                if UPPER_THRESHOLD:
                    return False
                if BOTTOM_THRESHOLD:
                    return False
    
                if positive:
                    self.y += self.VEL
                else:
                    self.y -= self.VEL
        #single pad or two pads logic
        else:
            LEFT_THRESHOLD = not positive and self.x - Paddle.VEL < 5
            RIGHT_THRESHOLD = positive and self.x + \
                self.HORIZ_PAD_WIDTH + Paddle.VEL > window_width-5

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
