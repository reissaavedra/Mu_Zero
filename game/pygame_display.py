import pygame
import sys


class Pygame_Tictactoe:
    def __init__(self):
        self.screenSize = 600
        self.margin = 50
        self.gameSize = 600 - (2 * self.margin)
        self.lineSize = 10
        self.backgroundColor = (0, 0, 0)
        self.lineColor = (255, 255, 255)
        self.xColor = (200, 0, 0)
        self.oColor = (0, 0, 200)
        self.xMark = 'X'
        self.oMark = 'o'
        self.board = [[None, None, None], [None, None, None], [None, None, None]]
        pygame.init()
        self.screen = pygame.display.set_mode((self.screenSize, self.screenSize))
        pygame.display.set_caption("Tic Tac Toe")
        pygame.font.init()
        self.myFont = pygame.font.SysFont('Tahoma', self.gameSize // 3)
        self.screen.fill(self.backgroundColor)
        self.canPlay = True
        self.draw_lines()

    def draw_lines(self):
        # Vertical lines
        pygame.draw.line(self.screen, self.lineColor, (self.margin + self.gameSize // 3, self.margin),
                         (self.margin + self.gameSize // 3, self.screenSize - self.margin), self.lineSize)
        pygame.draw.line(self.screen, self.lineColor, (self.margin + (self.gameSize // 3) * 2, self.margin),
                         (self.margin + (self.gameSize // 3) * 2, self.screenSize - self.margin), self.lineSize)

    def draw_board(self, board):
        self.myFont = pygame.font.SysFont('Tahoma', self.gameSize // 3)
        for y in range(3):
            for x in range(3):
                if board[y][x] == 1:
                    mark = self.xMark
                    color = self.xColor
                elif board[y][x] == -1:
                    mark = self.oMark
                    color = self.oColor
                text_surface = self.myFont.render(mark, False, color)
                self.screen.blit(text_surface,
                                (y * (self.gameSize // 3) + self.margin + (self.gameSize // 18),
                                 x * (self.gameSize // 3) + self.margin))
