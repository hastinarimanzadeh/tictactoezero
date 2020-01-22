from enum import Enum
import numpy as np
class TicTacToe():

    class Player(Enum):
        X = 1
        O = 2

    class GameState(Enum):
        XWin = 1
        OWin = 2
        Tie = 3
        Continue = 4

    """initalizes an empty board"""
    def __init__(self):
        self.current_player = TicTacToe.Player.X
        self.x_matrix = np.zeros((3,3))
        self.o_matrix = np.zeros((3,3))

    def turn(self):
        return self.current_player

    def legal_moves(self):
        if self.state() is TicTacToe.GameState.Continue:
            return [(i,j) for i in range(3) for j in range(3) if self.x_matrix[i,j] == 0 and self.o_matrix[i,j] == 0]
        else:
            return []

    """returns a TicTacToe.GameState"""
    def state(self):
        for i in range(3):
            if self.x_matrix[:,i].all() or self.x_matrix[i,:].all() or self.x_matrix.diagonal().all():
                return TicTacToe.GameState.XWin
            elif self.o_matrix[:,i].all() or self.o_matrix[i,:].all() or self.o_matrix.diagonal().all():
                return TicTacToe.GameState.OWin

        if not (self.x_matrix == self.o_matrix).any():
            return TicTacToe.GameState.Tie
        else:
            return TicTacToe.GameState.Continue

    def play(self, move):
        # play the move
        row, col = move
        if self.current_player is TicTacToe.Player.X:
            self.x_matrix[row, col] = 1
        else:
            self.o_matrix[row, col] = 1

        # give the turn to the next player
        if self.current_player is TicTacToe.Player.X:
            self.current_player = TicTacToe.Player.O
        else:
            self.current_player = TicTacToe.Player.X

    def draw_baord(self):
        occupants = ['x', 'o', '-']
        print('\n')
        for i in range(3):
            for j in range(3):
                if self.x_matrix[i,j] == 1:
                    print(occupants[0], end=" "*4)
                elif self.o_matrix[i,j] == 1:
                    print(occupants[1], end=" "*4)
                else:
                    print(occupants[2], end=" "*4)
            print('\n')
