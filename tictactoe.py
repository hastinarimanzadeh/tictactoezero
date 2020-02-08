import numpy as np
from game import Player, GameState
class TicTacToe:
    """initalizes an empty board"""
    def __init__(self):
        self.__current_player = Player.X
        self.__x_matrix = np.zeros((3,3))
        self.__o_matrix = np.zeros((3,3))
        self.__winner = None
        self.__current_state = GameState.Continue

    def turn(self):
        return self.__current_player

    def legal_moves(self):
        if self.__current_state is GameState.Continue:
            return [(i,j) for i in range(3) for j in range(3) if self.__x_matrix[i,j] == 0 and self.__o_matrix[i,j] == 0]
        else:
            return []

    """returns a GameState"""
    def state(self):
        return self.__current_state

    def winner(self):
        return self.__winner

    def __calculate_state(self):
        for i in range(3):
            if self.__x_matrix[:, i].all() or \
                self.__x_matrix[i, :].all() or \
                self.__x_matrix.diagonal().all() or \
                np.fliplr(self.__x_matrix).diagonal().all():
                return GameState.Win
            elif self.__o_matrix[:, i].all() or \
                self.__o_matrix[i, :].all() or \
                self.__o_matrix.diagonal().all() or \
                np.fliplr(self.__o_matrix).diagonal().all():
                return GameState.Win

        if not (self.__x_matrix == self.__o_matrix).any():
            return GameState.Tie
        else:
            return GameState.Continue

    def play(self, move):
        # play the move
        row, col = move
        if self.__current_player is Player.X:
            self.__x_matrix[row, col] = 1
        else:
            self.__o_matrix[row, col] = 1

        # check game state and check who the winner is
        self.__current_state = self.__calculate_state()
        if self.__current_state is GameState.Win:
            self.__winner = self.__current_player

        # give the turn to the next player
        if self.__current_player is Player.X:
            self.__current_player = Player.O
        else:
            self.__current_player = Player.X

    def to_tensor(self):
        turn = np.full(
            (3, 3),
            1 if self.__current_player is Player.X else -1,
            dtype=np.float32)
        return np.stack((
                turn,
                self.__x_matrix,
                self.__o_matrix), axis=-1).astype(np.float32)

    def policy_to_tensor(self, policy):
        policy_tensor = np.zeros((3, 3, 1))
        for (i, j), s in policy:
            policy_tensor[i, j, 0] = s
        return policy_tensor.astype(np.float32)

    def __repr__(self):
        occupants = ['x', 'o', '-']
        s = "<TicTacToe\n\tstate: " + str(self.__current_state) + "\n"
        s += "\tturn: " + str(self.__current_player) + "\n"
        if self.__winner:
            s += "\twinner: " + str(self.__winner) + "\n"
        for i in range(3):
            s += "\t"
            for j in range(3):
                if self.__x_matrix[i,j] == 1:
                    s += occupants[0] + " "*4
                elif self.__o_matrix[i,j] == 1:
                    s += occupants[1] + " "*4
                else:
                    s += occupants[2] +" "*4
            s += "\n"
        s += ">"
        return s
