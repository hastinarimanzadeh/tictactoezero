from enum import Enum

class Player(Enum):
    X = 1
    O = 2

class GameState(Enum):
    Win = 1
    Tie = 2
    Continue = 3
