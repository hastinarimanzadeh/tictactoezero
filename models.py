import random
from game import GameState

class RandomModel:
    def evaluate(self, state):
        initial_turn = state.turn()
        while state.state() is GameState.Continue:
            move = random.choice(state.legal_moves())
            state.play(move)
        if state.state() is GameState.Tie:
            return 0
        elif initial_turn is state.winner():
            return -1
        else:
            return +1
    
