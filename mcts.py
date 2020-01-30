import numpy as np
import random
from copy import deepcopy
from game import GameState
class Node:
    def __init__(self, parent, action):
        self.__action = action
        self.__parent = parent
        self.wins = 0
        self.simulations = 0
        self.__children = []

    def add_child(self, action):
        ch = Node(self, action)
        self.__children.append(ch)
        return ch

    def is_root(self):
        return not self.__parent

    def is_leaf(self):
        return self.__children == []

    def parent(self):
        return self.__parent

    def children(self):
        return self.__children

    def action(self):
        return self.__action

class TreeSearch:
    def __init__(self, state, exploration_factor):
        self.__initial_state = state
        self.__exploration_factor = exploration_factor
        self.__root = Node(None, None)
        for move in state.legal_moves():
            self.__root.add_child(move)

    def ucb1(self, parent, child):
        if child.simulations != 0:
            n_i = child.simulations
            w_i = child.wins
            c = self.__exploration_factor
            return (w_i/n_i + c * np.sqrt(np.log(parent.simulations)/n_i))
        else:
            return float('inf')

    def iterate(self):
        current_node = self.__root
        current_state = deepcopy(self.__initial_state)

        while not current_node.is_leaf():
            current_node = max(current_node.children(),
                key=lambda n: self.ucb1(current_node, n))
            current_state.play(current_node.action())
        if current_node.simulations != 0:
            for move in current_state.legal_moves():
                current_node.add_child(move)
            if current_node.children():
                current_node = random.choice(current_node.children())
                current_state.play(current_node.action())
        w = self.rollout(current_state)

        while current_node is not None:
            current_node.simulations += 1
            current_node.wins += w
            current_node = current_node.parent()
            w = -w

    # returns action and its weight for children of root
    def policy(self):
        return [(child.action(), child.simulations/self.__root.simulations)
                    for child in self.__root.children()]

    def rollout(self, state):
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
