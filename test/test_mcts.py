import unittest
import numpy as np
from mcts import Node, TreeSearch
from game import GameState, Player
from models import RandomModel


class TestGame:
    def __init__(self):
        self.__choice = None

    def turn(self):
        if self.__choice:
            return Player.O
        else:
            return Player.X

    def legal_moves(self):
        if self.__choice:
            return []
        else:
            return ["win", "lose"]

    def state(self):
        if self.__choice:
            return GameState.Win
        else:
            return GameState.Continue

    def winner(self):
        if self.__choice == "win":
            return Player.X
        elif self.__choice == "lose":
            return Player.O
        else:
            return None

    def play(self, move):
        self.__choice = move

    def to_tensor(self):
        value = 0
        if self.__choice is None:
            value = 0
        elif self.__choice == "win":
            value = 1
        else:
            value = -1
        return np.array([[[value]]])

    def policy_to_tensor(self, policy):
        return np.array([[[policy["win"], policy["lose"]]]])

class MCTSTests(unittest.TestCase):
    def test_single_node_tree(self):
        n = Node(None, None)
        self.assertTrue(n.is_root())
        self.assertTrue(n.is_leaf())

    def test_two_node_tree(self):
        root = Node(None, None)
        child = root.add_child("move")
        self.assertTrue(root.is_root())
        self.assertFalse(root.is_leaf())
        self.assertFalse(child.is_root())
        self.assertTrue(child.is_leaf())
        self.assertIs(child.parent(), root)
        self.assertEqual(child.action(), "move")

    def test_mcts_two_choice_game(self):
        test_game = TestGame()
        model = RandomModel()
        ts = TreeSearch(test_game, model, 2)
        for i in range(200):
            ts.iterate()
        policy = dict(ts.policy())
        self.assertGreater(policy["win"], 0.9)
        self.assertLess(policy["lose"], 0.1)
        self.assertAlmostEqual(policy["win"] + policy["lose"], 1.0)
