import unittest

from tictactoe import TicTacToe
from game import GameState, Player

class TicTacToeTests(unittest.TestCase):
    def test_empty_board(self):
        board = TicTacToe()
        self.assertEqual(board.state(), GameState.Continue)
        self.assertEqual(board.turn(), Player.X)
        self.assertCountEqual(board.legal_moves(),
            [(i,j) for i in range(3) for j in range(3)])
        self.assertIs(board.winner(), None)

    def test_some_game(self):
        board = TicTacToe()
        board.play((0,0))
        board.play((1,1))
        board.play((2,0))
        self.assertEqual(board.state(), GameState.Continue)
        self.assertEqual(board.turn(), Player.O)
        self.assertCountEqual(board.legal_moves(),
            [(0,1), (0,2), (1,0), (1, 2), (2, 1), (2, 2)])
        self.assertIs(board.winner(), None)

    def test_tie_game(self):
        board = TicTacToe()
        moves = [(1, 1), (2, 2), (1, 2), (1, 0), (2, 0),
                    (0, 2), (0, 1), (2, 1), (0, 0)]
        for move in moves:
            board.play(move)

        self.assertEqual(board.state(), GameState.Tie)
        self.assertEqual(board.legal_moves(), [])
        self.assertIs(board.winner(), None)


    def test_x_winner_game(self):
        board = TicTacToe()
        moves = [(2, 0), (1, 0), (2, 1), (0, 0), (2, 2)]

        for move in moves:
            board.play(move)

        self.assertEqual(board.state(), GameState.Win)
        self.assertEqual(board.legal_moves(), [])
        self.assertIs(board.winner(), Player.X)

    def test_o_winner_game(self):
        board = TicTacToe()
        moves = [(0, 2), (2, 0), (1, 0), (2, 1), (0, 0), (2, 2)]

        for move in moves:
            board.play(move)

        self.assertEqual(board.state(), GameState.Win)
        self.assertEqual(board.legal_moves(), [])
        self.assertIs(board.winner(), Player.O)
