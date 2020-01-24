import unittest
from mcts import node

class MCTSTests(unittest.TestCase):

    def test_single_node_tree(self):
        n = node(None, None)
        self.assertTrue(n.is_root())
        self.assertTrue(n.is_leaf())

    def test_two_node_tree(self):
        root = node(None, None)
        child = root.add_child("move")
        self.assertTrue(root.is_root())
        self.assertFalse(root.is_leaf())
        self.assertFalse(child.is_root())
        self.assertTrue(child.is_leaf())
        self.assertIs(child.parent(), root)
        self.assertEqual(child.action(), "move")
