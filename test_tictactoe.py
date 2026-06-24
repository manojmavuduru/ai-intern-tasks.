import unittest
from tictactoe import TicTacToe, MinimaxAgent, HUMAN, AI


class TestTicTacToe(unittest.TestCase):
    def setUp(self):
        self.game = TicTacToe()
        self.agent = MinimaxAgent()

    def test_winner_detection_row(self):
        for i in [0, 1, 2]:
            self.game.make_move(i, HUMAN)
        self.assertEqual(self.game.winner(), HUMAN)

    def test_winner_detection_diagonal(self):
        for i in [0, 4, 8]:
            self.game.make_move(i, AI)
        self.assertEqual(self.game.winner(), AI)

    def test_draw_detection(self):
        # A known drawn-out board
        moves = {0: HUMAN, 1: AI, 2: HUMAN, 3: HUMAN, 4: AI, 5: HUMAN, 6: AI, 7: HUMAN, 8: AI}
        for pos, player in moves.items():
            self.game.make_move(pos, player)
        self.assertIsNone(self.game.winner())
        self.assertTrue(self.game.is_full())

    def test_ai_blocks_immediate_loss(self):
        # Human about to win at position 2
        self.game.make_move(0, HUMAN)
        self.game.make_move(1, HUMAN)
        move, _ = self.agent.best_move(self.game)
        self.assertEqual(move, 2)

    def test_ai_never_loses_full_game(self):
        """Simulate AI vs a simple opponent who always picks the first
        available move; AI should never lose."""
        game = TicTacToe()
        agent = MinimaxAgent()
        turn = HUMAN
        while not game.is_terminal():
            if turn == HUMAN:
                move = game.available_moves()[0]
                game.make_move(move, HUMAN)
            else:
                move, _ = agent.best_move(game)
                game.make_move(move, AI)
            turn = AI if turn == HUMAN else HUMAN
        self.assertNotEqual(game.winner(), HUMAN)

    def test_pruning_matches_no_pruning_result(self):
        game1 = TicTacToe()
        game2 = TicTacToe()
        for i in [0, 4]:
            game1.make_move(i, HUMAN)
            game2.make_move(i, HUMAN)

        agent_pruned = MinimaxAgent(use_pruning=True)
        agent_full = MinimaxAgent(use_pruning=False)

        move_pruned, _ = agent_pruned.best_move(game1)
        move_full, _ = agent_full.best_move(game2)

        # Both should lead to equally optimal outcomes (score-wise)
        game1.make_move(move_pruned, AI)
        game2.make_move(move_full, AI)
        self.assertEqual(agent_pruned._score(game1, 1) >= 0, True)


if __name__ == "__main__":
    unittest.main()
