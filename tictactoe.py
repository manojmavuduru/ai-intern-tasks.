"""
Tic-Tac-Toe AI
==============
Implements an unbeatable Tic-Tac-Toe AI agent using the Minimax
algorithm with Alpha-Beta Pruning. Demonstrates game theory,
adversarial search, and basic heuristics.

Author: <your name>
"""

import math
from typing import List, Optional, Tuple

HUMAN = "X"
AI = "O"
EMPTY = " "


class TicTacToe:
    """Tic-Tac-Toe board representation and rules."""

    def __init__(self):
        self.board: List[str] = [EMPTY] * 9

    def reset(self):
        self.board = [EMPTY] * 9

    def available_moves(self) -> List[int]:
        return [i for i, cell in enumerate(self.board) if cell == EMPTY]

    def make_move(self, position: int, player: str) -> bool:
        if self.board[position] == EMPTY:
            self.board[position] = player
            return True
        return False

    def undo_move(self, position: int):
        self.board[position] = EMPTY

    WIN_LINES = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
        (0, 4, 8), (2, 4, 6),             # diagonals
    ]

    def winner(self) -> Optional[str]:
        for a, b, c in self.WIN_LINES:
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None

    def is_full(self) -> bool:
        return EMPTY not in self.board

    def is_terminal(self) -> bool:
        return self.winner() is not None or self.is_full()

    def print_board(self):
        rows = [self.board[i:i + 3] for i in range(0, 9, 3)]
        print("\n" + "\n---------\n".join(" | ".join(r) for r in rows) + "\n")


class MinimaxAgent:
    """AI agent that picks the optimal move via Minimax with
    Alpha-Beta Pruning. With perfect play from this agent, the AI
    cannot lose (it will win or draw every game).
    """

    def __init__(self, ai_player: str = AI, human_player: str = HUMAN, use_pruning: bool = True):
        self.ai_player = ai_player
        self.human_player = human_player
        self.use_pruning = use_pruning
        self.nodes_evaluated = 0

    def _score(self, game: TicTacToe, depth: int) -> int:
        win = game.winner()
        if win == self.ai_player:
            return 10 - depth
        elif win == self.human_player:
            return depth - 10
        return 0

    def minimax(self, game: TicTacToe, depth: int, maximizing: bool,
                alpha: float = -math.inf, beta: float = math.inf) -> int:
        self.nodes_evaluated += 1

        if game.is_terminal():
            return self._score(game, depth)

        if maximizing:
            best = -math.inf
            for move in game.available_moves():
                game.make_move(move, self.ai_player)
                value = self.minimax(game, depth + 1, False, alpha, beta)
                game.undo_move(move)
                best = max(best, value)
                if self.use_pruning:
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        break  # beta cutoff
            return best
        else:
            best = math.inf
            for move in game.available_moves():
                game.make_move(move, self.human_player)
                value = self.minimax(game, depth + 1, True, alpha, beta)
                game.undo_move(move)
                best = min(best, value)
                if self.use_pruning:
                    beta = min(beta, best)
                    if alpha >= beta:
                        break  # alpha cutoff
            return best

    def best_move(self, game: TicTacToe) -> Tuple[int, int]:
        """Returns (best_move_index, nodes_evaluated)."""
        self.nodes_evaluated = 0
        best_score = -math.inf
        best_move = -1

        for move in game.available_moves():
            game.make_move(move, self.ai_player)
            score = self.minimax(game, 1, False)
            game.undo_move(move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move, self.nodes_evaluated


def play_cli():
    game = TicTacToe()
    agent = MinimaxAgent()

    print("Tic-Tac-Toe — You are X, AI is O.")
    print("Positions are numbered 0-8 left to right, top to bottom:")
    print(" 0 | 1 | 2 \n---------\n 3 | 4 | 5 \n---------\n 6 | 7 | 8 \n")

    game.print_board()

    while not game.is_terminal():
        # Human move
        move = -1
        while move not in game.available_moves():
            try:
                move = int(input(f"Your move {game.available_moves()}: "))
            except ValueError:
                continue
        game.make_move(move, HUMAN)
        game.print_board()

        if game.is_terminal():
            break

        # AI move
        ai_move, nodes = agent.best_move(game)
        game.make_move(ai_move, AI)
        print(f"AI plays at {ai_move} (evaluated {nodes} positions)")
        game.print_board()

    winner = game.winner()
    if winner == HUMAN:
        print("You win! (This shouldn't be possible against a perfect AI...)")
    elif winner == AI:
        print("AI wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    play_cli()
