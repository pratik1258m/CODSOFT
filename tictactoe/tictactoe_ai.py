#!/usr/bin/env python3
"""
Tic-Tac-Toe AI
CODSOFT AI Internship - Task 2

This script implements an unbeatable AI player for Tic-Tac-Toe using the Minimax 
Search Algorithm with Alpha-Beta Pruning. It provides an interactive CLI game loop.
"""

import sys


class TicTacToe:
    def __init__(self):
        # Board representation: 9-element list representing 3x3 grid
        # Indices:
        #  0 | 1 | 2
        # -----------
        #  3 | 4 | 5
        # -----------
        #  6 | 7 | 8
        self.board = [" "] * 9
        self.human = "X"
        self.ai = "O"

    def print_board(self):
        """Prints the current state of the board."""
        print()
        for row in range(3):
            print(f" {self.board[row*3]} | {self.board[row*3 + 1]} | {self.board[row*3 + 2]} ")
            if row < 2:
                print("-----------")
        print()

    def available_moves(self):
        """Returns a list of empty indices on the board."""
        return [i for i, spot in enumerate(self.board) if spot == " "]

    def make_move(self, index, player):
        """Places a mark on the board for the specified player."""
        if self.board[index] == " ":
            self.board[index] = player
            return True
        return False

    def undo_move(self, index):
        """Removes a mark from the board."""
        self.board[index] = " "

    def check_winner(self, player):
        """Checks if the specified player has won the game."""
        # Winning combinations: rows, columns, and diagonals
        win_states = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for combo in win_states:
            if all(self.board[i] == player for i in combo):
                return True
        return False

    def is_board_full(self):
        """Returns True if the board is full (no spaces left)."""
        return " " not in self.board

    def is_game_over(self):
        """Checks if the game has ended by a win or a draw."""
        return self.check_winner(self.human) or self.check_winner(self.ai) or self.is_board_full()

    def evaluate_board(self, depth):
        """Evaluates the board state. Returns positive score for AI win, negative for Human win."""
        if self.check_winner(self.ai):
            return 10 - depth  # Subtract depth to favor faster wins
        elif self.check_winner(self.human):
            return depth - 10  # Add depth to favor slower losses
        return 0

    def minimax(self, depth, is_maximizing, alpha, beta):
        """
        The Minimax Algorithm with Alpha-Beta Pruning.
        Recursively searches the game-tree to evaluate moves.
        """
        # Base cases: game over or leaf node
        if self.is_game_over():
            return self.evaluate_board(depth), None

        best_move = None

        if is_maximizing:
            # AI's turn (maximize score)
            max_eval = -sys.maxsize
            for move in self.available_moves():
                self.make_move(move, self.ai)
                evaluation, _ = self.minimax(depth + 1, False, alpha, beta)
                self.undo_move(move)

                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move

                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval, best_move

        else:
            # Human's turn (minimize score)
            min_eval = sys.maxsize
            for move in self.available_moves():
                self.make_move(move, self.human)
                evaluation, _ = self.minimax(depth + 1, True, alpha, beta)
                self.undo_move(move)

                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move

                beta = min(beta, evaluation)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval, best_move


def main():
    print("=" * 60)
    print("        ❌ WELCOME TO UNBEATABLE TIC-TAC-TOE AI ⭕")
    print("             Created for the CODSOFT AI Internship")
    print("=" * 60)
    print("You are playing as 'X'. The AI plays as 'O'.")
    print("Board spots are numbered 1 to 9 from top-left to bottom-right:")
    print(" 1 | 2 | 3 ")
    print("-----------")
    print(" 4 | 5 | 6 ")
    print("-----------")
    print(" 7 | 8 | 9 \n")

    game = TicTacToe()
    
    # Prompt user for who starts the game
    first_turn = ""
    while first_turn not in ["y", "n"]:
        first_turn = input("Do you want to play first? (y/n): ").strip().lower()

    if first_turn == "n":
        print("\nAI is calculating its first move...")
        _, ai_move = game.minimax(0, True, -sys.maxsize, sys.maxsize)
        if ai_move is not None:
            game.make_move(ai_move, game.ai)
        game.print_board()
    else:
        game.print_board()

    while not game.is_game_over():
        # --- HUMAN TURN ---
        valid_move = False
        while not valid_move:
            try:
                move_str = input("Enter your move (1-9): ").strip()
                move_idx = int(move_str) - 1
                if move_idx < 0 or move_idx > 8:
                    print("Out of bounds! Choose a spot between 1 and 9.")
                    continue
                if game.board[move_idx] != " ":
                    print("Spot already taken! Choose an empty spot.")
                    continue
                valid_move = True
            except ValueError:
                print("Invalid input! Please enter a number between 1 and 9.")

        game.make_move(move_idx, game.human)
        game.print_board()

        if game.is_game_over():
            break

        # --- AI TURN ---
        print("AI is thinking...")
        # AI maximizes, hence is_maximizing = True
        _, ai_move = game.minimax(0, True, -sys.maxsize, sys.maxsize)
        
        if ai_move is not None:
            game.make_move(ai_move, game.ai)
            print(f"AI chose spot {ai_move + 1}")
        
        game.print_board()

    # --- GAME OVER ---
    if game.check_winner(game.human):
        print("🎉 Congratulations! You somehow beat the unbeatable AI! (This shouldn't happen!)")
    elif game.check_winner(game.ai):
        print("🤖 The AI wins! Better luck next time!")
    else:
        print("🤝 It's a draw! Well played!")


if __name__ == "__main__":
    main()
