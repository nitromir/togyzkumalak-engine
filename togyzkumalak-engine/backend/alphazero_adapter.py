import sys
import os
import numpy as np

# Add alpha-zero-general to path
ALPHAZERO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "alpha-zero-general-master"))
sys.path.append(ALPHAZERO_PATH)

try:
    from Game import Game
except ImportError:
    # Fallback if path is different
    print(f"[WARNING] Could not find alpha-zero-general at {ALPHAZERO_PATH}")
    # Create a dummy base class if not found to avoid crash during dev
    class Game:
        pass

from .game_manager import TogyzkumalakBoard

class TogyzkumalakAlphaZeroGame(Game):
    """
    Adapter class for Togyzkumalak to fit the alpha-zero-general interface.
    """
    
    def __init__(self):
        super().__init__()
        self.board_logic = TogyzkumalakBoard()

    def getInitBoard(self):
        """Returns the initial board state."""
        return np.array(self.board_logic.fields)

    def getBoardSize(self):
        """Returns the dimensions of the board (not strictly used by our flat vector)."""
        return (1, 23)  # 18 pits + 2 tuzduks + 2 kazans + 1 current_player

    def getActionSize(self):
        """Returns the number of possible actions (9 pits)."""
        return 9

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board (numpy array)
            player: current player (1 for player1, -1 for player2)
            action: pit index (0-8)
        Returns:
            nextBoard: board after action
            nextPlayer: -player
        """
        # Create a temporary board logic instance
        temp_board = TogyzkumalakBoard(board.tolist())
        
        # Action is 0-8, move is 1-9
        success, _ = temp_board.make_move(action + 1)
        
        return (np.array(temp_board.fields), -player)

    def getValidMoves(self, board, player):
        """
        Returns a binary vector of length 9.
        """
        temp_board = TogyzkumalakBoard(board.tolist())
        valid_moves = temp_board.get_legal_moves()
        
        moves = np.zeros(9)
        for m in valid_moves:
            moves[m] = 1
        return moves

    def getGameEnded(self, board, player):
        """
        Returns:
            0 if not ended, 1 if player won, -1 if player lost, small non-zero for draw.
        """
        temp_board = TogyzkumalakBoard(board.tolist())
        if not temp_board.is_finished:
            return 0
        
        winner = temp_board.winner
        if winner == "draw":
            return 1e-4
        
        # Map winner to current player perspective
        # player 1 is white (fields[22]=0), player -1 is black (fields[22]=1)
        if player == 1:
            return 1 if winner == "white" else -1
        else:
            return 1 if winner == "black" else -1

    def getCanonicalForm(self, board, player):
        """
        Returns the board in a player-independent format.
        If player is -1 (Black), we swap sides.
        """
        if player == 1:
            return board
        
        # Player is -1 (Black), swap everything
        new_board = np.copy(board)
        
        # Swap pits [0-8] with [9-17]
        white_pits = board[0:9]
        black_pits = board[9:18]
        new_board[0:9] = black_pits
        new_board[9:18] = white_pits
        
        # Swap tuzduk info [18] with [19]
        white_tuzduk = board[18]
        black_tuzduk = board[19]
        new_board[18] = black_tuzduk
        new_board[19] = white_tuzduk
        
        # Swap kazans [20] with [21]
        white_kazan = board[20]
        black_kazan = board[21]
        new_board[20] = black_kazan
        new_board[21] = white_kazan
        
        # Swap current player [22]
        new_board[22] = 1 - board[22]
        
        return new_board

    def getSymmetries(self, board, pi):
        """
        Togyzkumalak doesn't have simple rotational/flip symmetries 
        like Othello or TicTacToe due to the board structure.
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """Quick conversion to string for MCTS hashing."""
        return board.tobytes()

    def get_observation(self, board):
        """Helper to convert raw fields to NN input observation (128-dim)."""
        temp_board = TogyzkumalakBoard(board.tolist())
        return temp_board.to_observation()
