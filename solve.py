import sys
import os
import numpy as np
from pieces import Board, Piece


def check_solvable(board, pieces):

    """
    Do some checks that puzzle is possible to solve
    """

    # Check that number of squares on board adds to total squares in combined pieces

    # Check that total numbers of in and out connectors are the same

    pass


def solve(board, pieces, start_position=None):

    if start_position is None:
        # Find the unoccupied board spot to fill next. 
        # This is selected as the most restricted by occupied neighbours
        i, j = board.find_most_restricted()
    else:
        i, j = start_position

    # Get any type and direction of a connector facing unnoccupied space i, j 
    conn_type, k = board.get_any_connector(i, j)

    # Find complementary type and rotation
    conn_type = -conn_type
    k = (k + 2) % 4

    # Try all pieces
    for piece in pieces:

        # Use stack to simultaneously try all orientations of piece exposing complementary connector
        piece_stack = piece.stacks[conn_type][k]
    
        # Create board stack of matching dimension
        board_stack = np.broadcast_to(board.representation, 

        board.representation + 


if __name__ == '__main__':

    # Enter folder as CLI arg
    folder = sys.argv[1]

    board = Board.from_csvs(f'{folder}board')
    pieces = [Piece.from_csvs(f'{folder}piece{i}') for i in range(3)]

    check_solvable(board, pieces)

    solution = solve(board, pieces)

