import numpy as np
import os
from pieces import Board, Piece, Shape


class Puzzle:


    def __init__(self, board, pieces):
        self.board = board
        self.pieces = pieces
        self.n_pieces = len(self.pieces)
        self.piece_indexes = list(range(self.n_pieces))
        self.solutions = []

        self.check_solvable()

        # Some interesting statistics
        self.n_placements = 0
        self.n_valid_placements = 0
        self.n_solutions = 0


    def check_solvable(self):

        """
        Do some checks that the puzzle is possible to solve
        """

        # Check that number of squares on board adds to total squares in combined pieces
        board_n_squares = self.board.occupancy.size - self.board.occupancy.sum()
        pieces_n_squares = [p.occupancy.sum() for p in self.pieces]
        
        if board_n_squares == sum(pieces_n_squares):
            print(f'\nCheck passed: no. squares on board matches total no. squares in pieces ({board_n_squares})')
        else:
            raise RuntimeError(f'Puzzle not valid: no. squares on board is {board_n_squares}, pieces only add to {sum(pieces_n_squares)}')

        # Check that total numbers of in and out connectors are the same
        total_in, total_out = 0, 0
        for shape in [self.board] + self.pieces:
            n_in, n_out = shape.n_connectors()
            total_in += n_in
            total_out += n_out

        if total_in == total_out:
            print(f'Check passed: total in connectors same as total out ({total_in})')
        else:
            raise RuntimeError(f'Puzzle not valid: total in connectors is {total_in}, total out is {total_out}')


    @staticmethod
    def is_valid(board_stack):

        # Does it place squares on top of each other?
        occupancy_conflicts = np.any(board_stack[:, 0] > 1, axis=(1, 2))
        
        # Does it have conflicting N/S connectors?
        ns_interactions = np.roll(board_stack[:, 1], 1, axis=1) + board_stack[:, 3]
        ns_conflicts = np.any(np.abs(ns_interactions) > 1, axis=(1, 2))

        # Does it have conflicting E/W connectors?
        ew_interactions = np.roll(board_stack[:, 4], 1, axis=2) + board_stack[:, 2]
        ew_conflicts = np.any(np.abs(ew_interactions) > 1, axis=(1, 2))

        # If none of the above, valid!
        any_conflict = np.any(np.vstack([occupancy_conflicts, ns_conflicts, ew_conflicts]), axis=0)
                
        return np.logical_not(any_conflict)


    def solve(self, board=None, history={}, start_position=None):

        if board is None:
            board = self.board

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
        for piece_index in self.piece_indexes:

            if piece_index not in history:

                piece = self.pieces[piece_index]

                board_stack = board.place_pieces(piece, i, j, conn_type, k, layer=None)

                # Determine which new boards are valid
                valid = self.is_valid(board_stack)

                # Is puzzle complete?
                puzzle_complete = len(history) + 1 == self.n_pieces

                for layer, (new_board, v) in enumerate(zip(board_stack, valid)):
                    
                    # TODO: edit Board so we don't have to re-initialise
                    new_board = Board(new_board)
                    self.n_placements += 1

                    if self.n_placements % 1000 == 0:
                        print('Attempted piece placements: ', self.n_placements)

                    if v:
                        new_history = {**history, piece_index: (i, j, conn_type, k, layer)}

                        self.n_valid_placements += 1

                        if puzzle_complete:
                            self.solutions.append(new_history)
                            self.n_solutions += 1
                        else:
                            self.solve(new_board, new_history)


    def visualize_solutions(self, savefolder=None):

        for s_index, solution in enumerate(self.solutions, 1):
            
            solution_board = Board(self.board.copy())
            display_representation = solution_board.make_display_representation()
            
            solution_representation = np.zeros(display_representation.shape, dtype='uint8')

            for piece_index, placement in solution.items():

                piece = self.pieces[piece_index]

                solution_board = solution_board.place_pieces(piece, *placement)
                new_display_representation = solution_board.make_display_representation()
                
                difference = new_display_representation - display_representation
                display_representation = new_display_representation

                solution_representation = solution_representation + (piece_index + 2)*difference                

            solution_title = f'Solution {s_index} of {len(self.solutions)}'
            savefile = os.path.join(savefolder, solution_title.replace(' ', '_') + '.png')
            Shape.plot_display_representation(solution_representation, solution_title, savefile)


    def print_statistics(self):
        stats = f'''\n
        Total piece placements: {self.n_placements}\n
        Total valid piece placements: {self.n_valid_placements}\n
        Solutions: {self.n_solutions}\n
        '''
        print(stats)


if __name__ == '__main__':

    import sys

    # Enter folder as CLI arg
    folder = sys.argv[1]

    board = Board.from_csvs(os.path.join(folder, 'board'))
    pieces = [Piece.from_csvs(os.path.join(folder, f)) for f in os.listdir(folder) if 'piece' in f]

    puzzle = Puzzle(board, pieces)

    puzzle.solve(start_position=(8, 8))
    puzzle.print_statistics()
    puzzle.visualize_solutions(savefolder=folder)

