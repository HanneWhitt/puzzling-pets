import numpy as np
from pandas import read_csv
import os
from matplotlib import pyplot as plt


class Shape(np.ndarray):

    """
    Describes a specific rotation of a piece, or the board.         
    """

    def __new__(cls, representation):  
        return np.asarray(representation).view(cls)


    # def __init__(self) -> None:
    #     super().__init__()

    #     self.representation = representation
    #     self.I = representation.shape[1]
    #     self.J = representation.shape[2]
    #     self.length = max(self.I, self.J)


    def occupancy(self):
        return self.representation[0]


    def connectors(self):
        return self[1:]


    @staticmethod
    def get_rotation(representation, k):

        # Rotate representation
        rot = np.rot90(representation.copy(), k, axes=(-2, -1))
        
        # Re-order connectors: N becomes W, etc.
        if rot.ndim == 3:
            rot[1:] = np.roll(rot[1:], -k, axis=0)
        else:
            rot[:, 1:, :, :] = np.roll(rot[:, 1:, :, :], -k, axis=1)

        return rot


    @classmethod
    def from_csvs(cls, folder):
        occupancy_file = os.path.join(folder, 'occupancy.csv')
        occupancy = read_csv(occupancy_file, header=None).to_numpy(dtype=int)
        connectors_file = os.path.join(folder, 'connectors.csv')
        connectors_raw = read_csv(connectors_file, header=None).T.squeeze().to_list()
        occupancy, connectors = cls.unwrap_connectors(occupancy, connectors_raw)
        representation = np.stack([occupancy, *connectors])
        return cls(representation)


    @classmethod
    def unwrap_connectors(cls, occupancy, connectors_raw):

        '''
        Connectors are saved as a list of 1/0 (in/out) values for ease of entry. 

        Unwrap them into four matrices, each describing connectors facing a single direction.                
        '''

        # Current occupied square (square on edge of piece),
        # initialised at the first occupied square (reading row by row) with a zero above
        start_pos = np.array(np.where((occupancy == 1) & (np.roll(occupancy, 1, axis=0) == 0)))[:, 0]
        occ = start_pos

        # Initialise direction of normal to current edge as along a northern edge
        normal = np.array([0, 1])

        # Initialise connectors tensor
        connectors = np.zeros((4, *occupancy.shape), dtype=int)

        # Initialise layer indexes for in and out connectors
        out_layer_idx = 0

        # Iterate over the list of connectors
        for conn in connectors_raw:

            # Current edge direction is at 90 degrees to current normal (we rotate clockwise around piece)
            edge = np.array([normal[1], -normal[0]])

            # Get coordinates of positions relevant to determining next occupied square and normal vector
            # Note, negative edge vec is equivalent to normal in matrix indexing.
            opp = occ - edge
            adj = occ + normal
            opp_adj = adj - edge

            # Allocate current connector.
            out_layer_idx = out_layer_idx % 4
            in_layer_idx = (out_layer_idx + 2) % 4
            if conn == 1:
                connectors[out_layer_idx][tuple(opp)] = 1
            elif conn == 0:
                connectors[in_layer_idx][tuple(occ)] = -1
            else:
                raise ValueError(f"List elements of connectors_raw arg should be 1 or 0: found value '{conn}'")

            # Check that square outside current edge is unoccupied (otherwise code broken...)
            assert occupancy[tuple(opp)] == 0, "Occupied square should be unoccupied"

            # ...and that square inside current edge is occupied (likewise)
            assert occupancy[tuple(occ)] == 1, "Unoccupied square should be occupied"

            # Determine new occ and normal direction
            adj_value = occupancy[tuple(adj)]
            if adj_value == 0:
                # We rotate clockwise and deal with another edge of the same square (occ unchanged)
                normal = edge
                out_layer_idx += 1
            else:
                opp_adj_value = occupancy[tuple(opp_adj)]
                if opp_adj_value == 0:
                    # We continue along the edge to another, parellel edge section (normal, layer indexes unchanged)
                    occ = adj
                else:
                    # We rotate anticlockwise, occ and normal both change
                    occ = opp_adj
                    normal = -edge
                    out_layer_idx -= 1

        # Verify that we terminate back in the starting position
        assert np.array_equal(occ, start_pos), "Unwrapping did not reach original position!"

        # Verify that we terminate with the original normal vector
        assert np.array_equal(normal, np.array([0, 1])), "Unwraping terminated in correct position, but with normal vector in wrong direction"

        return occupancy, connectors


    def visualize_shape(self, title=None, savefile=None):

        """
        Show a plot of this rotation of this piece
        """

        display_rep = self.make_display_representation()
        RGB = self.make_RGB(display_rep)
        plt.imshow(RGB)
        plt.title(title)

        if savefile:
            plt.savefig(savefile)
            print(f'Saved figure to {savefile}')
        else:
            plt.show()


    def make_display_representation(self):

        """
        Create an enlarged version of the occupancy matrix with representation of the connectors,
        ready to visualize
        """

        # Extend each square into a 5*5 block of repeats to facilitate adding connector representation
        display_rep = np.kron(self.occupancy(), np.ones((5,5), dtype=int))

        # Add connectors into correct positions       
        for mat, adjust in zip(self.connectors(), [(-1, 2), (2, 0), (0, 2), (2, -1)]):
            base = np.zeros((5, 5), dtype=int)
            base[adjust] = 1
            display_rep += np.kron(mat, base)

        return display_rep


    @staticmethod
    def make_RGB(matrix):

        # Convert a display representation into RGB values so it can be shown

        palette = np.array([
            [255, 255, 255],   # white
            [  0,   0,   0],   # black
            [255,   0,   0],   # red
            [  0, 255,   0],   # green
            [  0,   0, 255]   # blue
        ])

        # Assign colours
        RGB = palette[matrix]

        return RGB


    def n_connectors(self):
        values, counts = np.unique(self.connectors(), return_counts=True)
        n_in, _, n_out = counts
        return n_in, n_out


class Piece(Shape):

    """
    Describes a puzzle piece including all of its 4 rotations, which are generated upon instantiation.

    Unwraps connectors, given as a list read clockwise around the edge of the piece, into N, E, S, W connector matrices
    """


    def __init__(self, representation):
        
        super().__init__(representation)

        # Generate all 4 rotations
        self.rotations = [Shape(self.get_rotation(representation, k)) for k in range(4)]

        # Build stacks
        self.square_size = 2*self.length - 3
        self.square_centre = self.length - 2

        self.stacks = {}

        n_in, n_out = self.n_connectors()

        for connector_value, n_lyrs in [(-1, n_in), (1, n_out)]:

            # Create a single tensor to contain full representations of all possible orientations of the piece
            stack = np.zeros((n_lyrs, 5, self.square_size, self.square_size), dtype=int)

            # Enumerate all rotations and build stack
            lyr = 0
            for rotation in self.rotations:

                # Select top row of each rotation, northward facing connectors only
                row_idxs, col_idxs = np.where(rotation.connectors()[connector_value - 1] == connector_value)
                if connector_value == 1:
                    row_idxs += 1

                i_mins = self.length - 2 - row_idxs
                j_mins = self.length - 2 - col_idxs
                i_maxs, j_maxs = i_mins + rotation.I, j_mins + rotation.J

                for imin, imax, jmin, jmax in zip(i_mins, i_maxs, j_mins, j_maxs):
                    stack[lyr, :, imin:imax, jmin:jmax] = rotation.representation
                    lyr += 1

            # if connector_value == -1:
            #     stack = self.get_rotation(stack, 2)

            # We pre-compute all the rotations of the stack to save time later
            self.stacks[connector_value] = [self.get_rotation(stack, k) for k in range(4)]
            

    @classmethod
    def unwrap_connectors(cls, occupancy, connectors):

        # Add padding with zeroes
        occupancy = np.pad(occupancy, 1)

        # Apply parent function
        occupancy, connectors = super().unwrap_connectors(occupancy, connectors)

        return occupancy, connectors


class Board(Shape):


    # def __init__(
    #         self,
    #         representation,
    #         history=None
    # ):
        
    #     super().__init__(representation)
    #     self.history = history


    def find_most_restricted(self, tiebreaker_steps=5):
        
        """
        Find the unoccupied square that has most occupied NESW neighbours
        """

        # Number of occupied neighbours for all unoccupied positions
        tiebreaker = self.sum_occupied_neighbours(self.occupancy(), self.occupancy())
        candidates = (tiebreaker == np.max(tiebreaker))

        # Break ties by looking at N occupied neighbours for unoccupied neighbours of remaining candidates
        # Finite number of tiebreaking rounds avoids infinite loop
        for _ in range(tiebreaker_steps):
            n_candidates = np.count_nonzero(candidates)
            if n_candidates == 1:
                break
            elif n_candidates > 1:
                tiebreaker = self.sum_occupied_neighbours(tiebreaker, self.occupancy())
                candidate_scores = tiebreaker*candidates
                candidates = (candidate_scores == np.max(candidate_scores))
            else:
                raise RuntimeError('Tiebreaker eliminated all candidates')

        # After all tiebreaker rounds, take indices of first candidate
        candidate_indices = np.nonzero(candidates)

        return candidate_indices[0][0], candidate_indices[1][0]


    @staticmethod
    def sum_occupied_neighbours(starting_squares, occupancy):

        """
        Sum NESW neighbours for all unoccupied positions in an np.array
        """

        pad_occ = np.pad(starting_squares, 1)
        n_neighbours = pad_occ[:-2, 1:-1] + pad_occ[2:, 1:-1] + pad_occ[1:-1, :-2] + pad_occ[1:-1, 2:]
        return n_neighbours*(1 - occupancy)


    def get_any_connector(self, i, j):

        """
        Return any connector facing unnoccupied square i, j        
        """

        out_connectors = self.connectors()[[0, 3, 2, 1], i, j]
        for k, value in enumerate(out_connectors):
            if value == 1:
                return 1, k

        in_connectors = self.connectors()[[2, 1, 0, 3], [i+1, i, i-1, i], [j, j+1, j, j-1]]

        for k, value in enumerate(in_connectors):
            if value == -1:
                return -1, k

        raise RuntimeError('get_any_connector() terminated without finding a connector')
    
    

if __name__ == "__main__":

    folder = 'puzzles/3_by_3/'


    board = Board.from_csvs(f'{folder}board')

    # board.visualize_shape()

    # board.visualize_shape('3 by 3 test', folder + 'board_figure.png')

    i = 3
    j = 3

    conn_type, k = board.get_any_connector(i, j)
    #print(conn_type, k)


    # # piece.visualize_shape()

    # # in_stack = piece.stacks[-1]

    # # for lyr in out_stack:
    # #     p = Shape(lyr)
    # #     p.visualize_shape()

    # # for r in range(4):
    # #     for lyr in in_stack[r]:
    # #         p = Shape(lyr)
    # #         p.visualize_shape(f'R = {r}')
        


    # #i, j = board.find_most_restricted()







    piece = Piece.from_csvs(f'{folder}piece0')

    # for r in piece.rotations:
    #     r.visualize_shape()


    # print(conn_type, k)

    # Get the opposite type of connector, facing the opposite way
    conn_type = -conn_type
    k = (k+2)%4
    

    rep = piece.stacks[conn_type][k][-1, :, :,]
    # for rep in piece_reps:

    print(board.I, board.J)
    print(piece.square_size)

    # print(rep)
    print()

    # Shape(rep).visualize_shape()

    


