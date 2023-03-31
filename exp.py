from pieces import Board
import pandas as pd


occupancy_file = 'puzzles/priority_test/board/occupancy.csv'
occupancy = pd.read_csv(occupancy_file, header=None).to_numpy()


board = Board(occupancy, None, None, None, None)

i, j = board.find_most_restricted()

print(i, j)