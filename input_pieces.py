from argparse import ArgumentParser
from pathlib import Path
from pieces import Piece
import pandas as pd
import numpy as np


def split_row(row):
    no_spaces = row.replace(" ", "")
    integer_list = [int(i) for i in no_spaces.split(",")]
    return integer_list


def enter_piece():
    
    occupancy_done = False
    row_idx = 0
    rows = []

    print("\n\nNew piece:\n")
    while not occupancy_done:
        row = input(f"Enter row {row_idx}, hit enter to move on, or 'exit' -->  ")
        if row == "":
            occupancy_done = True
        elif row == 'exit':
            exit()
        else:
            rows.append(split_row(row))
            row_idx += 1

    occupancy = np.array(rows)

    print("\nOccupancy done:")
    print(occupancy)

    connectors = input("\nEnter connectors -->  ")

    connectors = split_row(connectors)

    print("\nConnectors done:")
    print(connectors)

    return occupancy, connectors


def check_piece(occupancy, connectors, all_rotations=True):
    piece = Piece(occupancy, connectors)
    for rot in piece.rotations:
        print(rot.occupancy)
        rot.visualize_rotation(colour_index=1, pad=2)
        if not all_rotations:
            break


def save_piece(occupancy, connectors, folder, piece_idx):
    piece_folder = f"{folder}/pieces/piece{piece_idx}/"
    Path(piece_folder).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(occupancy).to_csv(piece_folder + 'occupancy.csv', header=None, index=None)
    pd.DataFrame(connectors).T.to_csv(piece_folder + 'connectors.csv', header=None, index=None)
    print(f"\nPiece saved to {piece_folder}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('folder', type=str, help="Folder where pieces to be stored")
    parser.add_argument('--start_piece', '-start', type=int, default=0, help="Index of start piece, if entry started already")
    args = parser.parse_args()


    print("\nLet's enter some pieces!")

    print("\n\nCreating main folder")
    Path(args.folder).mkdir(parents=True, exist_ok=True)

    piece_index = args.start_piece

    while True:

        occupancy, connectors = enter_piece()

        check_all_rotations = input("Check all rotations -->  ").lower().startswith("y")

        check_piece(occupancy, connectors, all_rotations=check_all_rotations)

        happy = input("Happy? -->  ")

        if happy.lower().startswith("y"):
            save_piece(occupancy, connectors, "puzzles/3_by_3", piece_index)
            piece_index += 1
        else:
            print(f"\nLet's try piece {piece_index} again...")