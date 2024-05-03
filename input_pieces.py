from argparse import ArgumentParser
from pathlib import Path
from pieces import Board, Piece
import pandas as pd
import numpy as np
import os


def split_row(row):
    no_spaces = row.replace(" ", "")
    integer_list = [int(i) for i in no_spaces.split(",")]
    return integer_list


def enter_shape():
    
    occupancy_done = False
    row_idx = 0
    rows = []

    print("\n\nNew entry:\n")
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


def save_shape(occupancy, connectors, folder):
    Path(folder).mkdir(parents=True, exist_ok=True)
    occ_file = os.path.join(folder, 'occupancy.csv')
    pd.DataFrame(occupancy).to_csv(occ_file, header=None, index=None)
    conn_file = os.path.join(folder, 'connectors.csv')
    pd.DataFrame(connectors).T.to_csv(conn_file, header=None, index=None)
    print(f"\nPiece saved to {folder}")


def check_shape(folder, is_board=False, all_rotations=True):
    
    if is_board:
        board = Board.from_csvs(folder)
        board.visualize_shape()

    else:
        piece = Piece.from_csvs(folder)
        for rot in piece.rotations:
            print(rot.occupancy)
            rot.visualize_shape()
            if not all_rotations:
                break



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('folder', type=str, help="Folder where pieces to be stored")
    parser.add_argument('--start_piece', '-start', type=int, default=0, help="Index of start piece, if entry started already")
    args = parser.parse_args()


    print("\nLet's enter some pieces!")

    print("\n\nCreating main folder")
    Path(args.folder).mkdir(parents=True, exist_ok=True)



    # Enter board
    enter_board = input('Enter board?').lower().startswith("y")
    
    if enter_board:
        occupancy, connectors = enter_shape()
        board_folder = os.path.join(args.folder, 'board')
        save_shape(occupancy, connectors, board_folder)
        check_shape(board_folder, is_board=True)
    
    # Enter pieces
    piece_index = args.start_piece

    while True:

        occupancy, connectors = enter_shape()
        check_all_rotations = input("Check all rotations -->  ").lower().startswith("y")
        piece_folder = os.path.join(args.folder, f'piece{piece_index}')
        save_shape(occupancy, connectors, piece_folder)
        check_shape(piece_folder, is_board=False, all_rotations=check_all_rotations)

        happy = input("Happy? -->  ")
        if happy.lower().startswith("y"):
            piece_index += 1
        else:
            print(f"\nLet's try piece {piece_index} again...")