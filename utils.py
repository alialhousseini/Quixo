'''
A script containing utility functions for the project.
'''
from game import Move, Game
from typing import Tuple
import numpy as np


def decode_action(action: int) -> Tuple[Tuple[int, int], Move]:
    '''
    Converts the action from the agent to the action for the environment.
    Explanation:
    - An action selected by the Agent is a number ranged from 0 to 43.
    - This number has to be converted to pos, Move.
    - Where pos is a tuple (X,Y) position of the cell to move.
    - Move is a Move enum value. (Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT)

    Returns a tuple (pos, Move)
    '''

    if action < 0 or action >= 44:
        raise ValueError("Invalid action number. Must be between 0 and 43.")

    # Mapping for corner actions
    corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
    corner_directions = [
        [Move.RIGHT, Move.BOTTOM],  # Top Left Corner
        [Move.LEFT, Move.BOTTOM],   # Top Right Corner
        [Move.RIGHT, Move.TOP],     # Bottom Left Corner
        [Move.LEFT, Move.TOP]       # Bottom Right Corner
    ]
    if action < 8:  # First 8 actions are corner actions
        corner_index = action // 2
        move_index = action % 2
        return corners[corner_index], corner_directions[corner_index][move_index]

    # Mapping for other periphery actions
    action -= 8  # Adjust action number for the periphery mapping
    periphery = [
        (0, i) for i in range(1, 4)  # Top row (excluding corners)
    ] + [
        (i, 4) for i in range(1, 4)  # Right column (excluding corners)
    ] + [
        (i, 0) for i in range(1, 4)  # Left column (excluding corners)
    ] + [
        (4, i) for i in range(1, 4)  # Bottom row (excluding corners)
    ]
    cell_index = action // 3
    move_index = action % 3
    if cell_index < 3:  # Top row
        side_directions = [Move.LEFT, Move.RIGHT, Move.BOTTOM]
    elif cell_index < 6:  # Right column
        side_directions = [Move.TOP, Move.BOTTOM, Move.LEFT]
    elif cell_index < 9:  # Left column
        side_directions = [Move.TOP, Move.BOTTOM, Move.RIGHT]
    else:  # Bottom row
        side_directions = [Move.LEFT, Move.RIGHT, Move.TOP]

    return periphery[cell_index], side_directions[move_index]


def encode_action(pos: tuple[int, int], slide: Move) -> int:
    '''
    Converts the pos, Move from the environment to the action number for the agent.

    pos: Tuple representing the position (X,Y) of the cell.
    slide: Move enum value. (Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT)

    Returns an action number ranging from 0 to 43.

    Raises ValueError if the combination of pos and slide is invalid.
    '''

    # Define corners and their associated moves
    corners = {
        (0, 0): [Move.RIGHT, Move.BOTTOM],
        (0, 4): [Move.LEFT, Move.BOTTOM],
        (4, 0): [Move.RIGHT, Move.TOP],
        (4, 4): [Move.LEFT, Move.TOP]
    }
    corner_action_numbers = {(0, 0): 0, (0, 4): 2, (4, 0): 4, (4, 4): 6}

    # Check if the position is a corner and the move is valid
    if pos in corners:
        if slide not in corners[pos]:
            raise ValueError("Invalid move from corner position.")
        return corner_action_numbers[pos] + corners[pos].index(slide)

    # Define sides and their associated moves
    sides = {
        (0, 1): 8, (0, 2): 11, (0, 3): 14,  # Top row (excluding corners)
        (1, 4): 17, (2, 4): 20, (3, 4): 23,  # Right column (excluding corners)
        (1, 0): 26, (2, 0): 29, (3, 0): 32,  # Left column (excluding corners)
        (4, 1): 35, (4, 2): 38, (4, 3): 41   # Bottom row (excluding corners)
    }
    side_moves = {
        (0, 1): [Move.LEFT, Move.RIGHT, Move.BOTTOM],
        (0, 2): [Move.LEFT, Move.RIGHT, Move.BOTTOM],
        (0, 3): [Move.LEFT, Move.RIGHT, Move.BOTTOM],
        (1, 4): [Move.TOP, Move.BOTTOM, Move.LEFT],
        (2, 4): [Move.TOP, Move.BOTTOM, Move.LEFT],
        (3, 4): [Move.TOP, Move.BOTTOM, Move.LEFT],
        (1, 0): [Move.TOP, Move.BOTTOM, Move.RIGHT],
        (2, 0): [Move.TOP, Move.BOTTOM, Move.RIGHT],
        (3, 0): [Move.TOP, Move.BOTTOM, Move.RIGHT],
        (4, 1): [Move.LEFT, Move.RIGHT, Move.TOP],
        (4, 2): [Move.LEFT, Move.RIGHT, Move.TOP],
        (4, 3): [Move.LEFT, Move.RIGHT, Move.TOP]
    }

    # Check if the position is a side and the move is valid
    if pos in sides:
        if slide not in side_moves[pos]:
            raise ValueError("Invalid move from side position.")
        return sides[pos] + side_moves[pos].index(slide)

    # If the position is neither a corner nor a side
    raise ValueError("Invalid position for a move.")


def all_possible_actions(game: Game, player_id: int) -> list[int]:
    '''
    Returns a list of all possible action numbers for a given game state and player.

    game: The current state of the game.
    player_id: The ID of the player (0 or 1) for whom to calculate possible actions.
    '''

    possible_actions = []
    board = game.get_board()

    # Define all border positions
    border_positions = [
        (0, i) for i in range(5)  # Top row
    ] + [
        (4, i) for i in range(5)  # Bottom row
    ] + [
        (i, 0) for i in range(1, 4)  # Left column (excluding corners)
    ] + [
        (i, 4) for i in range(1, 4)  # Right column (excluding corners)
    ]

    for pos in border_positions:
        # Check if the cell belongs to the player or is neutral
        if board[pos] in [player_id, -1]:
            # Determine possible slides
            slides = Game.acceptable_slides(pos)
            # Encode each valid (pos, slide) pair into an action
            for slide in slides:
                action = encode_action(pos, slide)
                possible_actions.append(action)

    return possible_actions


# def tester():
#     for i in range(44):
#         pos, slide = decode_action(i)
#         print(f"{i}: {pos} {slide}")
#     print()


# tester()
