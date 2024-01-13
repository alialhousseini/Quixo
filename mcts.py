import numpy as np
from collections import defaultdict
from game import Game, Move
from typing import List, Tuple
from utils import decode_action, encode_action, all_possible_actions
import random


class MonteCarloTreeSearchNode():
    def __init__(self, game: Game, parent=None, parent_action=None) -> None:
        self.game: Game = game
        # self.state: np.ndarray = self.game.get_board()
        self.parent: MonteCarloTreeSearchNode = parent
        self.parent_action: int = parent_action
        self.children: List[MonteCarloTreeSearchNode] = []
        self._number_of_visits: int = 0

        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0

        self._untried_actions: List[int] = None
        self._untried_actions = self.untried_actions()

        return

    def untried_actions(self) -> List[int]:
        self._untried_actions = all_possible_actions(self.game, 1)
        return self._untried_actions

    def q(self) -> int:
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self) -> int:
        return self._number_of_visits

    def expand(self) -> "MonteCarloTreeSearchNode":
        m = random.choice(self._untried_actions)
        pos, slide = decode_action(m)
        valid = self.game.move(pos, slide, 1)

        while not valid:
            m = random.choice(self._untried_actions)
            pos, slide = decode_action(m)
            valid = self.game.move(pos, slide, 1)

        child_node = MonteCarloTreeSearchNode(
            self.game, parent=self, parent_action=encode_action(pos, slide))

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self) -> bool:
        return self.game.check_winner() != -1

    def rollout(self):
        current_rollout_state = self.game

        while not current_rollout_state.is_game_over():

            possible_moves = all_possible_actions(
                current_rollout_state, 1)

            action = self.rollout_policy(possible_moves)
            pos, slide = decode_action(action)

            self.game.move(pos, slide, 1)

        return current_rollout_state.is_game_over()

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):

        choices_weights = [(c.q() / c.n()) + c_param *
                           np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 100

        for i in range(simulation_no):

            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return self.best_child(c_param=0.)

    def get_legal_actions(self):
        '''
        Modify according to your game or
        needs. Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
        return all_possible_actions(self.game, 1)

    def is_game_over(self):
        '''
        Modify according to your game or 
        needs. It is the game over condition
        and depends on your game. Returns
        true or false
        '''
        pass

    def game_result(self):
        '''
        Modify according to your game or 
        needs. Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        '''
        if self.game.check_winner() == 1:
            return 1
        else:
            return 0

    def move(self, action):
        '''
        Modify according to your game or 
        needs. Changes the state of your 
        board with a new value. For a normal
        Tic Tac Toe game, it can be a 3 by 3
        array with all the elements of array
        being 0 initially. 0 means the board 
        position is empty. If you place x in
        row 2 column 3, then it would be some 
        thing like board[2][3] = 1, where 1
        represents that x is placed. Returns 
        the new state after making a move.
        '''
        pass


def main():
    g = Game()
    root = MonteCarloTreeSearchNode(g)
    selected_node = root.best_action()
    print(selected_node.parent_action)
    return


main()
