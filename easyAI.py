from game import Game, Move, Player
import random
from utils import decode_action
import numpy as np
from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax


class QuixoAI(TwoPlayersGame):
    def __init__(self, players, board=None):
        self.players = players
        self.game = Game()
        self.nplayer = 0

    def possible_moves(self):
        # return self.game.possible_moves()
        
    def make_move(self, move):
        pass

    def unmake_move(self, move):
        pass

    def is_over(self):
        pass

    def show(self):
        pass

    def scoring(self):
        pass
