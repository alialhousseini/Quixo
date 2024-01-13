
from game import Game, Move, Player
import random
from utils import decode_action
import numpy as np
from stable_baselines3 import PPO


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        acceptable_moves = game.acceptable_slides(from_pos)
        move = random.choice(acceptable_moves)
        return from_pos, move


class Opponent(Player):
    def __init__(self, path: str) -> None:
        super().__init__()
        try:
            # C:\Users\a_h9\Desktop\Quixo\quixo_ppo_random_opponent_2M.zip
            self.model = PPO.load(path)
        except:
            print("Error loading model")

    def make_move(self, game: 'Game', obs: np.ndarray) -> tuple[tuple[int, int], Move]:
        action, _ = self.model.predict(obs)
        pos, slide = decode_action(action)
        valid = game.move(pos, slide, 1)
        while not valid:
            action, _ = self.model.predict(obs)
            pos, slide = decode_action(action)
            valid = game.move(pos, slide, 1)
        return decode_action(action)
