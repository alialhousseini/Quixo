import numpy as np
import gymnasium as gym
import stable_baselines3 as sb3
import random
from copy import deepcopy
from typing import Tuple, List
from game import Game, Move
from utils import *
from opponent import *

class QuixoEnv(gym.Env):
    """
    Quixo Env --- Gym environment for the Quixo game against itself
    """

    def __init__(self, opponent: Opponent) -> None:
        """
        Quixo Env constructor

        """
        self.n_actions = 44
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(5,5), dtype=int)
        self.game = Game()
        self.game.current_player_idx = 0
        self.state = self.game.get_board()
        
        self.reward = 0
        self.num_steps = 0
        self.opponent = opponent
        
        self.info = {}
    
    def getstate(self) -> np.ndarray:
        return self.game.get_board()
    
    def step(self, action) -> Tuple[np.ndarray, int, bool, bool, dict]:
        """
        take a step in the game
        """
        self.num_steps += 1
        reward = 0
        done = False
        self.game.current_player_idx = 0
        
        # Check limit
        if self.num_steps > 60: #Could be changed
            # The game must finish with a win to our agent in less than 50 moves (1 step = 1 move)
            done = True
            reward = -20
            return self.getstate(), reward, done, False, self.info
        
        #print(f"Player before first check is : {self.game.get_current_player()}")
        if self.game.get_current_player() == 0: # Agent
            pos, slide = decode_action(action)
            is_valid = self.game.move(pos, slide, self.game.get_current_player())
            print(f"Move: {pos} {slide} by {self.game.current_player_idx}")
            if not is_valid:
                done = True
                reward = -20
                print(f"Invalid move: {pos} {slide} because {self.game.get_board()[pos[1], pos[0]]}")
                return self.getstate(), reward, done, False, self.info

        if self.game.check_winner() == 0: # Agent won
            done = True
            reward = 30
            print(f"Agent won: {self.game.get_current_player()}")
            return self.getstate(), reward, done, False, self.info
        
        #print(f"Player before: {self.game.get_current_player()}")
        self.game.current_player_idx = 1 - self.game.current_player_idx # Switch player
        
        #print(f"Player After: {self.game.get_current_player()}")
        if self.game.get_current_player() == 1: # Opponent
            pos2, slide2 = self.opponent.make_move(self.game, self.getstate())
            try:
                self.game.move(pos2, slide2, self.game.get_current_player())
                print(f"Move: {pos2} {slide2} by {self.game.current_player_idx}")
            except self.game.get_current_player() == 0:
                print("Error")
            
        if self.game.check_winner() == 1: # Opponent won
            done = True
            reward = -30
            print(f"Opponent won {self.game.get_current_player()}")
            return self.getstate(), reward, done, False, self.info
        
        return self.getstate(), reward, done, False, self.info
    
    def reset(self, seed=None, Options=None) -> Tuple[np.ndarray, dict]:
        """
        reset the board game and state
        """
        self.num_steps = 0
        
        self.game = Game()
        
        self.game.current_player_idx = 0
        
        self.state = self.game.get_board()
        
        return self.state, self.info
    
    def render(self, mode="human") -> None:
        return  self.game.print()


