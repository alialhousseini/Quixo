from env_previous import QuixoEnv as QuixoEnvPrevious
from env_random import QuixoEnv as QuixoEnvRandom
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from main import RandomPlayer
from opponent import Opponent
import random


def make_env():
    choice = random.choice([0, 1])
    if choice:
        return QuixoEnvPrevious(opponent=Opponent("quixo_ppo_random_opponent_2M_previous"))
    else:
        return QuixoEnvRandom(opponent=RandomPlayer())


vec_env = make_vec_env(make_env, n_envs=5)

# Instantiate the agent
model = PPO("MlpPolicy", vec_env, verbose=1,
            tensorboard_log="./ppo/", device='cuda')

model = model.load("quixo_ppo_random_opponent_longest")

model.set_env(vec_env)

try:
    model.learn(total_timesteps=4e6)
except KeyboardInterrupt as e:
    print("Training interrupted")
    # Save the model
    model.save("quixo_ppo_random_opponent_longest2")
model.save("quixo_ppo_random_opponent_longest2")
