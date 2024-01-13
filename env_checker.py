from stable_baselines3.common.env_checker import check_env
from env_random import QuixoEnv
from main import RandomPlayer

p = RandomPlayer()
env = QuixoEnv(p)
# It will check your custom environment and output additional warnings if needed
check_env(env)