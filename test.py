from env_random import QuixoEnv as QuixoEnvRandom
from env_previous import QuixoEnv as QuixoEnvPrevious
from main import RandomPlayer
from opponent import Opponent
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import random


def make_env():
    # choice = random.choice([0, 1])
    # if choice:
    return QuixoEnvPrevious(opponent=Opponent("quixo_ppo_random_opponent_longest"))
    # else:
    # return QuixoEnvRandom(opponent=RandomPlayer())


vec_env = make_vec_env(make_env, n_envs=5)

# model = PPO.load("quixo_ppo_random_opponent_longest")
# mean_reward, std_reward = evaluate_policy(
#     model, vec_env, n_eval_episodes=10, deterministic=True)
# print(mean_reward, std_reward)

env = make_env()
env.reset()
model = PPO.load("quixo_ppo_random_opponent_longest")
obs, _ = env.reset()
done = False
for i in range(1):
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _,  info = env.step(action)
        env.render()
        if done:
            break
