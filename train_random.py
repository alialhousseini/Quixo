from env_random import QuixoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from main import RandomPlayer

# Create a single environment instance
env = QuixoEnv(opponent=RandomPlayer())


def make_env():
    return QuixoEnv(opponent=RandomPlayer())


# Wrap it in a vectorized environment
vec_env = make_vec_env(make_env, n_envs=5)

params = {'learning_rate': 0.00047987507467331137, 'gamma': 0.9037646415026495,
          'gae_lambda': 0.8496204061939574, 'batch_size': 256, 'n_steps': 930, 'ent_coef': 0.04630735973520918}

# Instantiate the agent
model = PPO("MlpPolicy", vec_env, verbose=1,
            tensorboard_log="./ppo/", device='cuda', **params)

model.load("quixo_ppo_random_opponent")
try:
    model.learn(total_timesteps=2e6)
except KeyboardInterrupt as e:
    print("Training interrupted")
    # Save the model
    model.save("quixo_ppo_random_opponent_2M")
model.save("quixo_ppo_random_opponent_2M")
