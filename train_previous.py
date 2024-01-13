from env_previous import QuixoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from main import RandomPlayer
from opponent import Opponent

# Create a single environment instance
env = QuixoEnv(opponent=Opponent("quixo_ppo_random_opponent_2M_previous"))


def make_env():
    return QuixoEnv(opponent=Opponent("quixo_ppo_random_opponent_2M_previous"))


# Wrap it in a vectorized environment
vec_env = make_vec_env(make_env, n_envs=5)

params = {'learning_rate': 3.8516223541221167e-05, 'gamma': 0.935919311668545,
          'gae_lambda': 0.869609567097365, 'batch_size': 256, 'n_steps': 997, 'ent_coef': 0.08256606818578316}

# Instantiate the agent
model = PPO("MlpPolicy", vec_env, verbose=1,
            tensorboard_log="./ppo/", device='cuda', **params)

model.load("quixo_ppo_random_opponent_2M_previous")

try:
    model.learn(total_timesteps=2e6)
except KeyboardInterrupt as e:
    print("Training interrupted")
    # Save the model
    model.save("quixo_ppo_random_opponent_2M_previous")
model.save("quixo_ppo_random_opponent_2M_previous")
