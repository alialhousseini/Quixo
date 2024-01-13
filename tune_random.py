import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env_random import QuixoEnv, RandomPlayer
from stable_baselines3.common.evaluation import evaluate_policy


def make_env():
    env = QuixoEnv(opponent=RandomPlayer())
    return env


def optimize_agent(trial):
    # Model Architecture Parameters
    n_layers = trial.suggest_int("n_layers", 2, 4)
    layer_size = trial.suggest_categorical("layer_size", [32, 64, 128])

    # Learning Parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)

    # Optimization Parameters
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_steps = trial.suggest_int("n_steps", 256, 2048, log=True)

    # Exploration-Exploitation Balance
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)

    # Create the environment
    # Make sure to define QuixoEnv and RandomPlayer correctly
    env = QuixoEnv(opponent=RandomPlayer())
    vec_env = make_vec_env(make_env, n_envs=4)

    # Create the PPO agent
    model = PPO("MlpPolicy", vec_env, verbose=0,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                batch_size=batch_size,
                n_steps=n_steps,
                ent_coef=ent_coef,
                policy_kwargs={"net_arch": [layer_size]*n_layers})

    # Train the agent
    for i in range(10):
        model.learn(total_timesteps=1000)
        model.save(f"/tund2/ppo_quixo{i+1}")

    # Evaluate the agent's performance
    reward = evaluate_model(env, model)
    return reward


def evaluate_model(env, model):
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True)
    return mean_reward


study = optuna.create_study(direction="maximize")
study.optimize(optimize_agent, n_trials=50)
