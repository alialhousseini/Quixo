## Quixo Game using Reinforcement Learning

#### How to read this repo?
First I invite you have a look to ```draft.pdf``` where it is a scientific research draft article where it contains all information.

Main files:
- `env_previous.py`: Env that uses previous trained version of it
- `env_random.py`: Env that uses a random player
- `game.py`: The game functions and play methods
- `train_*.py`: Training Scripts
- `tune_*.py`: Tuning scripts
- `utils.py`: Necessary functions to connect between the game and the RL env (encode and decode actions)
- `test.py`: Testing the trained agent
- `env_checker.py`: A script to check the configuration of the env.

Secondary files:
- `Quixo.pdf`: An explanation of the game
- `requirements.txt`: The python env libraries and packages
- `quixo_ppo_random_opponent_*.zip`: PPO trained models.
    - `2M, 2M_previous, longest` are the first, second and third trainings respectively.

Non-used files:
- `easyAI.py`: An non-complete implementation of Negamax for Quixo.
- `mcts.py`: An non-complete implementation of Monte-Carlo Tree Search for Quixo.

_____
This work has been done as a project for the Computational Intelligence course 23/24 @ PoliTO.

https://github.com/squillero/computational-intelligence
