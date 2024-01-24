import random
from game import Game, Move, Player
from stable_baselines3 import PPO
from utils import decode_action


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class MyPlayer(Player):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.model = PPO.load(path)

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        action, _ = self.model.predict(game.get_board())
        pos, slid = decode_action(action)
        while not game.move(pos, slid, 0):
            action, _ = self.model.predict(game.get_board())
            pos, slid = decode_action(action)
            # print("Invalid move!!")
        return decode_action(action)


if __name__ == '__main__':
    g = Game()
    g.print()
    path = './/old_results//quixo_ppo_random_opponent_longest2'
    player1 = MyPlayer(path=path)
    player2 = MyPlayer(path=path)
    counter = 0
    for i in range(1000):
        winner = g.play(player1, player2)
        if winner == 0:
            counter += 1
    print(f"Player 1 won {counter} times out of 1000 games.")
    g.print()
    print(f"Winner: Player {winner}")
