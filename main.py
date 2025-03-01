import torch
import numpy as np
from game import SnakeGameAI, Direction, Point
from agent import Agent

def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)),

        # Danger right
        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),

        # Direction
        dir_l, dir_r, dir_u, dir_d,

        # Food location
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y   # food down
    ]
    return np.array(state, dtype=int)

def train():
    input_size = 11
    hidden_size = 256
    output_size = 3  # [straight, right turn, left turn]
    lr = 0.001
    gamma = 0.9

    agent = Agent(input_size, hidden_size, output_size, lr, gamma)
    game = SnakeGameAI()

    while True:
        state_old = get_state(game)
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        state_new = get_state(game)

        agent.remember(state_old, action, reward, state_new, done)
        agent.train_long_memory()

        if done:
            game.reset()
            agent.n_games += 1
            print(f'Game {agent.n_games}, Score: {score}')

            if agent.n_games % 10 == 0:
                torch.save(agent.model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train()
