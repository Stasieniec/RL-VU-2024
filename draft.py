from env.maze_env import MazeEnv

maze = [
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0]
]

env = MazeEnv(maze)
state = env.reset()
env.render()

# Example moves (agent tries to move right, down, etc.)
actions = [3, 3, 1, 1]  # right, right, down, down
for action in actions:
    state, _, done, _ = env.step(action)
    env.render()