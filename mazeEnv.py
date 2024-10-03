import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.registration import register
import time

register(
     id="gym_examples/mazeEnv-v0",
     entry_point="gym_examples.envs:MazeEnv",
     max_episode_steps=300,
)

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        # Define the maze layout
        self.maze = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 3, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 4, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

        self.size = self.maze.shape[0]  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Initial positions for the agent, sub-goal, and end-goal
        self.start_position = np.array([1, 1])
        self.sub_goal_position = np.array([5, 3])
        self.end_goal_position = np.array([8, 7])
        self.agent_position = self.start_position.copy()

        self.reached_sub_goal = False
        self.reached_end_position = False

        # Observation space: agent's position
        self.observation_space = gym.spaces.Box(0, self.size - 1, shape=(2,), dtype=int)

        # Action space: 4 possible actions (right, up, left, down)
        self.action_space = gym.spaces.Discrete(4)

        # Movement direction corresponding to each action
        self._action_to_direction = {
            0: np.array([0, 1]),   # Right
            1: np.array([-1, 0]),  # Up
            2: np.array([0, -1]),  # Left
            3: np.array([1, 0]),   # Down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def is_valid_position(self, position):
        """
        Function that checks wheter the position is in the valid bounds of the environment
        
        :param position: position of the agent
        
        :return valid: boolean value signifying wheter the position is valid

        FUNCTION IS YET TO BE IMPLEMENTED
        """
        # IMPLEMENT THIS FUNCTION BELOW
        return True
        
        
    def reset(self):
        """
        Resets the environment to the initial state 
        """
        # put agent on the starting position
        self.agent_position = self.start_position.copy()

        # reset all the flags to false 
        self.reached_sub_goal = False
        self.reached_end_position = False

        # return the reset agent position
        return self.agent_position

    def step(self, action):
        """
        Perform the step action for the agent.

        :param action: action of the agent (up, down, left, right)

        :return: observation (agent's position), reward, done, info dictionary
        """

        # Take the action and calculate the new position
        direction = self._action_to_direction[action]
        new_position = self.agent_position + direction

        # Initialize default values
        reward = 0
        done = False

        # Check if the new position is valid
        if self.is_valid_position(new_position):
            self.agent_position = new_position

        # Check if the agent has reached the sub-goal
        if np.array_equal(self.agent_position, self.sub_goal_position):
            self.reached_sub_goal = True

        # Check if the agent has reached the end-goal
        if np.array_equal(self.agent_position, self.end_goal_position):
            self.reached_end_position = True

        # If both sub-goal and end-goal are reached, the episode is done
        if self.reached_sub_goal and self.reached_end_position:
            done = True
            reward = 1  # Reward for completing both goals

        return self.agent_position, reward, done, {}
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        # Initialize pygame window if it's not already created
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create a canvas of the window size to draw on
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # Set background to white
        pix_square_size = self.window_size / self.size  # Size of each grid square

        # Draw the maze grid (walls and open paths)
        for row in range(self.size):
            for col in range(self.size):
                if self.maze[row, col] == 1:  # Wall
                    color = (0, 0, 0)  # Black for walls
                elif self.maze[row, col] == 0:  # Open path
                    color = (255, 255, 255)  # White for open paths
                elif self.maze[row, col] == 2:  # Start position
                    color = (0, 255, 0)  # Green for start
                elif self.maze[row, col] == 3:  # Sub-goal
                    color = (255, 165, 0)  # Orange for sub-goal
                elif self.maze[row, col] == 4:  # End goal
                    color = (255, 0, 0)  # Red for end goal
                
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        col * pix_square_size, row * pix_square_size, pix_square_size, pix_square_size
                    ),
                )

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # Blue for the agent
            (self.agent_position[1] * pix_square_size + pix_square_size / 2,
            self.agent_position[0] * pix_square_size + pix_square_size / 2),
            pix_square_size / 3
        )

        # Add gridlines for better visualization
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),  # Black for gridlines
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),  # Black for gridlines
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # Copy the canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Ensure that rendering occurs at the set frame rate
            self.clock.tick(self.metadata["render_fps"])
        else:  # If rendering as an rgb array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


# Create the environment
env = MazeEnv(render_mode="human")
env.reset()

# Run the environment for a few steps to visualize it
done = False
while not done:
    action = env.action_space.sample()  # Take a random action
    obs, reward, done, info = env.step(action)
    env.render()  # Render the environment

    time.sleep(0.2)  # Add a small delay to slow down the loop

# Close the environment after the loop ends
env.close()


    
