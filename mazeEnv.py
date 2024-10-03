import gymnasium as gym
import numpy as np

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
