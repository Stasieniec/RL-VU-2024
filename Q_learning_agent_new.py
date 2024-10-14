# Here is the modified version of the Q-learning agent to track visited paths and output heatmaps every 20 episodes.
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from mazeEnv import MazeEnv

import numpy as np
import matplotlib.pyplot as plt

class QLearningAgentModified:
    def __init__(self, env, learning_rate=0.8, gamma=0.95, epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.001):
        self.env = env
        self.state_size = env.maze.shape[0] * env.maze.shape[1]  # Flattened state space
        self.action_size = env.action_space.n
        self.qtable = np.zeros((self.state_size, self.action_size))  # Initialize Q-table
        
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Exploration-exploitation parameters
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        
        # Tracking rewards
        self.rewards = []
        
        # Heatmap tracking variables
        self.visit_counts = np.zeros(env.maze.shape)  # To store visit counts
        self.episode_visit_counts = np.zeros(env.maze.shape)  # Counts for the current 20 episodes
        
    def to_state_index(self, position):
        """Helper function to convert 2D position to 1D state index."""
        return position[0] * self.env.maze.shape[1] + position[1]
    
    def train(self, total_episodes=1000, max_steps=100, display_interval=20):
        """Train the agent using Q-learning and output heatmaps every 'display_interval' episodes."""
        prev_qtable = np.copy(self.qtable)

        for episode in range(total_episodes):
            # Reset the environment at the start of each episode
            state = self.env.reset()
            done = False
            total_rewards = 0
            step = 0
            
            # Reset visit counts for this episode
            episode_visits = np.zeros(self.env.maze.shape)
            
            while not done and step < max_steps:
                step += 1
                
                # Convert state to 2D coordinates (position)
                pos = (state[0], state[1])
                
                # Add the visit to tracking
                episode_visits[pos] += 1
                
                # Take action based on epsilon-greedy strategy
                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()  # Explore
                else:
                    action = np.argmax(self.qtable[self.to_state_index(pos)])  # Exploit
                
                # Perform the action in the environment
                new_state, reward, done, info = self.env.step(action)
                total_rewards += reward
                
                # Update Q-table
                next_pos = (new_state[0], new_state[1])
                self.qtable[self.to_state_index(pos), action] = self.qtable[self.to_state_index(pos), action] + \
                    self.learning_rate * (reward + self.gamma * np.max(self.qtable[self.to_state_index(next_pos)]) - self.qtable[self.to_state_index(pos), action])
                
                # Transition to the new state
                state = new_state

            # Accumulate episode visits into overall visit counts
            self.episode_visit_counts += episode_visits

            # Decay epsilon to encourage exploitation
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

            # After every display_interval episodes, output the heatmap
            if (episode + 1) % display_interval == 0:
                self.display_heatmap(episode + 1)

                # Reset visit counts for the next batch of episodes
                self.episode_visit_counts = np.zeros(self.env.maze.shape)
    
    def display_heatmap(self, episode):
        """Display a heatmap of the most visited paths."""
        plt.figure(figsize=(8, 6))
        plt.imshow(self.episode_visit_counts, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Visit Frequency')
        plt.title(f'Heatmap of Visited Positions - Up to Episode {episode}')
        plt.show()

# This class will now track the visits and display heatmaps every 20 episodes.
# You can integrate this into your existing system and call the `train()` method.
if __name__ == "__main__":
    # Initialize environment
    env = MazeEnv(render_mode="human")

    # Create Q-learning agent
    agent = QLearningAgentModified(env)

    # Train the agent
    agent.train(total_episodes=1000, max_steps=100)

    # Save the learned Q-table
    # agent.save_qtable()

    # Print the final Q-table
    print("Final Q-table:")
    print(agent.get_qtable())

    # Plot the final Q-table with correctly rotated arrows
    agent.plot_qtable_on_maze()

    # Plot average cumulative rewards
    agent.plot_average_cumulative_rewards()