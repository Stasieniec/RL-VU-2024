# Modified REINFORCE agent with heatmap visualization every 20 episodes
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from mazeEnv import MazeEnv

import numpy as np
import matplotlib.pyplot as plt

class REINFORCEAgentModified:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Initialize policy: a table of action probabilities (softmax over actions)
        self.policy = np.ones((env.maze.shape[0] * env.maze.shape[1], env.action_space.n)) / env.action_space.n
        
        # Heatmap tracking variables
        self.episode_visit_counts = np.zeros(env.maze.shape)  # Counts for the current 20 episodes
    
    def to_state_index(self, position):
        """Helper function to convert 2D position to 1D state index."""
        return position[0] * self.env.maze.shape[1] + position[1]
    
    def get_action(self, state_index):
        """Select an action based on the current policy (softmax probabilities)."""
        action_probabilities = self.policy[state_index, :]
        action = np.random.choice(self.env.action_space.n, p=action_probabilities)
        return action
    
    def train(self, total_episodes=1000, max_steps=100, display_interval=20):
        """Train the agent using REINFORCE and output heatmaps every 'display_interval' episodes."""
        
        for episode in range(total_episodes):
            # Reset the environment
            state = self.env.reset()
            done = False
            step = 0
            
            # Reset visit counts for this episode
            episode_visits = np.zeros(self.env.maze.shape)
            
            while not done and step < max_steps:
                step += 1
                
                # Convert state to 2D coordinates (position)
                pos = (state[0], state[1])
                
                # Add the visit to tracking
                episode_visits[pos] += 1
                
                # Choose action based on the current policy
                state_index = self.to_state_index(pos)
                action = self.get_action(state_index)
                
                # Perform the action in the environment
                new_state, reward, done, info = self.env.step(action)
                
                # Update the policy based on rewards here (this is the REINFORCE update logic)
                # [This section would contain policy gradient updates]
                
                # Transition to the new state
                state = new_state

            # Accumulate episode visits into overall visit counts
            self.episode_visit_counts += episode_visits

            # Every 'display_interval' episodes, output the heatmap
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

    # Create REINFORCE agent
    reinforce_agent = REINFORCEAgentModified(env)

    # Train the agent
    reinforce_agent.train(total_episodes=1000, max_steps=100)

    # After training, you can evaluate the learned policy
    print("Training complete!")