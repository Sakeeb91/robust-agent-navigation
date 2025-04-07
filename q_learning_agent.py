import numpy as np
import matplotlib.pyplot as plt
import random

class QLearningAgent:
    """
    Q-Learning agent that learns to navigate the grid environment.
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.05):
        """
        Initialize the Q-Learning agent.
        
        Args:
            env: The grid environment
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Rate at which epsilon decays after each episode
            min_epsilon (float): Minimum exploration rate
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((env.num_states, env.num_actions))
        
        # For tracking training progress
        self.episode_rewards = []
        self.episode_steps = []
        self.epsilon_history = []
        
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state (int): Current state
            
        Returns:
            int: Selected action
        """
        # Exploration: choose a random action
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.actions)
        
        # Exploitation: choose best action (with random tie-breaking)
        q_values = self.q_table[state, :]
        max_q = np.max(q_values)
        actions_with_max_q = np.where(q_values == max_q)[0]
        
        # Choose randomly among actions with maximum Q-value (for tie-breaking)
        return np.random.choice(actions_with_max_q)
    
    def update_q_table(self, state, action, reward, next_state, terminated):
        """
        Update Q-table using the Q-learning update rule.
        
        Args:
            state (int): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (int): Next state
            terminated (bool): Whether the episode has terminated
        """
        # Calculate best future Q-value from next_state
        # If terminated, future Q-value is 0
        if terminated:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state, :])
        
        # Current Q-value for state-action pair
        current_q = self.q_table[state, action]
        
        # Q-learning update rule
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        
        # Update Q-table
        self.q_table[state, action] = new_q
    
    def decay_epsilon(self):
        """Decay epsilon, but not below min_epsilon."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def train(self, num_episodes):
        """
        Train the agent for the specified number of episodes.
        
        Args:
            num_episodes (int): Number of training episodes
            
        Returns:
            tuple: Lists of total rewards and steps per episode
        """
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            
            # Initialize episode variables
            total_reward = 0
            steps = 0
            terminated = False
            
            # Episode loop
            while not terminated:
                # Choose action
                action = self.choose_action(state)
                
                # Take action and observe outcome
                next_state, reward, terminated, _, _ = self.env.step(action)
                
                # Update Q-table
                self.update_q_table(state, action, reward, next_state, terminated)
                
                # Update state and tracking variables
                state = next_state
                total_reward += reward
                steps += 1
            
            # Decay exploration rate
            self.decay_epsilon()
            
            # Record episode metrics
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.epsilon_history.append(self.epsilon)
            
            # Print progress periodically
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Reward: {total_reward:.2f}, "
                      f"Steps: {steps}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        return self.episode_rewards, self.episode_steps
    
    def get_policy(self):
        """
        Extract the policy from the learned Q-values.
        
        Returns:
            dict: Map from state to optimal action
        """
        policy = {}
        for state in range(self.env.num_states):
            policy[state] = np.argmax(self.q_table[state, :])
        return policy
    
    def get_state_values(self):
        """
        Extract the state values from the learned Q-values.
        
        Returns:
            dict: Map from state to its value (max Q-value)
        """
        state_values = {}
        for state in range(self.env.num_states):
            state_values[state] = np.max(self.q_table[state, :])
        return state_values 