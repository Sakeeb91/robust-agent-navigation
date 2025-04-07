import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import os

from grid_environment import GridEnvironment
from q_learning_agent import QLearningAgent
import visualization as viz

# Create output directory for saving visualizations
if not os.path.exists('results'):
    os.makedirs('results')

def main():
    # Define grid layouts
    grid_layout = [
        "XXXXXX",
        "XS...X",
        "X.XX.X",
        "X..H.X",
        "X...GX",
        "XXXXXX"
    ]
    
    print("Starting Q-Learning simulation in grid world...")
    
    # Low slip probability environment
    print("\nTraining agent with LOW slip probability...")
    env_low_slip = GridEnvironment(grid_layout, p_intended=0.9, p_slip_left=0.05, p_slip_right=0.05)
    
    # Create and train Q-learning agent
    agent_low_slip = QLearningAgent(env_low_slip, alpha=0.1, gamma=0.95, 
                                   epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01)
    
    # Train the agent for 1000 episodes
    rewards_low_slip, steps_low_slip = agent_low_slip.train(1000)
    
    # Get policy and state values
    policy_low_slip = agent_low_slip.get_policy()
    state_values_low_slip = agent_low_slip.get_state_values()
    
    # High slip probability environment
    print("\nTraining agent with HIGH slip probability...")
    env_high_slip = GridEnvironment(grid_layout, p_intended=0.6, p_slip_left=0.2, p_slip_right=0.2)
    
    # Create and train Q-learning agent
    agent_high_slip = QLearningAgent(env_high_slip, alpha=0.1, gamma=0.95, 
                                    epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01)
    
    # Train the agent for 1000 episodes
    rewards_high_slip, steps_high_slip = agent_high_slip.train(1000)
    
    # Get policy and state values
    policy_high_slip = agent_high_slip.get_policy()
    state_values_high_slip = agent_high_slip.get_state_values()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # VIZ 1: Learning Curves
    fig_learning = viz.plot_learning_curves(rewards_low_slip, steps_low_slip)
    fig_learning.savefig('results/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig_learning)
    
    # VIZ 2: Final Optimal Policy
    fig_policy = viz.plot_policy(env_low_slip, policy_low_slip)
    fig_policy.savefig('results/optimal_policy.png', dpi=300, bbox_inches='tight')
    plt.close(fig_policy)
    
    # VIZ 3: Learned Value Heatmap
    fig_value = viz.plot_value_heatmap(env_low_slip, state_values_low_slip)
    fig_value.savefig('results/value_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig_value)
    
    # VIZ 4: Policy Comparison (Low vs. High Slip)
    fig_comparison = viz.plot_comparison(env_low_slip, policy_low_slip, 
                                          env_high_slip, policy_high_slip)
    fig_comparison.savefig('results/policy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig_comparison)
    
    # VIZ 5: Epsilon Decay Curve
    fig_epsilon = viz.plot_epsilon_decay(agent_low_slip.epsilon_history)
    fig_epsilon.savefig('results/epsilon_decay.png', dpi=300, bbox_inches='tight')
    plt.close(fig_epsilon)
    
    # VIZ 7: Optimal Path Animation
    print("\nCreating policy animation...")
    fig_anim, anim = viz.create_policy_animation(env_low_slip, policy_low_slip, 
                                               num_frames=200, interval=200)
    
    # Save animation as GIF
    writer = PillowWriter(fps=5)
    anim.save('results/policy_animation.gif', writer=writer)
    plt.close(fig_anim)
    
    print("\nSimulation complete! Results saved to 'results/' directory.")
    print("\nKey findings:")
    print(f"- Low slip (p={env_low_slip.p_slip_left + env_low_slip.p_slip_right:.2f}): Final avg reward: {np.mean(rewards_low_slip[-100:]):.2f}, Steps: {np.mean(steps_low_slip[-100:]):.2f}")
    print(f"- High slip (p={env_high_slip.p_slip_left + env_high_slip.p_slip_right:.2f}): Final avg reward: {np.mean(rewards_high_slip[-100:]):.2f}, Steps: {np.mean(steps_high_slip[-100:]):.2f}")
    print("\nPolicy differences:")
    print("- The high slip policy tends to be more conservative, avoiding walls and hazards where slips could be costly.")
    print("- The agent learns to balance exploration and exploitation through epsilon-greedy strategy.")

if __name__ == "__main__":
    main() 