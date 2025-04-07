import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.path import Path

def plot_learning_curves(episode_rewards, episode_steps, window_size=100):
    """
    Plot the learning curves: rewards and steps per episode.
    
    Args:
        episode_rewards (list): Rewards for each episode
        episode_steps (list): Steps taken for each episode
        window_size (int): Size of the moving average window
    """
    episodes = range(1, len(episode_rewards) + 1)
    
    # Calculate moving averages
    smoothed_rewards = []
    smoothed_steps = []
    
    for i in range(len(episode_rewards)):
        start_idx = max(0, i - window_size + 1)
        smoothed_rewards.append(np.mean(episode_rewards[start_idx:i+1]))
        smoothed_steps.append(np.mean(episode_steps[start_idx:i+1]))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot reward curve
    ax1.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
    ax1.plot(episodes, smoothed_rewards, color='blue', linewidth=2, label=f'Moving Avg ({window_size})')
    ax1.set_ylabel('Reward per Episode')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot steps curve
    ax2.plot(episodes, episode_steps, alpha=0.3, color='green', label='Raw')
    ax2.plot(episodes, smoothed_steps, color='green', linewidth=2, label=f'Moving Avg ({window_size})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps per Episode')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_policy(env, policy):
    """
    Visualize the learned policy on the grid.
    
    Args:
        env: The grid environment
        policy (dict): Map from state to optimal action
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Define action arrows
    action_arrows = {
        0: '↑',  # Up
        1: '↓',  # Down
        2: '←',  # Left
        3: '→'   # Right
    }
    
    # Draw grid cells
    for i in range(env.rows):
        for j in range(env.cols):
            cell_type = env.grid[i][j]
            state = env.coords_to_state((i, j))
            
            # Set cell color based on type
            if cell_type == 'S':
                color = 'green'
                text = 'S'
            elif cell_type == 'G':
                color = 'gold'
                text = 'G'
            elif cell_type == 'H':
                color = 'red'
                text = 'H'
            elif cell_type == 'X':
                color = 'gray'
                text = ''
            else:  # Empty cell
                color = 'white'
                # Show policy arrow
                text = action_arrows[policy[state]] if state in policy else ''
            
            # Draw cell rectangle
            rect = plt.Rectangle((j, env.rows - i - 1), 1, 1, 
                                 facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            
            # Add text
            ax.text(j + 0.5, env.rows - i - 0.5, text, 
                    ha='center', va='center', fontsize=20, 
                    color='black' if color in ['white', 'gold'] else 'white')
    
    # Set axis limits and labels
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_xticks(np.arange(0, env.cols + 1, 1))
    ax.set_yticks(np.arange(0, env.rows + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.set_title('Learned Policy', fontsize=16)
    
    # Create legend
    legend_elements = [
        patches.Patch(facecolor='green', edgecolor='black', alpha=0.7, label='Start'),
        patches.Patch(facecolor='gold', edgecolor='black', alpha=0.7, label='Goal'),
        patches.Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Hazard'),
        patches.Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='Wall')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    plt.tight_layout()
    return fig

def plot_value_heatmap(env, state_values):
    """
    Visualize the state values as a heatmap.
    
    Args:
        env: The grid environment
        state_values (dict): Map from state to value
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a grid of values
    value_grid = np.zeros((env.rows, env.cols))
    min_value = float('inf')
    max_value = float('-inf')
    
    for state, value in state_values.items():
        i, j = env.state_to_coords(state)
        value_grid[i, j] = value
        
        # Track min/max values for non-wall cells
        if env.grid[i][j] != 'X':
            min_value = min(min_value, value)
            max_value = max(max_value, value)
    
    # Mark walls with a special value (will be colored black)
    for i in range(env.rows):
        for j in range(env.cols):
            if env.grid[i][j] == 'X':
                value_grid[i, j] = min_value - 1  # Lower than min for walls
    
    # Create a colormap that maps walls to black
    colors = [(0, 0, 0), (0, 0, 0.7), (0, 0.7, 0.9), (1, 1, 0), (1, 0, 0)]
    positions = [0, 0.05, 0.5, 0.8, 1]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))
    
    # Plot heatmap
    im = ax.imshow(value_grid, cmap=cmap, origin='upper')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('State Value', rotation=270, labelpad=20)
    
    # Add text markers for special cells
    for i in range(env.rows):
        for j in range(env.cols):
            cell_type = env.grid[i][j]
            text = ''
            
            if cell_type == 'S':
                text = 'S'
            elif cell_type == 'G':
                text = 'G'
            elif cell_type == 'H':
                text = 'H'
            elif cell_type == 'X':
                text = 'X'
                
            if text:
                ax.text(j, i, text, ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=12)
            
            # Add value text for non-wall cells
            if cell_type != 'X':
                val = value_grid[i, j]
                if abs(val) > 1:
                    val_text = f'{val:.1f}'
                else:
                    val_text = f'{val:.2f}'
                ax.text(j, i + 0.3, val_text, ha='center', va='center', 
                        color='white', fontsize=8)
    
    # Set grid lines
    ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    ax.set_title('State Value Landscape', fontsize=16)
    plt.tight_layout()
    
    return fig

def plot_epsilon_decay(epsilon_history):
    """
    Plot the epsilon decay over episodes.
    
    Args:
        epsilon_history (list): Epsilon values for each episode
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = range(1, len(epsilon_history) + 1)
    
    ax.plot(episodes, epsilon_history, 'b-', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon (Exploration Rate)')
    ax.set_title('Epsilon Decay Over Training')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def create_policy_animation(env, policy, num_frames=100, interval=200):
    """
    Create an animation of the agent following the optimal policy.
    
    Args:
        env: The grid environment
        policy (dict): Map from state to optimal action
        num_frames (int): Number of frames to generate
        interval (int): Interval between frames in milliseconds
        
    Returns:
        matplotlib.animation.FuncAnimation: The animation object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set up the figure
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_xticks(np.arange(0, env.cols + 1, 1))
    ax.set_yticks(np.arange(0, env.rows + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.set_title('Agent Following Optimal Policy', fontsize=16)
    
    # Reset environment for animation
    env.reset()
    
    # Draw initial grid
    grid_rects = []
    for i in range(env.rows):
        for j in range(env.cols):
            cell_type = env.grid[i][j]
            
            # Set cell color based on type
            if cell_type == 'S':
                color = 'green'
            elif cell_type == 'G':
                color = 'gold'
            elif cell_type == 'H':
                color = 'red'
            elif cell_type == 'X':
                color = 'gray'
            else:  # Empty cell
                color = 'white'
            
            # Draw cell rectangle
            rect = plt.Rectangle((j, env.rows - i - 1), 1, 1, 
                                facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            grid_rects.append(rect)
            
            # Add text for special cells
            if cell_type in ['S', 'G', 'H']:
                ax.text(j + 0.5, env.rows - i - 0.5, cell_type, 
                        ha='center', va='center', fontsize=20, 
                        color='black' if color == 'gold' else 'white')
    
    # Create agent marker (circle)
    agent_marker = plt.Circle((0.5, 0.5), 0.3, color='blue', alpha=0.8)
    ax.add_patch(agent_marker)
    
    # Create legend
    legend_elements = [
        patches.Patch(facecolor='green', edgecolor='black', alpha=0.7, label='Start'),
        patches.Patch(facecolor='gold', edgecolor='black', alpha=0.7, label='Goal'),
        patches.Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Hazard'),
        patches.Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='Wall'),
        patches.Circle((0, 0), 0.3, facecolor='blue', alpha=0.8, label='Agent'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
            bbox_to_anchor=(0.5, -0.05), ncol=5)
    
    # Frame update function
    def update(frame):
        # Reset if starting a new episode
        if frame == 0:
            state = env.reset()
            r, c = env.agent_position
            agent_marker.center = (c + 0.5, env.rows - r - 0.5)
            return [agent_marker]
        
        # Get current state and position
        state = env.coords_to_state(env.agent_position)
        
        # Choose action based on policy
        if state in policy:
            action = policy[state]
        else:
            # If state not in policy (e.g., terminal state), stay put
            return [agent_marker]
        
        # Take action
        next_state, reward, terminated, _, _ = env.step(action)
        
        # Update agent marker position
        r, c = env.agent_position
        agent_marker.center = (c + 0.5, env.rows - r - 0.5)
        
        # Reset if terminated or reached frame limit
        if terminated:
            env.reset()
        
        return [agent_marker]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=num_frames, 
                                 interval=interval, blit=True)
    
    plt.tight_layout()
    return fig, anim

def plot_comparison(env_low_slip, policy_low_slip, env_high_slip, policy_high_slip):
    """
    Compare policies with different slip probabilities.
    
    Args:
        env_low_slip: Environment with low slip probability
        policy_low_slip: Policy for low slip environment
        env_high_slip: Environment with high slip probability
        policy_high_slip: Policy for high slip environment
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define action arrows
    action_arrows = {
        0: '↑',  # Up
        1: '↓',  # Down
        2: '←',  # Left
        3: '→'   # Right
    }
    
    # Function to draw grid on an axis
    def draw_grid(ax, env, policy, title):
        # Draw grid cells
        for i in range(env.rows):
            for j in range(env.cols):
                cell_type = env.grid[i][j]
                state = env.coords_to_state((i, j))
                
                # Set cell color based on type
                if cell_type == 'S':
                    color = 'green'
                    text = 'S'
                elif cell_type == 'G':
                    color = 'gold'
                    text = 'G'
                elif cell_type == 'H':
                    color = 'red'
                    text = 'H'
                elif cell_type == 'X':
                    color = 'gray'
                    text = ''
                else:  # Empty cell
                    color = 'white'
                    # Show policy arrow
                    text = action_arrows[policy[state]] if state in policy else ''
                
                # Draw cell rectangle
                rect = plt.Rectangle((j, env.rows - i - 1), 1, 1, 
                                    facecolor=color, edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
                
                # Add text
                ax.text(j + 0.5, env.rows - i - 0.5, text, 
                        ha='center', va='center', fontsize=20, 
                        color='black' if color in ['white', 'gold'] else 'white')
        
        # Set axis limits and labels
        ax.set_xlim(0, env.cols)
        ax.set_ylim(0, env.rows)
        ax.set_xticks(np.arange(0, env.cols + 1, 1))
        ax.set_yticks(np.arange(0, env.rows + 1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='black', linestyle='-', linewidth=1)
        ax.set_title(title, fontsize=16)
    
    # Draw grids
    draw_grid(ax1, env_low_slip, policy_low_slip, 
              f'Low Slip Probability (p_slip={env_low_slip.p_slip_left + env_low_slip.p_slip_right:.2f})')
    draw_grid(ax2, env_high_slip, policy_high_slip, 
              f'High Slip Probability (p_slip={env_high_slip.p_slip_left + env_high_slip.p_slip_right:.2f})')
    
    # Create legend
    legend_elements = [
        patches.Patch(facecolor='green', edgecolor='black', alpha=0.7, label='Start'),
        patches.Patch(facecolor='gold', edgecolor='black', alpha=0.7, label='Goal'),
        patches.Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Hazard'),
        patches.Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='Wall')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0), ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    return fig 