import numpy as np
import matplotlib.pyplot as plt

class GridEnvironment:
    """
    Grid world environment with stochastic movement (slippery floor).
    The agent can move in four directions, but with some probability,
    it may slip to the left or right of the intended direction.
    """
    
    def __init__(self, grid_layout, p_intended=0.8, p_slip_left=0.1, p_slip_right=0.1):
        """
        Initialize the grid environment.
        
        Args:
            grid_layout (list): A list of strings representing the grid layout.
                               'S' - Start, 'G' - Goal, '.' - Empty, 'X' - Wall/Obstacle, 'H' - Hazard
            p_intended (float): Probability of moving in the intended direction
            p_slip_left (float): Probability of slipping left relative to intended direction
            p_slip_right (float): Probability of slipping right relative to intended direction
        """
        self.grid = [list(row) for row in grid_layout]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.p_intended = p_intended
        self.p_slip_left = p_slip_left
        self.p_slip_right = p_slip_right
        
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.actions = [0, 1, 2, 3]
        self.action_to_direction = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        # Find start position
        self.start_position = None
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] == 'S':
                    self.start_position = (i, j)
                    break
            if self.start_position:
                break
        
        if not self.start_position:
            raise ValueError("Grid layout must contain a start position 'S'")
        
        # State mapping: map (row, col) coordinates to a state number
        self.state_mapping = {}
        self.inverse_state_mapping = {}
        state_idx = 0
        for i in range(self.rows):
            for j in range(self.cols):
                self.state_mapping[(i, j)] = state_idx
                self.inverse_state_mapping[state_idx] = (i, j)
                state_idx += 1
                
        self.num_states = self.rows * self.cols
        self.num_actions = len(self.actions)
        
        # Current agent position
        self.agent_position = None
    
    def reset(self):
        """Reset environment to starting state."""
        self.agent_position = self.start_position
        return self.coords_to_state(self.agent_position)
    
    def step(self, action):
        """
        Execute action and return next state, reward, and termination status.
        
        Args:
            action (int): Action to take (0=Up, 1=Down, 2=Left, 3=Right)
        
        Returns:
            tuple: (next_state, reward, terminated, truncated, info)
        """
        # Check if action is valid
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")
            
        # Determine outcome based on stochasticity
        outcome = np.random.choice(['intended', 'slip_left', 'slip_right'], 
                                 p=[self.p_intended, self.p_slip_left, self.p_slip_right])
        
        # Convert action to intended direction based on outcome
        if outcome == 'intended':
            dr, dc = self.action_to_direction[action]
        elif outcome == 'slip_left':
            # Slip left relative to intended direction
            slip_actions = {0: 2, 1: 3, 2: 1, 3: 0}  # Up→Left, Down→Right, Left→Down, Right→Up
            dr, dc = self.action_to_direction[slip_actions[action]]
        else:  # slip_right
            # Slip right relative to intended direction
            slip_actions = {0: 3, 1: 2, 2: 0, 3: 1}  # Up→Right, Down→Left, Left→Up, Right→Down
            dr, dc = self.action_to_direction[slip_actions[action]]
            
        # Calculate new position
        new_r, new_c = self.agent_position[0] + dr, self.agent_position[1] + dc
        
        # Handle boundary conditions and obstacles
        if (0 <= new_r < self.rows and 0 <= new_c < self.cols and 
            self.grid[new_r][new_c] != 'X'):
            new_position = (new_r, new_c)
        else:
            # Hit wall or boundary - stay in place
            new_position = self.agent_position
            
        # Determine reward and termination
        reward = 0
        terminated = False
        
        cell_type = self.grid[new_position[0]][new_position[1]]
        
        if cell_type == 'G':
            reward = 50
            terminated = True
        elif cell_type == 'H':
            reward = -50
            terminated = True
        elif cell_type == '.':
            reward = -0.1  # Small penalty for each step
        elif cell_type == 'S':
            reward = -0.1  # Small penalty for each step
        elif new_position == self.agent_position:
            # Penalty for bumping into wall or boundary
            reward = -1
            
        # Update agent position
        self.agent_position = new_position
        
        return (self.coords_to_state(new_position), reward, terminated, False, {})
    
    def coords_to_state(self, coords):
        """Convert (row, col) coordinates to state number."""
        return self.state_mapping[coords]
    
    def state_to_coords(self, state):
        """Convert state number to (row, col) coordinates."""
        return self.inverse_state_mapping[state]
    
    def render(self, mode='human'):
        """Render the environment."""
        grid_copy = [row.copy() for row in self.grid]
        r, c = self.agent_position
        
        # Mark agent position with 'A', unless it's on start, goal, or hazard
        if grid_copy[r][c] not in ['S', 'G', 'H']:
            grid_copy[r][c] = 'A'
            
        for row in grid_copy:
            print(''.join(row))
        print()
    
    def get_cell_type(self, coords):
        """Get the type of cell at the given coordinates."""
        r, c = coords
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c]
        return None 