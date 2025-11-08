# Multi-Agent System Simulation with Cooperative Game Theory
# This code simulates a multi-agent system where agents must navigate to targets
# while avoiding obstacles, maintaining formation, and dealing with adversaries.

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

# Fix for Kaggle readonly database error - prevents IPython history issues
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='IPython')

# Disable IPython history to avoid readonly database issues in some environments
import os
os.environ['IPYTHONDIR'] = '/tmp/.ipython'

try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.magic('config HistoryManager.hist_file = ""')
        ipython.magic('config HistoryManager.enabled = False')
except:
    pass

# Standard imports for numerical computation and visualization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque

# ============================================================================
# CORE CLASSES
# ============================================================================

class Agent:
    """
    Represents a single agent in the Multi-Agent System.
    
    Each agent has:
    - A unique ID for identification
    - A current position in 2D space
    - A history of all previous positions for trajectory tracking
    """
    def __init__(self, agent_id, position):
        """
        Initialize an agent.
        
        Args:
            agent_id (int): Unique identifier for this agent
            position (list/array): Initial [x, y] position
        """
        self.id = agent_id
        self.pos = np.array(position, dtype=float)  # Current position as numpy array
        self.history = [self.pos.copy()]  # Track all positions for visualization

    def update_position(self, new_position):
        """
        Updates the agent's position and stores its history.
        
        Args:
            new_position (array): New [x, y] position
        """
        self.pos = new_position
        self.history.append(self.pos.copy())  # Store copy to avoid reference issues

class Adversary:
    """
    Represents the external interference or moving obstacle.
    
    The adversary follows a predetermined trajectory that agents must avoid.
    It can follow different movement patterns depending on the simulation time.
    """
    def __init__(self, initial_position, trajectory_params):
        """
        Initialize the adversary.
        
        Args:
            initial_position (list/array): Starting [x, y] position
            trajectory_params (dict): Parameters defining movement pattern
        """
        self.pos = np.array(initial_position, dtype=float)
        self.params = trajectory_params  # Contains movement parameters (a, b, c, x_vel)
        self.history = [self.pos.copy()]  # Track positions for visualization
        self.time = 0  # Internal time counter

    def update_position(self):
        """
        Updates the adversary's position based on predefined movement rules.
        
        For time steps 11-15: Follows a linear interference path from (50,35) to (20,0)
        Otherwise: Follows a quadratic trajectory defined by parameters
        """
        self.time += 1

        # Special interference path during time steps 11-15
        if 11 <= self.time <= 15:
            start_pos = np.array([50.0, 35.0])
            end_pos = np.array([20.0, 0.0])
            # Linear interpolation between start and end points
            ratio = (self.time - 11) / 4.0  # Progress from 0 to 1 over 4 steps
            self.pos = start_pos + ratio * (end_pos - start_pos)
        else:
            # Standard quadratic trajectory: y = ax² + bx + c
            a, b, c, x_vel = self.params['a'], self.params['b'], self.params['c'], self.params['x_vel']
            new_x = self.pos[0] + x_vel  # Move horizontally at constant velocity
            new_y = a * (new_x**2) + b * new_x + c  # Calculate y using quadratic formula
            
            # Ensure position stays within simulation bounds [0, 100]
            new_x = max(0, min(100, new_x))
            new_y = max(0, min(100, new_y))
            self.pos = np.array([new_x, new_y])
        
        self.history.append(self.pos.copy())

# ============================================================================
# UTILITY FUNCTIONS FOR GEOMETRIC CALCULATIONS
# ============================================================================

def point_to_line_distance(point, line_start, line_end):
    """
    Calculate the shortest distance from a point to a line segment.
    
    This is used to determine how close agents are to forbidden paths.
    
    Args:
        point (array): [x, y] coordinates of the point
        line_start (array): [x, y] start of line segment
        line_end (array): [x, y] end of line segment
    
    Returns:
        float: Shortest distance from point to line segment
    """
    line_vec = line_end - line_start  # Vector from start to end of line
    point_vec = point - line_start    # Vector from line start to point
    
    line_len = np.linalg.norm(line_vec)  # Length of the line segment
    if line_len == 0:  # Handle degenerate case where line is a point
        return np.linalg.norm(point_vec)
    
    # Project point onto the line and clamp to segment bounds
    line_unitvec = line_vec / line_len
    proj_length = np.dot(point_vec, line_unitvec)
    proj_length = max(0, min(line_len, proj_length))  # Clamp to [0, line_len]
    
    # Find closest point on segment and calculate distance
    closest_point = line_start + proj_length * line_unitvec
    return np.linalg.norm(point - closest_point)

def crosses_interference_line(pos1, pos2, line_start, line_end):
    """
    Check if movement from pos1 to pos2 crosses the interference line segment.
    
    Uses the crossing number algorithm to detect line intersection.
    
    Args:
        pos1, pos2 (array): Start and end positions of movement
        line_start, line_end (array): Endpoints of interference line
    
    Returns:
        bool: True if the movement path crosses the interference line
    """
    def ccw(A, B, C):
        """Counter-clockwise test for three points"""
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    # Two line segments intersect if the endpoints are on opposite sides of each other
    return ccw(pos1, line_start, line_end) != ccw(pos2, line_start, line_end) and \
           ccw(pos1, pos2, line_start) != ccw(pos1, pos2, line_end)

# ============================================================================
# MAIN SIMULATION CLASS
# ============================================================================

class MASSimulation:
    """
    Orchestrates the Multi-Agent System simulation based on cooperative game theory.
    
    This class implements a sophisticated multi-agent coordination system where:
    - Agents must maintain formation while moving to targets
    - Agents avoid collisions with each other and obstacles
    - Agents form coalitions when facing external threats
    - Different scenarios test various navigation challenges
    """
    
    def __init__(self, use_forward_prediction=False, scenario_type="formation"):
        """
        Initialize the simulation with all parameters and agent configurations.
        
        Args:
            use_forward_prediction (bool): Whether agents predict future adversary positions
            scenario_type (str): Type of scenario ("formation", "interference", "obstacle")
        """
        # ==================== SIMULATION PARAMETERS ====================
        self.n_agents = 9  # Total number of agents (3x3 grid)
        self.rho = 7.0     # Desired distance between connected agents in formation
        self.safe_distance = 10.0  # Minimum safe distance from adversary
        self.delta_a = 1.5  # Step size for agent movement
        self.scenario_type = scenario_type
        
        # ==================== TARGET POSITION SETUP ====================
        # Different scenarios have different target locations
        if scenario_type == "formation":
            initial_center = np.array([12.0, 12.0])  # Center of initial formation
            self.target_position = initial_center + np.array([30.0, 0.0])  # Move right
        elif scenario_type == "interference" or scenario_type == "obstacle":
            self.target_position = np.array([80.0, 40.0])  # Far target requiring navigation
            
        # ==================== PREDICTION SETTINGS ====================
        self.use_forward_prediction = use_forward_prediction
        self.c = 1e-6  # Small constant to prevent division by zero in loss functions
        
        # ==================== INTERFERENCE LINE DEFINITION ====================
        # Defines the forbidden path that agents cannot cross in interference scenario
        self.interference_line_start = np.array([50.0, 35.0])
        self.interference_line_end = np.array([20.0, 0.0])
        self.has_crossed_line = False  # Track if any agent violates this constraint
        
        # ==================== COLLISION TRACKING ====================
        self.exact_collision_count = 0      # Count of agents at identical positions
        self.proximity_violation_count = 0  # Count of agents too close together
        
        # ==================== PREDICTION PARAMETERS ====================
        # For forward prediction of adversary movement
        self.prediction_steps = 10  # How many steps to predict ahead
        # Weights decrease over time - near future more important than far future
        self.prediction_weights = np.array([0.25, 0.20, 0.15, 0.12, 0.09, 0.07, 0.05, 0.03, 0.02, 0.02])

        # ==================== INITIALIZE AGENTS AND ADVERSARY ====================
        self._initialize_agents_and_adversary()
        self._define_communication_topology()
        
        # ==================== LOSS TRACKING ====================
        # Track different types of losses over time for analysis
        self.step_losses = []
        self.detailed_losses = {
            'intra_group': [],   # Losses from formation maintenance and target reaching
            'coalition': [],     # Losses from coalition coordination
            'external': [],      # Losses from adversary proximity and obstacles
            'total': []         # Sum of all loss types
        }

        # ==================== ACTION SPACE ====================
        # Available actions for each agent at each time step
        self.actions = {
            'stay': np.array([0, 0]),                    # No movement
            'up': np.array([0, self.delta_a]),           # Move up
            'down': np.array([0, -self.delta_a]),        # Move down
            'left': np.array([-self.delta_a, 0]),        # Move left
            'right': np.array([self.delta_a, 0])         # Move right
        }

    def _initialize_agents_and_adversary(self):
        """
        Create agents in a 3x3 grid formation and initialize the adversary.
        
        The agents start in a structured formation that they must maintain
        while moving toward their target.
        """
        self.agents = []
        
        # Create 3x3 grid of initial positions
        initial_positions = [
            [5, 5],   [12, 5],   [19, 5],    # Bottom row
            [5, 12],  [12, 12],  [19, 12],   # Middle row  
            [5, 19],  [12, 19],  [19, 19]    # Top row
        ]
        
        # Create agent objects
        for i in range(self.n_agents):
            self.agents.append(Agent(agent_id=i, position=initial_positions[i]))
        
        # Initialize adversary with quadratic trajectory parameters
        adversary_params = {
            'a': 0.01,    # Quadratic coefficient
            'b': 0.5,     # Linear coefficient  
            'c': 20,      # Constant term
            'x_vel': 2.0  # Horizontal velocity
        }
        self.adversary = Adversary(initial_position=[10, 20], trajectory_params=adversary_params)
        self.adversary.update_position()  # Initialize first position

    def _define_communication_topology(self):
        """
        Define which agents can communicate with each other.
        
        Creates an adjacency matrix where agents can only communicate
        with their immediate neighbors in the 3x3 grid (4-connectivity).
        This simulates limited communication range in real systems.
        """
        self.adj_matrix = np.zeros((self.n_agents, self.n_agents))
        
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                # Convert linear index to 2D grid coordinates
                xi, yi = i % 3, i // 3  # Agent i's position in 3x3 grid
                xj, yj = j % 3, j // 3  # Agent j's position in 3x3 grid
                
                # Connect agents that are immediate neighbors (Manhattan distance = 1)
                if abs(xi - xj) + abs(yi - yj) == 1:
                    self.adj_matrix[i, j] = self.adj_matrix[j, i] = 1

    def _predict_adversary_pos(self):
        """
        Predict future positions of the adversary using polynomial extrapolation.
        
        Uses the adversary's movement history to fit a polynomial and predict
        where it will be in the next several time steps. This allows agents
        to plan ahead and avoid future collisions.
        
        Returns:
            list: Predicted future positions as [x, y] arrays
        """
        history = np.array(self.adversary.history)
        
        # Need at least 3 points for meaningful prediction
        if len(history) < 3:
             return [self.adversary.pos] * self.prediction_steps
        
        # Fit polynomial to x and y coordinates separately
        degree = min(3, len(history) - 1)  # Use 3rd degree or less if insufficient data
        t = np.arange(len(history))  # Time indices
        
        # Fit polynomials to x and y trajectories
        fx = np.poly1d(np.polyfit(t, history[:, 0], degree))  # x-coordinate polynomial
        fy = np.poly1d(np.polyfit(t, history[:, 1], degree))  # y-coordinate polynomial

        # Generate future time points and predict positions
        future_t = np.arange(len(history), len(history) + self.prediction_steps)
        return [np.array([fx(t_val), fy(t_val)]) for t_val in future_t]

    # ============================================================================
    # LOSS FUNCTION COMPONENTS
    # ============================================================================
    # The simulation uses a multi-objective loss function with three components:
    # 1. Intra-group loss: Formation maintenance and target reaching
    # 2. Coalition loss: Coordination within coalitions
    # 3. External loss: Adversary avoidance and obstacle navigation

    def _calculate_intra_group_loss(self, agent_id, next_pos, agent_next_states):
        """
        Calculate loss related to formation maintenance and target reaching.
        
        This loss function encourages agents to:
        1. Avoid collisions with other agents (very high penalty)
        2. Maintain desired distances to connected neighbors
        3. Move toward the target position
        
        Args:
            agent_id (int): ID of the agent being evaluated
            next_pos (array): Proposed next position for this agent
            agent_next_states (dict): Proposed positions for all agents
            
        Returns:
            float: Intra-group loss value
        """
        loss = 0.0
        
        # COLLISION AVOIDANCE: Severe penalty for being too close to other agents
        for j in range(self.n_agents):
            if j != agent_id:
                distance = np.linalg.norm(next_pos - agent_next_states[j])
                if distance < 2.5:  # Minimum safe distance between agents
                    # Cubic penalty that grows very large as agents get closer
                    loss += (2.5 - distance)**3 * 1000
        
        # FORMATION MAINTENANCE: Maintain desired distances to connected neighbors
        for j in range(self.n_agents):
            if self.adj_matrix[agent_id, j] == 1:  # If agents are connected in topology
                actual_distance = np.linalg.norm(next_pos - agent_next_states[j])
                # Penalty for deviation from desired formation distance
                loss += abs(actual_distance - self.rho)
        
        # TARGET REACHING: Move toward the target position
        if self.target_position is not None:
            target_distance = np.linalg.norm(next_pos - self.target_position)
            loss += target_distance * 15.0  # Weight factor for target attraction
            
        return loss

    def _calculate_coalition_loss(self, agent_id, next_pos, agent_next_states, coalition_adj_matrix):
        """
        Calculate loss related to coalition coordination.
        
        When agents form coalitions to deal with external threats, they need
        additional coordination beyond the basic formation constraints.
        
        Args:
            agent_id (int): ID of the agent being evaluated
            next_pos (array): Proposed next position for this agent
            agent_next_states (dict): Proposed positions for all agents
            coalition_adj_matrix (array): Adjacency matrix for coalition connections
            
        Returns:
            float: Coalition coordination loss
        """
        loss = 0.0
        
        # Coordinate with other coalition members
        for j in range(self.n_agents):
            if coalition_adj_matrix[agent_id, j] == 1:  # If connected in coalition
                coalition_distance = np.linalg.norm(next_pos - agent_next_states[j])
                # Penalty for deviation from desired coalition distance
                loss += (coalition_distance - self.rho)
                
        return loss

    def _calculate_external_loss(self, agent_pos, adversary_pos_list, current_pos=None):
        """
        Calculate loss related to external threats and obstacles.
        
        This is the most complex loss component, handling:
        1. Adversary avoidance (with optional prediction)
        2. Interference line avoidance
        3. Obstacle zone avoidance
        
        Args:
            agent_pos (array): Position being evaluated
            adversary_pos_list (list): Current and predicted adversary positions
            current_pos (array): Current agent position (for line crossing detection)
            
        Returns:
            float: External threat loss value
        """
        base_loss = 0.0
        
        # ==================== ADVERSARY AVOIDANCE ====================
        if not self.use_forward_prediction:
            # Simple proximity-based avoidance
            distance_to_adversary = np.linalg.norm(agent_pos - adversary_pos_list[0])
            base_loss = 1.0 / (distance_to_adversary + self.c)  # Inverse distance penalty
        else:
            # Sophisticated prediction-based avoidance
            losses = []
            for pos in adversary_pos_list[:self.prediction_steps + 1]:
                distance = np.linalg.norm(agent_pos - pos)
                losses.append(1.0 / (distance + self.c))
            
            # Weight losses by time - nearer future more important
            for k, loss in enumerate(losses):
                if k < len(self.prediction_weights):
                    base_loss += self.prediction_weights[k] * loss
        
        enhanced_loss = base_loss
        
        # ==================== SCENARIO-SPECIFIC CONSTRAINTS ====================
        
        if self.scenario_type == "interference":
            # INTERFERENCE LINE CONSTRAINT
            # Agents must not cross the forbidden interference path
            if current_pos is not None:
                if crosses_interference_line(current_pos, agent_pos, 
                                           self.interference_line_start, 
                                           self.interference_line_end):
                    enhanced_loss += 100000  # Extremely high penalty for crossing
            
            # Additional penalty for getting too close to the line (if using prediction)
            if self.use_forward_prediction:
                line_distance = point_to_line_distance(agent_pos, 
                                                     self.interference_line_start, 
                                                     self.interference_line_end)
                if line_distance < 15.0:
                    enhanced_loss += (15.0 - line_distance)**2 / 50

        elif self.scenario_type == "obstacle":
            # OBSTACLE AVOIDANCE CONSTRAINT
            obstacle_center = np.array([60, 35])  # Fixed obstacle location
            obstacle_distance = np.linalg.norm(agent_pos - obstacle_center)
            
            # Different danger zones based on prediction capability
            danger_radius = 15.0 if self.use_forward_prediction else 10.0
            if obstacle_distance < danger_radius:
                enhanced_loss += (obstacle_distance - danger_radius)**2 / 100
            
            # Absolute prohibition zone (only when using prediction)
            if self.use_forward_prediction and obstacle_distance < 8.5:
                enhanced_loss += (8.5 - obstacle_distance)**4 * 100
        
        return enhanced_loss

    def _determine_coalition_membership(self, coalition_info, agent_next_states):
        """
        Determine which agents should coordinate as a coalition.
        
        Coalitions form when agents face external threats that require
        coordinated response beyond normal formation maintenance.
        
        Args:
            coalition_info (dict): Information about potential coalition
            agent_next_states (dict): Proposed next positions for all agents
            
        Returns:
            tuple: (coalition_adj_matrix, signals) for coordination
        """
        coalition_adj_matrix = np.zeros((self.n_agents, self.n_agents))
        signals = np.zeros(self.n_agents)
        
        if not coalition_info['active']:
            return coalition_adj_matrix, signals
        
        # Create connections between all coalition members
        for i in range(self.n_agents):
            if coalition_info['members'][i]:
                for j in range(i + 1, self.n_agents):
                    if coalition_info['members'][j]:
                        coalition_adj_matrix[i, j] = coalition_adj_matrix[j, i] = 1
        
        # Generate coordination signals for coalition members
        for i in range(self.n_agents):
            if coalition_info['members'][i]:
                # Calculate both individual and coalition losses
                ri_intra = self._calculate_intra_group_loss(i, agent_next_states[i], agent_next_states)
                rc_coalition = self._calculate_coalition_loss(i, agent_next_states[i], 
                                                            agent_next_states, coalition_adj_matrix)
                
                # Activate coalition coordination if needed
                if rc_coalition > 0:
                    signals[i] = 1
                    
        return coalition_adj_matrix, signals

    # ============================================================================
    # SIMULATION EXECUTION
    # ============================================================================

    def run_step(self, step_count):
        """
        Execute one step of the simulation.
        
        This is the main simulation loop that:
        1. Updates adversary position
        2. Determines coalition formation
        3. Computes optimal actions for each agent
        4. Updates agent positions
        5. Checks for constraint violations
        6. Records performance metrics
        
        Args:
            step_count (int): Current simulation step number
        """
        # ==================== UPDATE ADVERSARY ====================
        self.adversary.update_position()
        
        # Get current and predicted adversary positions
        predicted_adv_pos = [self.adversary.pos]
        if self.use_forward_prediction:
            predicted_adv_pos.extend(self._predict_adversary_pos())

        # ==================== COALITION FORMATION ====================
        # Determine which agents should form coalitions based on external threat level
        min_external_loss = float('inf')
        coalition_leader = None
        potential_coalition_members = np.zeros(self.n_agents, dtype=bool)
        
        for i, agent in enumerate(self.agents):
            external_loss = self._calculate_external_loss(agent.pos, predicted_adv_pos, agent.pos)
            
            # If external threat is significant, consider coalition membership
            if external_loss > 1.0/self.safe_distance:
                potential_coalition_members[i] = True
                
                # Agent with lowest external loss becomes coalition leader
                if external_loss < min_external_loss:
                    min_external_loss = external_loss
                    coalition_leader = i

        # Create coalition information structure
        coalition_info = {
            'active': coalition_leader is not None,
            'leader': coalition_leader,
            'members': potential_coalition_members
        }

        # ==================== ACTION SELECTION ====================
        # Each agent selects the action that minimizes their total loss
        agent_actions = {}
        current_agent_states = {i: agent.pos for i, agent in enumerate(self.agents)}
        new_positions = {}

        # Track losses for this step
        step_intra_loss = 0.0
        step_coalition_loss = 0.0
        step_external_loss = 0.0

        for i, agent in enumerate(self.agents):
            best_action = 'stay'
            min_total_loss = float('inf')
            
            # Evaluate each possible action
            for action_name, action_vec in self.actions.items():
                # Calculate proposed next position (bounded by simulation area)
                next_pos = np.clip(agent.pos + action_vec, 0, 100)
                
                # Create temporary state with this agent's proposed position
                temp_next_states = {k: (next_pos if k == i else p) 
                                  for k, p in current_agent_states.items()}
                
                # Calculate all loss components
                external_loss = self._calculate_external_loss(next_pos, predicted_adv_pos, 
                                                            current_pos=agent.pos)
                intra_loss = self._calculate_intra_group_loss(i, next_pos, temp_next_states)
                
                # Determine coalition coordination requirements
                coalition_adj_matrix, signals = self._determine_coalition_membership(
                    coalition_info, temp_next_states)
                coalition_loss = self._calculate_coalition_loss(i, next_pos, 
                                                              temp_next_states, coalition_adj_matrix)
                
                # Get coalition coordination signal
                signal = signals[i] if (coalition_info['active'] and 
                                      coalition_info['members'][i]) else 0
                
                # Combine all loss components with appropriate weights
                total_loss = intra_loss + signal * coalition_loss + external_loss * 50

                # Select action with minimum total loss
                if total_loss < min_total_loss:
                    min_total_loss = total_loss
                    best_action = action_name
            
            # Store the selected position
            new_positions[i] = np.clip(agent.pos + self.actions[best_action], 0, 100)

        # ==================== CALCULATE ACTUAL LOSSES ====================
        # Calculate actual losses for the chosen positions
        for i, agent in enumerate(self.agents):
            temp_next_states = {j: new_positions[j] for j in range(self.n_agents)}
            
            intra_loss = self._calculate_intra_group_loss(i, new_positions[i], temp_next_states)
            external_loss = self._calculate_external_loss(new_positions[i], predicted_adv_pos, 
                                                         current_pos=agent.pos)
            coalition_adj_matrix, signals = self._determine_coalition_membership(
                coalition_info, temp_next_states)
            coalition_loss = self._calculate_coalition_loss(i, new_positions[i], 
                                                          temp_next_states, coalition_adj_matrix)
            
            step_intra_loss += intra_loss
            step_external_loss += external_loss
            step_coalition_loss += coalition_loss

        # ==================== CONSTRAINT VIOLATION CHECKING ====================
        # Check for interference line violations
        if self.scenario_type == "interference":
            for i, agent in enumerate(self.agents):
                if crosses_interference_line(agent.pos, new_positions[i], 
                                           self.interference_line_start, 
                                           self.interference_line_end):
                    self.has_crossed_line = True
                    print(f"🚩 VIOLATION: Agent {i+1} crossed the interference line at step {step_count}.")

        # ==================== UPDATE AGENT POSITIONS ====================
        for i, agent in enumerate(self.agents):
            agent.update_position(new_positions[i])
        
        # ==================== RECORD PERFORMANCE METRICS ====================
        step_total_loss = step_intra_loss + step_coalition_loss + step_external_loss
        self.detailed_losses['intra_group'].append(step_intra_loss)
        self.detailed_losses['coalition'].append(step_coalition_loss)
        self.detailed_losses['external'].append(step_external_loss)
        self.detailed_losses['total'].append(step_total_loss)
        
        # Check for collisions and proximity violations
        self.check_exact_collisions(step_count)

    # ============================================================================
    # SUCCESS CONDITION CHECKING
    # ============================================================================

    def check_target_reached(self, tolerance=5.0):
        """
        Check if all agents have reached the target position.
        
        Args:
            tolerance (float): Maximum allowed distance from target
            
        Returns:
            bool: True if all agents are within tolerance of target
        """
        if self.target_position is None:
            return False
        
        return all(np.linalg.norm(agent.pos - self.target_position) <= tolerance 
                  for agent in self.agents)

    def check_formation_constraints(self, tolerance=2.0):
        """
        Check if agents maintain proper formation distances.
        
        Args:
            tolerance (float): Allowed deviation from desired formation distance
            
        Returns:
            bool: True if formation is maintained within tolerance
        """
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if self.adj_matrix[i, j] == 1:  # If agents should be connected
                    actual_distance = np.linalg.norm(self.agents[i].pos - self.agents[j].pos)
                    if abs(actual_distance - self.rho) > tolerance:
                        return False
        return True

    def check_forbidden_areas(self):
        """
        Check if agents have violated any forbidden area constraints.
        
        Returns:
            bool: True if no forbidden areas have been violated
        """
        if self.scenario_type == "interference" and self.has_crossed_line:
            return False
        elif self.scenario_type == "obstacle":
            # Only check for violations if prediction is enabled
            if self.use_forward_prediction:
                obstacle_center = np.array([60, 35])
                obstacle_radius = 8.5
                for agent in self.agents:
                    if np.linalg.norm(agent.pos - obstacle_center) < obstacle_radius:
                        return False
        return True

    def check_agent_proximity_collisions(self, min_distance=2.0):
        """
        Check if any agents are too close to each other.
        
        Args:
            min_distance (float): Minimum allowed distance between agents
            
        Returns:
            bool: True if no agents are too close
        """
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.agents[i].pos - self.agents[j].pos) < min_distance:
                    return False
        return True
    
    def check_exact_collisions(self, step_count):
        """
        Check for and report exact collisions and proximity violations.
        
        Args:
            step_count (int): Current simulation step for reporting
        """
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                # Check for exact position overlap
                if np.array_equal(self.agents[i].pos, self.agents[j].pos):
                    self.exact_collision_count += 1
                    print(f"🚨 EXACT COLLISION: Agents {i+1} and {j+1} at location "
                          f"{self.agents[i].pos} on step {step_count}.")
                else:
                    # Check for proximity violations
                    distance = np.linalg.norm(self.agents[i].pos - self.agents[j].pos)
                    if distance < 2.0:
                        self.proximity_violation_count += 1
                        print(f"🚨 PROXIMITY VIOLATION: Agents {i+1} and {j+1} at locations "
                              f"{self.agents[i].pos} and {self.agents[j].pos} "
                              f"(distance: {distance:.2f}) on step {step_count}.")

    def check_success_conditions(self):
        """
        Check if all success conditions are met.
        
        Returns:
            bool: True if simulation is successful
        """
        return (self.check_target_reached() and 
                self.check_formation_constraints() and 
                self.check_forbidden_areas() and 
                self.check_agent_proximity_collisions())

    def run_simulation(self, max_steps=400):
        """
        Run the complete simulation for the specified number of steps.
        
        Args:
            max_steps (int): Maximum number of simulation steps
            
        Returns:
            tuple: (agents, adversary) final states
        """
        print(f"Starting {self.scenario_type} scenario simulation...")
        
        for step_count in range(1, max_steps + 1):
            self.run_step(step_count)
            
            # Check for early success (primarily for formation tasks)
            if self.scenario_type == "formation" and self.check_success_conditions():
                print(f"✅ SUCCESS! Formation task completed in {step_count} steps.")
                print(f"  - Exact Collisions: {self.exact_collision_count}")
                print(f"  - Proximity Violations: {self.proximity_violation_count}")
                return self.agents, self.adversary

        # Final status check
        success = self.check_success_conditions()
        if success:
            print(f"✅ SUCCESS! All conditions met within {max_steps} steps.")
        else:
            print(f"⚠️ Simulation ended at {max_steps} steps. Final Status:")
            print(f"  - Target Reached: {self.check_target_reached()}")
            print(f"  - No Forbidden Area Violations: {self.check_forbidden_areas()}")

        print(f"  - Exact Collisions: {self.exact_collision_count}")
        print(f"  - Proximity Violations: {self.proximity_violation_count}")
            
        return self.agents, self.adversary

    def get_loss_summary(self):
        """
        Return summary statistics for all loss types.
        
        Returns:
            dict: Comprehensive loss statistics
        """
        if not self.detailed_losses['total']:
            return {"error": "No loss data recorded"}
        
        summary = {}
        for loss_type in ['intra_group', 'coalition', 'external', 'total']:
            data = self.detailed_losses[loss_type]
            summary[loss_type] = {
                'total': sum(data),
                'average': np.mean(data),
                'max': max(data),
                'min': min(data)
            }
        
        return summary

# ============================================================================
# VISUALIZATION CLASS
# ============================================================================

class Visualizer:
    """
    Handles visualization of simulation results.
    
    Creates trajectory plots showing agent paths, obstacles, targets,
    and other scenario-specific elements.
    """
    
    def __init__(self, agents, adversary, title, scenario_type):
        """
        Initialize visualizer with simulation results.
        
        Args:
            agents (list): List of Agent objects with trajectory history
            adversary (Adversary): Adversary object with trajectory history  
            title (str): Plot title
            scenario_type (str): Type of scenario for specialized visualization
        """
        self.agents = agents
        self.adversary = adversary
        self.title = title
        self.scenario_type = scenario_type
        # Generate distinct colors for each agent
        self.colors = plt.cm.jet(np.linspace(0, 1, len(agents)))

    def plot_trajectories(self, subplot_ax):
        """
        Plot agent trajectories for formation and interference scenarios.
        
        Args:
            subplot_ax: Matplotlib axis object for plotting
        """
        # Set up plot
        subplot_ax.set_title(self.title)
        subplot_ax.set_xlabel("X")
        subplot_ax.set_ylabel("Y")
        subplot_ax.set_xlim(0, 100)
        subplot_ax.set_ylim(0, 100)

        # Plot agent trajectories
        for i, agent in enumerate(self.agents):
            hist = np.array(agent.history)
            subplot_ax.plot(hist[:, 0], hist[:, 1], 'o-', color=self.colors[i], 
                          markersize=5, linewidth=1.5)

        # Add scenario-specific elements
        target_pos = None
        if self.scenario_type == "interference":
            # Draw interference line
            subplot_ax.plot([50, 20], [35, 0], 'r--', linewidth=2, alpha=0.9, 
                          label="Interference Path")
            target_pos = np.array([80.0, 40.0])
        elif self.scenario_type == "formation":
            target_pos = np.array([12.0, 12.0]) + np.array([30.0, 0.0])
        
        # Mark target position
        if target_pos is not None:
            subplot_ax.plot(target_pos[0], target_pos[1], 'g*', markersize=15, label="Target")
        
        subplot_ax.grid(True, linestyle='--', alpha=0.6)
        subplot_ax.legend(fontsize='small')

    def plot_obstacle_avoidance_scenario(self, subplot_ax):
        """
        Plot specialized visualization for obstacle avoidance scenario.
        
        Shows the obstacle zone, agent trajectories, and goal position.
        
        Args:
            subplot_ax: Matplotlib axis object for plotting
        """
        # Set up plot
        subplot_ax.set_title(self.title)
        subplot_ax.set_xlabel("X")
        subplot_ax.set_ylabel("Y") 
        subplot_ax.set_xlim(0, 100)
        subplot_ax.set_ylim(0, 100)

        # Define obstacle and goal
        obstacle_center = np.array([60, 35])
        goal_pos = np.array([80, 40])
        obstacle_radius = 8.0
        
        # Draw obstacle zone
        circle = patches.Circle(obstacle_center, obstacle_radius, color='purple', 
                              alpha=0.4, label=f"Obstacle Zone")
        subplot_ax.add_patch(circle)
        subplot_ax.plot(obstacle_center[0], obstacle_center[1], 'ro', markersize=8, 
                       label="Obstacle Center")
        
        # Mark goal
        subplot_ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=20, label="Goal")
        
        # Plot agent trajectories
        for i, agent in enumerate(self.agents):
            trajectory = np.array(agent.history)
            subplot_ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=self.colors[i], 
                          markersize=5, linewidth=1.5)
        
        subplot_ax.grid(True, linestyle='--', alpha=0.6)
        subplot_ax.legend(fontsize='small')

# ============================================================================
# ANALYSIS AND REPORTING FUNCTIONS
# ============================================================================

def print_loss_analysis(scenario_results):
    """
    Print detailed loss analysis for all scenarios.
    
    Provides comprehensive breakdown of loss components across different
    scenarios and prediction settings.
    
    Args:
        scenario_results (dict): Dictionary mapping scenario names to results
    """
    print("\n" + "="*80)
    print("🎯 DETAILED LOSS ANALYSIS BY SCENARIO")
    print("="*80)
    
    total_losses_by_scenario = {}
    average_losses_by_scenario = {}
    
    for scenario_name, (agents, adversary, simulation) in scenario_results.items():
        print(f"\n📊 {scenario_name.upper()}")
        print("-" * 60)
        
        loss_summary = simulation.get_loss_summary()
        if "error" in loss_summary:
            print(f"   ⚠️ {loss_summary['error']}")
            continue
            
        # Print detailed breakdown for each loss type
        loss_types = [
            ('INTRA-GROUP', 'intra_group'),
            ('COALITION', 'coalition'),
            ('EXTERNAL', 'external'),
            ('TOTAL COMBINED', 'total')
        ]
        
        for display_name, loss_key in loss_types:
            data = loss_summary[loss_key]
            print(f"   🔹 {display_name} LOSSES:")
            print(f"      Total: {data['total']:.4f}")
            print(f"      Average per step: {data['average']:.4f}")
            print(f"      Max: {data['max']:.4f}")
            print(f"      Min: {data['min']:.4f}")
        
        # Store for summary tables
        total_losses_by_scenario[scenario_name] = loss_summary['total']['total']
        average_losses_by_scenario[scenario_name] = loss_summary['total']['average']
    
    # Print comparative summary tables
    print("\n" + "="*80)
    print("🎯 TOTAL LOSSES BY SCENARIO (Ranked):")
    print("="*80)
    for scenario, total_loss in sorted(total_losses_by_scenario.items(), key=lambda x: x[1]):
        print(f"   {scenario:25}: {total_loss:12.4f}")
    
    print("\n" + "="*80)
    print("🎯 AVERAGE LOSSES BY SCENARIO (Ranked per step):")
    print("="*80)
    for scenario, avg_loss in sorted(average_losses_by_scenario.items(), key=lambda x: x[1]):
        print(f"   {scenario:25}: {avg_loss:12.4f}")

def plot_loss_progression(scenario_results):
    """
    Create comprehensive bar charts showing loss analysis.
    
    Generates two figures:
    1. Total vs Average losses comparison
    2. Detailed breakdown by loss category
    
    Args:
        scenario_results (dict): Dictionary mapping scenario names to results
    """
    # Use jet colormap for consistency with trajectory plots
    n_scenarios = 6  # 3 scenario types × 2 prediction settings
    jet_colors = plt.cm.jet(np.linspace(0, 1, n_scenarios))
    
    # ==================== FIGURE 1: TOTAL vs AVERAGE COMPARISON ====================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig1.suptitle("Loss Analysis: Total vs Average by Scenario and Prediction Setting", 
                  fontsize=16, y=0.95)
    
    # Prepare data for bar plots
    scenarios = []
    total_losses = []
    avg_losses = []
    colors = []
    
    scenario_types = ["Formation", "Interference", "Obstacle"]
    prediction_settings = ["No Pred", "With Pred"]
    
    color_idx = 0
    for scenario_type in scenario_types:
        for pred_setting in prediction_settings:
            scenario_name = f"{scenario_type} ({pred_setting})"
            if scenario_name in scenario_results:
                _, _, simulation = scenario_results[scenario_name]
                loss_summary = simulation.get_loss_summary()
                
                scenarios.append(f"{scenario_type}\n({pred_setting})")
                total_losses.append(loss_summary['total']['total'])
                avg_losses.append(loss_summary['total']['average'])
                colors.append(jet_colors[color_idx])
                color_idx += 1
    
    # Plot total losses
    bars1 = ax1.bar(scenarios, total_losses, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.8)
    ax1.set_title("Total Accumulated Losses", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Total Loss Value", fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on total loss bars
    for bar, value in zip(bars1, total_losses):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot average losses
    bars2 = ax2.bar(scenarios, avg_losses, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.8)
    ax2.set_title("Average Loss per Step", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Average Loss Value", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on average loss bars
    for bar, value in zip(bars2, avg_losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Create legend
    legend_elements = []
    color_idx = 0
    for scenario_type in scenario_types:
        for pred_setting in prediction_settings:
            scenario_name = f"{scenario_type} ({pred_setting})"
            if scenario_name in scenario_results:
                legend_elements.append(
                    plt.Rectangle((0,0),1,1, facecolor=jet_colors[color_idx], alpha=0.8, 
                                label=f'{scenario_type} ({pred_setting})'))
                color_idx += 1
    
    fig1.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
               ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.1, 1, 0.93])
    plt.show()
    
    # ==================== FIGURE 2: DETAILED CATEGORY BREAKDOWN ====================
    fig2, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig2.suptitle("Loss Breakdown by Category: Intra-group, Coalition, and External", 
                  fontsize=16, y=0.95)
    
    loss_categories = ['intra_group', 'coalition', 'external']
    category_titles = ['Intra-group Losses', 'Coalition Losses', 'External Losses']
    
    for cat_idx, (category, title) in enumerate(zip(loss_categories, category_titles)):
        scenarios_cat = []
        losses_cat = []
        colors_cat = []
        
        color_idx = 0
        for scenario_type in scenario_types:
            for pred_setting in prediction_settings:
                scenario_name = f"{scenario_type} ({pred_setting})"
                if scenario_name in scenario_results:
                    _, _, simulation = scenario_results[scenario_name]
                    loss_summary = simulation.get_loss_summary()
                    
                    scenarios_cat.append(f"{scenario_type}\n({pred_setting})")
                    losses_cat.append(loss_summary[category]['total'])
                    colors_cat.append(jet_colors[color_idx])
                    color_idx += 1
        
        # Create bars for this category
        bars = axes[cat_idx].bar(scenarios_cat, losses_cat, color=colors_cat, alpha=0.8, 
                               edgecolor='black', linewidth=0.8)
        axes[cat_idx].set_title(title, fontsize=14, fontweight='bold')
        axes[cat_idx].set_ylabel("Total Loss Value", fontsize=12)
        axes[cat_idx].tick_params(axis='x', rotation=45)
        axes[cat_idx].grid(True, axis='y', alpha=0.3)
        
        # Add value labels (only for meaningful values)
        for bar, value in zip(bars, losses_cat):
            height = bar.get_height()
            if height > 0:  # Only show label if there's a meaningful value
                axes[cat_idx].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add legend to detailed breakdown
    fig2.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
               ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.1, 1, 0.93])
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Simulation parameters
    MAX_STEPS = 400        # Standard simulation length
    INTERFERENCE_STEPS = 600  # Longer time for interference scenarios
    
    print("🚀 Multi-Agent System: STRICT Safety & Obstacle Rules")
    print("🔴 Obstacle (No Pred): Agents MAY enter the purple zone.")
    print("🔴 Obstacle (With Pred): Agents MUST AVOID the purple zone.")
    print("=" * 80)

    # Define all simulation scenarios
    # Each scenario tests different aspects of multi-agent coordination
    scenarios = {
        "Formation (No Pred)": {"pred": False, "type": "formation", "steps": MAX_STEPS},
        "Formation (With Pred)": {"pred": True, "type": "formation", "steps": MAX_STEPS},
        "Interference (No Pred)": {"pred": False, "type": "interference", "steps": INTERFERENCE_STEPS},
        "Interference (With Pred)": {"pred": True, "type": "interference", "steps": INTERFERENCE_STEPS},
        "Obstacle (No Pred)": {"pred": False, "type": "obstacle", "steps": MAX_STEPS},
        "Obstacle (With Pred)": {"pred": True, "type": "obstacle", "steps": MAX_STEPS},
    }
    
    # Run all scenarios and collect results
    results = {}
    scenario_results = {}
    
    for name, params in scenarios.items():
        print(f"\n--- Running: {name} ---")
        sim = MASSimulation(use_forward_prediction=params["pred"], 
                           scenario_type=params["type"])
        agents, adversary = sim.run_simulation(params["steps"])
        results[name] = (agents, adversary)
        scenario_results[name] = (agents, adversary, sim)  # Include simulation for loss data

    # Generate comprehensive analysis
    print_loss_analysis(scenario_results)

    # ==================== TRAJECTORY VISUALIZATION ====================
    # Create 2x3 grid showing all scenarios
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("MAS Scenarios: Formation | Interference | Obstacle", fontsize=16, y=0.95)
    
    # Organize scenarios by type for column arrangement
    scenario_types = ["formation", "interference", "obstacle"]
    
    for col, scenario_type in enumerate(scenario_types):
        # No prediction (top row)
        no_pred_name = f"{scenario_type.capitalize()} (No Pred)"
        agents, adversary = results[no_pred_name]
        viz = Visualizer(agents, adversary, no_pred_name, scenario_type)
        
        if scenario_type == "obstacle":
            viz.plot_obstacle_avoidance_scenario(axes[0, col])
        else:
            viz.plot_trajectories(axes[0, col])
        
        # With prediction (bottom row)
        with_pred_name = f"{scenario_type.capitalize()} (With Pred)"
        agents, adversary = results[with_pred_name]
        viz = Visualizer(agents, adversary, with_pred_name, scenario_type)
        
        if scenario_type == "obstacle":
            viz.plot_obstacle_avoidance_scenario(axes[1, col])
        else:
            viz.plot_trajectories(axes[1, col])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    # ==================== LOSS ANALYSIS VISUALIZATION ====================
    plot_loss_progression(scenario_results)

    print("\n🎯 All simulations completed with detailed loss analysis.")
