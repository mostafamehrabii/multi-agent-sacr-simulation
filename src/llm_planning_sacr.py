# %% [markdown]
# # Complete Multi-Agent System with SACR Communication Protocol + Planning + Comprehensive Metrics
# Enhanced implementation with planning capabilities and complete metrics visualization

# %%
# Multi-Agent System with Comprehensive SACR Communication Protocol + Planning + Enhanced Metrics
# This implementation includes planning integration with comprehensive metrics tracking and visualization

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='IPython')

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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import time
import json

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

class HyperParameters:
    """Centralized hyperparameters for easy access and modification"""
    
    # Simulation parameters
    N_AGENTS = 9
    RHO = 7.0  # Formation distance
    SAFE_DISTANCE = 10.0
    DELTA_A = 1.5  # Movement step size
    C = 1e-6  # Small constant for loss calculations
    
    # Planning parameters (Full simulation planning)
    PLANNING_SUCCESS_THRESHOLD = 0.8  # Minimum success rate to consider planning effective
    
    # Safety zones
    COLLISION_RADIUS = 2.0
    WARNING_RADIUS = 4.0  
    COMMUNICATION_RADIUS = 15.0
    
    # Simulation limits
    MAX_STEPS = 400
    INTERFERENCE_STEPS = 600
    
    # Prediction parameters
    PREDICTION_STEPS = 10
    PREDICTION_WEIGHTS = np.array([0.25, 0.20, 0.15, 0.12, 0.09, 0.07, 0.05, 0.03, 0.02, 0.02])

# ============================================================================
# COMMUNICATION PROTOCOL INFRASTRUCTURE
# ============================================================================

class MessageType(Enum):
    """Types of messages in the SACR protocol"""
    MOVEMENT_INTENT = "movement_intent"
    PROXIMITY_WARNING = "proximity_warning"
    YIELD_REQUEST = "yield_request"
    YIELD_ACKNOWLEDGE = "yield_acknowledge"
    CLEAR_SIGNAL = "clear_signal"
    EMERGENCY_STOP = "emergency_stop"
    NEGOTIATION_COUNTER = "negotiation_counter"

@dataclass
class Message:
    """Message structure for inter-agent communication"""
    sender_id: int
    receiver_id: int
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: int
    priority_level: int = 0

@dataclass
class SafetyZone:
    """Defines safety zones around agents"""
    collision_radius: float = HyperParameters.COLLISION_RADIUS
    warning_radius: float = HyperParameters.WARNING_RADIUS
    communication_radius: float = HyperParameters.COMMUNICATION_RADIUS

@dataclass
class PriorityMetrics:
    """Metrics used for priority calculation"""
    distance_to_target: float
    agent_id: int
    current_velocity: float
    coalition_member: bool
    emergency_status: bool = False

class CommunicationManager:
    """
    Advanced communication manager implementing the SACR protocol.
    Handles all inter-agent messaging, conflict detection, and resolution.
    """
    
    def __init__(self, n_agents: int, safety_zone: SafetyZone):
        self.n_agents = n_agents
        self.safety_zone = safety_zone
        self.message_queues = {i: deque() for i in range(n_agents)}
        self.sent_messages = {i: [] for i in range(n_agents)}
        self.active_negotiations = {}
        self.agent_statuses = {}
        self.communication_topology = np.ones((n_agents, n_agents)) - np.eye(n_agents)
        self.step_count = 0
        
        # Enhanced statistics tracking
        self.stats = {
            'messages_sent': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'negotiations_initiated': 0,
            'emergency_stops': 0,
            'successful_yields': 0,
            'communication_overhead': 0,
            'behavior_changes_due_to_communication': 0
        }

    def update_communication_topology(self, agent_positions: Dict[int, np.ndarray]):
        """Update communication links based on current agent positions and communication range"""
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                distance = np.linalg.norm(agent_positions[i] - agent_positions[j])
                can_communicate = distance <= self.safety_zone.communication_radius
                self.communication_topology[i, j] = can_communicate
                self.communication_topology[j, i] = can_communicate

    def calculate_priority(self, agent_id: int, metrics: PriorityMetrics) -> float:
        """Calculate comprehensive priority score for conflict resolution."""
        priority_score = 0.0
        
        if metrics.distance_to_target > 0:
            priority_score += 100.0 / (metrics.distance_to_target + 1.0)
        
        priority_score += (self.n_agents - metrics.agent_id) * 0.1
        priority_score += metrics.current_velocity * 2.0
        
        if metrics.coalition_member:
            priority_score += 10.0
            
        if metrics.emergency_status:
            priority_score += 1000.0
            
        return priority_score

    def send_message(self, sender_id: int, receiver_id: int, 
                    message_type: MessageType, content: Dict[str, Any]) -> bool:
        """Send a message between agents if communication link exists"""
        if not self.communication_topology[sender_id, receiver_id]:
            return False
            
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=self.step_count
        )
        
        self.message_queues[receiver_id].append(message)
        self.sent_messages[sender_id].append(message)
        self.stats['messages_sent'] += 1
        self.stats['communication_overhead'] += 1
        return True

    def get_messages(self, agent_id: int) -> List[Message]:
        """Retrieve and clear all pending messages for an agent"""
        messages = list(self.message_queues[agent_id])
        self.message_queues[agent_id].clear()
        return messages

    def detect_proximity_conflicts(self, agent_positions: Dict[int, np.ndarray], 
                                 intended_moves: Dict[int, np.ndarray]) -> List[Tuple[int, int]]:
        """Predict proximity conflicts based on intended movements."""
        conflicts = []
        
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                future_pos_i = agent_positions[i] + intended_moves[i]
                future_pos_j = agent_positions[j] + intended_moves[j]
                
                future_distance = np.linalg.norm(future_pos_i - future_pos_j)
                
                if future_distance < self.safety_zone.collision_radius:
                    conflicts.append((i, j))
                elif future_distance < self.safety_zone.warning_radius:
                    current_distance = np.linalg.norm(agent_positions[i] - agent_positions[j])
                    if future_distance < current_distance:
                        conflicts.append((i, j))
                    
        if conflicts:
            self.stats['conflicts_detected'] += len(conflicts)
            
        return conflicts

    def initiate_negotiation(self, agent1_id: int, agent2_id: int, 
                           agent_positions: Dict[int, np.ndarray],
                           priority_metrics: Dict[int, PriorityMetrics]) -> Dict[str, Any]:
        """Initiate negotiation between conflicting agents."""
        self.stats['negotiations_initiated'] += 1
        
        priority1 = self.calculate_priority(agent1_id, priority_metrics[agent1_id])
        priority2 = self.calculate_priority(agent2_id, priority_metrics[agent2_id])
        
        negotiation_id = f"{min(agent1_id, agent2_id)}_{max(agent1_id, agent2_id)}_{self.step_count}"
        
        if priority1 > priority2:
            high_priority_agent = agent1_id
            low_priority_agent = agent2_id
            priority_difference = priority1 - priority2
        else:
            high_priority_agent = agent2_id
            low_priority_agent = agent1_id
            priority_difference = priority2 - priority1
            
        negotiation_result = {
            'negotiation_id': negotiation_id,
            'high_priority_agent': high_priority_agent,
            'low_priority_agent': low_priority_agent,
            'priority_scores': {agent1_id: priority1, agent2_id: priority2},
            'priority_difference': priority_difference,
            'resolution_type': 'yield_request',
            'timestamp': self.step_count,
            'urgency_level': 'high' if priority_difference > 50 else 'normal'
        }
        
        self.send_message(
            sender_id=high_priority_agent,
            receiver_id=low_priority_agent,
            message_type=MessageType.YIELD_REQUEST,
            content={
                'negotiation_id': negotiation_id,
                'priority_score': priority1 if high_priority_agent == agent1_id else priority2,
                'reason': 'proximity_conflict',
                'urgency': negotiation_result['urgency_level'],
                'suggested_wait_time': 2 if priority_difference > 50 else 1
            }
        )
        
        self.active_negotiations[negotiation_id] = negotiation_result
        return negotiation_result

    def process_agent_messages(self, agent_id: int, current_position: np.ndarray, 
                             intended_action: str) -> Dict[str, Any]:
        """Process all messages for an agent and return communication-based constraints."""
        messages = self.get_messages(agent_id)
        communication_constraints = {
            'must_yield': False,
            'can_proceed': True,
            'alternative_actions': [],
            'emergency_stop': False,
            'negotiation_active': False,
            'yield_duration': 0,
            'yield_reason': None,
            'received_communication': len(messages) > 0
        }
        
        for message in messages:
            if message.message_type == MessageType.YIELD_REQUEST:
                communication_constraints['must_yield'] = True
                communication_constraints['can_proceed'] = False
                communication_constraints['negotiation_active'] = True
                communication_constraints['yield_duration'] = message.content.get('suggested_wait_time', 1)
                communication_constraints['yield_reason'] = message.content.get('reason', 'unknown')
                
                self.send_message(
                    sender_id=agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.YIELD_ACKNOWLEDGE,
                    content={
                        'negotiation_id': message.content.get('negotiation_id'),
                        'compliance': True,
                        'estimated_delay': communication_constraints['yield_duration']
                    }
                )
                self.stats['successful_yields'] += 1
                
            elif message.message_type == MessageType.PROXIMITY_WARNING:
                communication_constraints['can_proceed'] = False
                communication_constraints['alternative_actions'] = ['stay', 'left', 'right']
                
            elif message.message_type == MessageType.EMERGENCY_STOP:
                communication_constraints['emergency_stop'] = True
                communication_constraints['can_proceed'] = False
                self.stats['emergency_stops'] += 1
                
            elif message.message_type == MessageType.CLEAR_SIGNAL:
                communication_constraints['can_proceed'] = True
                communication_constraints['must_yield'] = False
                
        return communication_constraints

    def step_update(self):
        """Update communication manager for new simulation step"""
        self.step_count += 1
        
        old_negotiations = [nid for nid, neg in self.active_negotiations.items() 
                          if self.step_count - neg['timestamp'] > 5]
        for nid in old_negotiations:
            del self.active_negotiations[nid]

# ============================================================================
# PLANNING SYSTEM
# ============================================================================

class PlanningSystem:
    """Handles full simulation planning request generation and response parsing"""
    
    def __init__(self, simulation, max_steps: int):
        self.simulation = simulation
        self.max_steps = max_steps
        self.planning_success_count = 0
        self.total_planning_attempts = 0
        self.current_plan = None
        self.plan_step = 0
        self.agents_followed_plan_per_step = []

    def generate_full_simulation_planning_request(self) -> str:
        """Generate a structured planning request for the entire simulation"""
        
        # Initial state information
        agent_positions = [agent.pos.tolist() for agent in self.simulation.agents]
        adversary_initial_position = self.simulation.adversary.pos.tolist()
        
        # Scenario-specific information
        scenario_info = {
            "type": self.simulation.scenario_type,
            "target_position": self.simulation.target_position.tolist() if self.simulation.target_position is not None else None,
            "prediction_enabled": self.simulation.use_forward_prediction
        }
        
        if self.simulation.scenario_type == "interference":
            scenario_info["interference_line"] = {
                "start": self.simulation.interference_line_start.tolist(),
                "end": self.simulation.interference_line_end.tolist()
            }
        elif self.simulation.scenario_type == "obstacle":
            scenario_info["obstacle"] = {
                "center": [60, 35],
                "radius": 8.0
            }
        
        # Safety constraints
        safety_constraints = {
            "collision_radius": self.simulation.safety_zone.collision_radius,
            "warning_radius": self.simulation.safety_zone.warning_radius,
            "formation_distance": self.simulation.rho
        }
        
        # Adversary trajectory information
        adversary_info = {
            "initial_position": adversary_initial_position,
            "trajectory_params": {
                "a": 0.01,
                "b": 0.5, 
                "c": 20,
                "x_vel": 2.0
            },
            "special_movement": "Steps 11-15: moves from [50,35] to [20,0]"
        }
        
        request = {
            "simulation_steps": self.max_steps,
            "initial_agent_positions": agent_positions,
            "adversary_trajectory": adversary_info,
            "scenario": scenario_info,
            "safety_constraints": safety_constraints,
            "available_actions": list(self.simulation.actions.keys()),
            "formation_adjacency": self.simulation.adj_matrix.tolist()
        }
        
        prompt = f"""
FULL SIMULATION MULTI-AGENT PLANNING REQUEST

You are a strategic multi-agent planning system. Plan actions for {HyperParameters.N_AGENTS} agents over the ENTIRE simulation of UP TO {self.max_steps} steps.

INITIAL STATE AND CONSTRAINTS:
{json.dumps(request, indent=2)}

OBJECTIVES:
1. All agents reach target position: {scenario_info['target_position']}
2. Maintain formation (adjacent agents ~{safety_constraints['formation_distance']} units apart)
3. Avoid collisions (min distance > {safety_constraints['collision_radius']})
4. Avoid adversary (it follows the given trajectory)
5. Follow scenario-specific constraints

ADVERSARY TRAJECTORY:
- Follows quadratic path: y = 0.01*x² + 0.5*x + 20, moving right by 2.0 units/step
- Special: Steps 11-15, moves linearly from [50,35] to [20,0]
- Predict its position and plan accordingly

FORMATION:
- Agents start in 3x3 grid formation
- Adjacent agents (in adjacency matrix) should maintain ~{safety_constraints['formation_distance']} unit spacing
- Formation should be preserved while moving to target

IMPORTANT NOTES:
- Simulation may end early if formation task completes successfully
- Plan for the full {self.max_steps} steps, but simulation may terminate when objectives are met
- If you provide fewer than {self.max_steps} steps, agents will use reactive behavior for remaining steps
- If you provide more than {self.max_steps} steps, only the first {self.max_steps} will be used

RESPONSE FORMAT (JSON ONLY):
{{
  "plan": {{
    "agent_0": ["action1", "action2", ..., "actionN"],
    "agent_1": ["action1", "action2", ..., "actionN"],
    "agent_2": ["action1", "action2", ..., "actionN"],
    "agent_3": ["action1", "action2", ..., "actionN"],
    "agent_4": ["action1", "action2", ..., "actionN"],
    "agent_5": ["action1", "action2", ..., "actionN"],
    "agent_6": ["action1", "action2", ..., "actionN"],
    "agent_7": ["action1", "action2", ..., "actionN"],
    "agent_8": ["action1", "action2", ..., "actionN"]
  }},
  "confidence": 0.85,
  "reasoning": "Strategic approach explanation"
}}

IMPORTANT:
- Each agent should have the same number of actions (ideally {self.max_steps})
- Available actions: {list(self.simulation.actions.keys())}
- Plan the complete trajectory from start to target
- Consider adversary position at each step
- Maintain formation throughout the movement
- Focus on efficiency - the simulation may end early when objectives are achieved
"""
        return prompt

    def parse_planning_response(self, response: str) -> Optional[Dict]:
        """Parse the full simulation planning response from the language model"""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            plan_data = json.loads(response)
            
            # Validate plan structure
            if 'plan' not in plan_data:
                print("❌ No 'plan' key found in response")
                return None
                
            plan = plan_data['plan']
            
            # Validate all agents have plans
            for i in range(HyperParameters.N_AGENTS):
                agent_key = f"agent_{i}"
                if agent_key not in plan:
                    print(f"❌ Missing plan for {agent_key}")
                    return None
                if not isinstance(plan[agent_key], list):
                    print(f"❌ Plan for {agent_key} is not a list")
                    return None
                
                # Flexible step count validation
                plan_length = len(plan[agent_key])
                if plan_length < self.max_steps:
                    print(f"⚠️  Plan for {agent_key} has {plan_length} steps, expected {self.max_steps}")
                    print(f"   Will use reactive behavior for remaining {self.max_steps - plan_length} steps")
                elif plan_length > self.max_steps:
                    print(f"⚠️  Plan for {agent_key} has {plan_length} steps, expected {self.max_steps}")
                    print(f"   Will use only first {self.max_steps} steps")
                    # Trim the plan to match expected steps
                    plan[agent_key] = plan[agent_key][:self.max_steps]
                    
                # Validate all actions are valid (for the actions we'll actually use)
                actions_to_validate = min(len(plan[agent_key]), self.max_steps)
                for step in range(actions_to_validate):
                    action = plan[agent_key][step]
                    if action not in self.simulation.actions:
                        print(f"❌ Invalid action '{action}' for {agent_key} at step {step}")
                        return None
                    
            return plan_data
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"❌ Failed to parse planning response: {e}")
            return None

    def get_planned_action(self, agent_id: int) -> Optional[str]:
        """Get the current planned action for an agent (handles variable plan lengths)"""
        if self.current_plan is None:
            return None
            
        agent_key = f"agent_{agent_id}"
        if agent_key not in self.current_plan:
            return None
            
        # Handle case where plan is shorter than current step
        if self.plan_step >= len(self.current_plan[agent_key]):
            return None  # Will fallback to reactive behavior
            
        action = self.current_plan[agent_key][self.plan_step]
        
        # Validate action is available
        if action not in self.simulation.actions:
            return None
            
        return action

    def update_step_planning_success(self, agents_followed_plan: int, total_agents: int):
        """Update planning success for current step"""
        self.agents_followed_plan_per_step.append(agents_followed_plan)
        self.total_planning_attempts += 1
        success_rate = agents_followed_plan / total_agents
        
        if success_rate >= HyperParameters.PLANNING_SUCCESS_THRESHOLD:
            self.planning_success_count += 1

    def get_planning_success_rate(self) -> float:
        """Get the overall planning success rate"""
        if self.total_planning_attempts == 0:
            return 0.0
        return (self.planning_success_count / self.total_planning_attempts) * 100
    
    def get_average_agents_following_plan(self) -> float:
        """Get average number of agents following plan per step"""
        if not self.agents_followed_plan_per_step:
            return 0.0
        return np.mean(self.agents_followed_plan_per_step)
    
    def get_total_plan_adherence_rate(self) -> float:
        """Get total adherence rate across all steps and agents"""
        if not self.agents_followed_plan_per_step:
            return 0.0
        total_possible = len(self.agents_followed_plan_per_step) * HyperParameters.N_AGENTS
        total_followed = sum(self.agents_followed_plan_per_step)
        return (total_followed / total_possible) * 100
    
    def get_plan_status(self) -> Dict[str, Any]:
        """Get current status of the planning system"""
        if self.current_plan is None:
            return {
                "has_plan": False,
                "plan_step": self.plan_step,
                "agents_with_remaining_actions": 0,
                "total_plan_length": 0
            }
        
        agents_with_actions = 0
        min_plan_length = float('inf')
        max_plan_length = 0
        
        for i in range(HyperParameters.N_AGENTS):
            agent_key = f"agent_{i}"
            if agent_key in self.current_plan:
                plan_length = len(self.current_plan[agent_key])
                min_plan_length = min(min_plan_length, plan_length)
                max_plan_length = max(max_plan_length, plan_length)
                
                if self.plan_step < plan_length:
                    agents_with_actions += 1
        
        return {
            "has_plan": True,
            "plan_step": self.plan_step,
            "agents_with_remaining_actions": agents_with_actions,
            "min_plan_length": min_plan_length if min_plan_length != float('inf') else 0,
            "max_plan_length": max_plan_length,
            "plan_exhausted": agents_with_actions == 0
        }

# ============================================================================
# ENHANCED AGENT CLASS
# ============================================================================

class EnhancedAgent:
    """Enhanced Agent with comprehensive communication capabilities."""
    
    def __init__(self, agent_id: int, position: List[float], communication_manager: CommunicationManager):
        self.id = agent_id
        self.pos = np.array(position, dtype=float)
        self.history = [self.pos.copy()]
        self.communication_manager = communication_manager
        
        # Communication and coordination state
        self.current_constraints = {}
        self.last_intended_action = 'stay'
        self.velocity = np.array([0.0, 0.0])
        self.is_yielding = False
        self.yield_counter = 0
        self.last_yield_reason = None
        
        # Performance tracking
        self.proximity_violations = 0
        self.successful_negotiations = 0
        self.messages_sent = 0
        self.messages_received = 0

    def update_position(self, new_position: np.ndarray):
        """Update agent position and calculate velocity for priority calculations"""
        old_pos = self.pos.copy()
        self.pos = new_position
        self.history.append(self.pos.copy())
        self.velocity = new_position - old_pos

    def broadcast_movement_intent(self, intended_action: str, action_vector: np.ndarray, 
                                agent_positions: Dict[int, np.ndarray]):
        """Broadcast movement intentions to agents within communication range"""
        future_position = self.pos + action_vector
        
        for other_id in range(self.communication_manager.n_agents):
            if other_id == self.id:
                continue
                
            distance = np.linalg.norm(self.pos - agent_positions[other_id])
            if distance <= self.communication_manager.safety_zone.communication_radius:
                success = self.communication_manager.send_message(
                    sender_id=self.id,
                    receiver_id=other_id,
                    message_type=MessageType.MOVEMENT_INTENT,
                    content={
                        'intended_action': intended_action,
                        'future_position': future_position.tolist(),
                        'current_velocity': np.linalg.norm(self.velocity),
                        'confidence': 0.9
                    }
                )
                if success:
                    self.messages_sent += 1

    def process_communication_constraints(self, intended_action: str) -> Dict[str, Any]:
        """Process incoming messages and determine movement constraints"""
        constraints = self.communication_manager.process_agent_messages(
            self.id, self.pos, intended_action
        )
        
        self.current_constraints = constraints
        
        if constraints.get('received_communication', False):
            if (constraints.get('must_yield', False) or 
                constraints.get('emergency_stop', False) or 
                not constraints.get('can_proceed', True)):
                self.communication_manager.stats['behavior_changes_due_to_communication'] += 1
        
        self.messages_received += len(self.communication_manager.get_messages(self.id))
        
        if constraints['must_yield'] or constraints['emergency_stop']:
            self.is_yielding = True
            self.yield_counter = constraints.get('yield_duration', 2)
            self.last_yield_reason = constraints.get('yield_reason', 'unknown')
                
        elif self.yield_counter > 0:
            self.yield_counter -= 1
            if self.yield_counter == 0:
                self.is_yielding = False
                self.last_yield_reason = None
                
        return constraints

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def point_to_line_distance(point, line_start, line_end):
    """Calculate shortest distance from point to line segment"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)
    
    line_unitvec = line_vec / line_len
    proj_length = np.dot(point_vec, line_unitvec)
    proj_length = max(0, min(line_len, proj_length))
    
    closest_point = line_start + proj_length * line_unitvec
    return np.linalg.norm(point - closest_point)

def crosses_interference_line(pos1, pos2, line_start, line_end):
    """Check if movement crosses interference line"""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    return ccw(pos1, line_start, line_end) != ccw(pos2, line_start, line_end) and \
           ccw(pos1, pos2, line_start) != ccw(pos1, pos2, line_end)

# ============================================================================
# ADVERSARY CLASS
# ============================================================================

class Adversary:
    """Adversary with predetermined trajectory"""
    def __init__(self, initial_position, trajectory_params):
        self.pos = np.array(initial_position, dtype=float)
        self.params = trajectory_params
        self.history = [self.pos.copy()]
        self.time = 0

    def update_position(self):
        self.time += 1

        if 11 <= self.time <= 15:
            start_pos = np.array([50.0, 35.0])
            end_pos = np.array([20.0, 0.0])
            ratio = (self.time - 11) / 4.0
            self.pos = start_pos + ratio * (end_pos - start_pos)
        else:
            a, b, c, x_vel = self.params['a'], self.params['b'], self.params['c'], self.params['x_vel']
            new_x = self.pos[0] + x_vel
            new_y = a * (new_x**2) + b * new_x + c
            new_x = max(0, min(100, new_x))
            new_y = max(0, min(100, new_y))
            self.pos = np.array([new_x, new_y])
        
        self.history.append(self.pos.copy())

# ============================================================================
# MAIN ENHANCED SIMULATION CLASS WITH PLANNING AND COMPREHENSIVE METRICS
# ============================================================================

class MASSimulationWithSACRAndPlanning:
    """Complete MAS simulation with SACR protocol, planning, and comprehensive metrics."""
    
    def __init__(self, use_forward_prediction=False, scenario_type="formation", max_steps=None):
        # Base simulation parameters
        self.n_agents = HyperParameters.N_AGENTS
        self.rho = HyperParameters.RHO
        self.safe_distance = HyperParameters.SAFE_DISTANCE
        self.delta_a = HyperParameters.DELTA_A
        self.scenario_type = scenario_type
        self.use_forward_prediction = use_forward_prediction
        self.c = HyperParameters.C
        
        # Determine max steps based on scenario if not provided
        if max_steps is None:
            if scenario_type == "interference":
                self.max_steps = HyperParameters.INTERFERENCE_STEPS
            else:
                self.max_steps = HyperParameters.MAX_STEPS
        else:
            self.max_steps = max_steps
        
        # SACR Protocol initialization
        self.safety_zone = SafetyZone()
        self.communication_manager = CommunicationManager(self.n_agents, self.safety_zone)
        
        # Planning system (now with full simulation length)
        self.planning_system = PlanningSystem(self, self.max_steps)
        
        # Target position setup (scenario-dependent)
        if scenario_type == "formation":
            initial_center = np.array([12.0, 12.0])
            self.target_position = initial_center + np.array([30.0, 0.0])
        elif scenario_type == "interference" or scenario_type == "obstacle":
            self.target_position = np.array([80.0, 40.0])
            
        # Interference line definition
        self.interference_line_start = np.array([50.0, 35.0])
        self.interference_line_end = np.array([20.0, 0.0])
        self.has_crossed_line = False
        
        # Enhanced collision tracking
        self.exact_collision_count = 0
        self.proximity_violation_count = 0
        
        # Prediction parameters
        self.prediction_steps = HyperParameters.PREDICTION_STEPS
        self.prediction_weights = HyperParameters.PREDICTION_WEIGHTS

        # Initialize agents and adversary
        self._initialize_agents_and_adversary()
        self._define_communication_topology()
        
        # Comprehensive loss tracking
        self.step_losses = []
        self.detailed_losses = {
            'intra_group': [],
            'coalition': [],
            'external': [],
            'communication': [],
            'total': []
        }

        # Action space
        self.actions = {
            'stay': np.array([0, 0]),
            'up': np.array([0, self.delta_a]),
            'down': np.array([0, -self.delta_a]),
            'left': np.array([-self.delta_a, 0]),
            'right': np.array([self.delta_a, 0])
        }

    def _initialize_agents_and_adversary(self):
        """Initialize enhanced agents with communication capabilities"""
        initial_positions = [
            [5, 5],   [12, 5],   [19, 5],
            [5, 12],  [12, 12],  [19, 12],
            [5, 19],  [12, 19],  [19, 19]
        ]
        
        self.agents = []
        for i in range(self.n_agents):
            agent = EnhancedAgent(
                agent_id=i, 
                position=initial_positions[i],
                communication_manager=self.communication_manager
            )
            self.agents.append(agent)
        
        # Initialize adversary
        adversary_params = {
            'a': 0.01, 'b': 0.5, 'c': 20, 'x_vel': 2.0
        }
        self.adversary = Adversary(initial_position=[10, 20], trajectory_params=adversary_params)
        self.adversary.update_position()

    def _define_communication_topology(self):
        """Define initial agent communication topology (3x3 grid neighbors)"""
        self.adj_matrix = np.zeros((self.n_agents, self.n_agents))
        
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                xi, yi = i % 3, i // 3
                xj, yj = j % 3, j // 3
                
                if abs(xi - xj) + abs(yi - yj) == 1:
                    self.adj_matrix[i, j] = self.adj_matrix[j, i] = 1

    def _predict_adversary_pos(self):
        """Predict future adversary positions using polynomial extrapolation"""
        history = np.array(self.adversary.history)
        
        if len(history) < 3:
             return [self.adversary.pos] * self.prediction_steps
        
        degree = min(3, len(history) - 1)
        t = np.arange(len(history))
        
        fx = np.poly1d(np.polyfit(t, history[:, 0], degree))
        fy = np.poly1d(np.polyfit(t, history[:, 1], degree))

        future_t = np.arange(len(history), len(history) + self.prediction_steps)
        return [np.array([fx(t_val), fy(t_val)]) for t_val in future_t]

    def _calculate_communication_loss(self, agent_id: int, next_pos: np.ndarray, 
                                    communication_constraints: Dict[str, Any]) -> float:
        """Calculate loss component for communication protocol compliance"""
        loss = 0.0
        
        if communication_constraints.get('must_yield', False):
            movement_distance = np.linalg.norm(next_pos - self.agents[agent_id].pos)
            if movement_distance > 0.1:
                loss += 500.0 * movement_distance
        
        if communication_constraints.get('emergency_stop', False):
            movement_distance = np.linalg.norm(next_pos - self.agents[agent_id].pos)
            if movement_distance > 0.1:
                loss += 1000.0 * movement_distance
        
        if not communication_constraints.get('can_proceed', True):
            movement_distance = np.linalg.norm(next_pos - self.agents[agent_id].pos)
            if movement_distance > 0.1:
                loss += 200.0 * movement_distance
                
        return loss

    def _calculate_intra_group_loss(self, agent_id: int, next_pos: np.ndarray, 
                                   agent_next_states: Dict[int, np.ndarray]) -> float:
        """Calculate intra-group loss with enhanced collision avoidance"""
        loss = 0.0
        
        # Enhanced collision avoidance with graduated penalties
        for j in range(self.n_agents):
            if j != agent_id:
                distance = np.linalg.norm(next_pos - agent_next_states[j])
                if distance < self.safety_zone.collision_radius:
                    loss += (self.safety_zone.collision_radius - distance)**3 * 2000
                elif distance < self.safety_zone.warning_radius:
                    loss += (self.safety_zone.warning_radius - distance)**2 * 100
        
        # Formation maintenance
        for j in range(self.n_agents):
            if self.adj_matrix[agent_id, j] == 1:
                actual_distance = np.linalg.norm(next_pos - agent_next_states[j])
                loss += abs(actual_distance - self.rho)
        
        # Target reaching
        if self.target_position is not None:
            target_distance = np.linalg.norm(next_pos - self.target_position)
            loss += target_distance * 15.0
            
        return loss

    def _calculate_coalition_loss(self, agent_id: int, next_pos: np.ndarray, 
                                agent_next_states: Dict[int, np.ndarray], 
                                coalition_adj_matrix: np.ndarray) -> float:
        """Calculate coalition coordination loss"""
        loss = 0.0
        
        for j in range(self.n_agents):
            if coalition_adj_matrix[agent_id, j] == 1:
                coalition_distance = np.linalg.norm(next_pos - agent_next_states[j])
                loss += (coalition_distance - self.rho)
                
        return loss

    def _calculate_external_loss(self, agent_pos: np.ndarray, adversary_pos_list: List[np.ndarray], 
                               current_pos: Optional[np.ndarray] = None) -> float:
        """Calculate external loss (adversary avoidance and scenario constraints)"""
        base_loss = 0.0
        
        # Adversary avoidance
        if not self.use_forward_prediction:
            distance_to_adversary = np.linalg.norm(agent_pos - adversary_pos_list[0])
            base_loss = 1.0 / (distance_to_adversary + self.c)
        else:
            losses = []
            for pos in adversary_pos_list[:self.prediction_steps + 1]:
                distance = np.linalg.norm(agent_pos - pos)
                losses.append(1.0 / (distance + self.c))
            
            for k, loss in enumerate(losses):
                if k < len(self.prediction_weights):
                    base_loss += self.prediction_weights[k] * loss
        
        enhanced_loss = base_loss
        
        # Scenario-specific constraints
        if self.scenario_type == "interference":
            if current_pos is not None:
                if crosses_interference_line(current_pos, agent_pos, 
                                           self.interference_line_start, 
                                           self.interference_line_end):
                    enhanced_loss += 100000
            
            if self.use_forward_prediction:
                line_distance = point_to_line_distance(agent_pos, 
                                                     self.interference_line_start, 
                                                     self.interference_line_end)
                if line_distance < 15.0:
                    enhanced_loss += (15.0 - line_distance)**2 / 50

        elif self.scenario_type == "obstacle":
            obstacle_center = np.array([60, 35])
            obstacle_distance = np.linalg.norm(agent_pos - obstacle_center)
            
            danger_radius = 15.0 if self.use_forward_prediction else 10.0
            if obstacle_distance < danger_radius:
                enhanced_loss += (obstacle_distance - danger_radius)**2 / 100
            
            if self.use_forward_prediction and obstacle_distance < 8.5:
                enhanced_loss += (8.5 - obstacle_distance)**4 * 100
        
        return enhanced_loss

    def _get_priority_metrics(self, agent_id: int, coalition_info: Dict) -> PriorityMetrics:
        """Calculate comprehensive priority metrics for an agent"""
        agent = self.agents[agent_id]
        distance_to_target = (np.linalg.norm(agent.pos - self.target_position) 
                            if self.target_position is not None else 0)
        velocity_magnitude = np.linalg.norm(agent.velocity)
        is_coalition_member = coalition_info.get('members', [False] * self.n_agents)[agent_id]
        
        return PriorityMetrics(
            distance_to_target=distance_to_target,
            agent_id=agent_id,
            current_velocity=velocity_magnitude,
            coalition_member=is_coalition_member
        )

    def _determine_coalition_membership(self, coalition_info: Dict, agent_next_states: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Determine coalition membership and coordination signals"""
        coalition_adj_matrix = np.zeros((self.n_agents, self.n_agents))
        signals = np.zeros(self.n_agents)
        
        if not coalition_info['active']:
            return coalition_adj_matrix, signals
        
        # Create connections between coalition members
        for i in range(self.n_agents):
            if coalition_info['members'][i]:
                for j in range(i + 1, self.n_agents):
                    if coalition_info['members'][j]:
                        coalition_adj_matrix[i, j] = coalition_adj_matrix[j, i] = 1
        
        # Generate coordination signals
        for i in range(self.n_agents):
            if coalition_info['members'][i]:
                ri_intra = self._calculate_intra_group_loss(i, agent_next_states[i], agent_next_states)
                rc_coalition = self._calculate_coalition_loss(i, agent_next_states[i], 
                                                            agent_next_states, coalition_adj_matrix)
                
                if rc_coalition > 0:
                    signals[i] = 1
                    
        return coalition_adj_matrix, signals

    def _get_action_name(self, action_vector: np.ndarray) -> str:
        """Convert action vector to action name"""
        for name, vec in self.actions.items():
            if np.allclose(action_vector, vec, atol=0.1):
                return name
        return 'stay'

    def run_step_with_planning(self, step_count: int):
        """Execute one simulation step with planning integration"""
        # Update communication manager
        self.communication_manager.step_update()
        
        # Update adversary
        self.adversary.update_position()
        
        # Get predicted adversary positions
        predicted_adv_pos = [self.adversary.pos]
        if self.use_forward_prediction:
            predicted_adv_pos.extend(self._predict_adversary_pos())

        # Current agent positions
        current_positions = {i: agent.pos for i, agent in enumerate(self.agents)}
        
        # Update communication topology
        self.communication_manager.update_communication_topology(current_positions)
        
        # Phase 1: Movement intention (considering planned actions)
        intended_moves = {}
        agents_following_plan = 0
        
        for i, agent in enumerate(self.agents):
            # Try to get planned action first
            planned_action = self.planning_system.get_planned_action(i)
            
            if planned_action is not None:
                intended_action_vec = self.actions[planned_action]
                agents_following_plan += 1
            else:
                # Fallback to target-directed movement
                if self.target_position is not None:
                    direction = self.target_position - agent.pos
                    if np.linalg.norm(direction) > 0.1:
                        direction = direction / np.linalg.norm(direction)
                        intended_action_vec = direction * self.delta_a
                    else:
                        intended_action_vec = np.array([0, 0])
                else:
                    intended_action_vec = np.array([0, 0])
                
            intended_moves[i] = intended_action_vec
            
            # Broadcast movement intention
            intended_action_name = self._get_action_name(intended_action_vec)
            agent.broadcast_movement_intent(intended_action_name, intended_action_vec, current_positions)
        
        # Update planning success statistics
        self.planning_system.update_step_planning_success(agents_following_plan, self.n_agents)
        
        # Phase 2: Coalition formation and conflict detection
        coalition_info = {'active': False, 'members': [False] * self.n_agents}
        
        max_external_loss = 0.0
        coalition_leader = None
        potential_coalition_members = np.zeros(self.n_agents, dtype=bool)
        
        for i, agent in enumerate(self.agents):
            external_loss = self._calculate_external_loss(agent.pos, predicted_adv_pos, agent.pos)
            
            if external_loss > 1.0/self.safe_distance:
                potential_coalition_members[i] = True
                
                if external_loss > max_external_loss:
                    max_external_loss = external_loss
                    coalition_leader = i

        coalition_info = {
            'active': coalition_leader is not None,
            'leader': coalition_leader,
            'members': potential_coalition_members,
            'leader_threat_level': max_external_loss
        }

        priority_metrics = {i: self._get_priority_metrics(i, coalition_info) for i in range(self.n_agents)}
        
        # Detect proximity conflicts
        conflicts = self.communication_manager.detect_proximity_conflicts(current_positions, intended_moves)
        
        # Initiate negotiations for conflicts
        for agent1_id, agent2_id in conflicts:
            self.communication_manager.initiate_negotiation(
                agent1_id, agent2_id, current_positions, priority_metrics
            )
        
        # Phase 3: Action selection with all constraints
        agent_actions = {}
        new_positions = {}
        step_losses = {'intra_group': 0.0, 'coalition': 0.0, 'external': 0.0, 'communication': 0.0}
        
        for i, agent in enumerate(self.agents):
            # Process communication constraints
            preliminary_action = self._get_action_name(intended_moves[i])
            communication_constraints = agent.process_communication_constraints(preliminary_action)
            
            # Action selection with all constraints
            best_action = 'stay'
            min_total_loss = float('inf')
            
            for action_name, action_vec in self.actions.items():
                next_pos = np.clip(agent.pos + action_vec, 0, 100)
                
                # Check communication constraints
                if communication_constraints.get('emergency_stop', False) and action_name != 'stay':
                    continue
                if communication_constraints.get('must_yield', False) and action_name != 'stay':
                    continue
                if not communication_constraints.get('can_proceed', True) and action_name != 'stay':
                    continue
                
                # Calculate all loss components
                temp_next_states = {j: (next_pos if j == i else current_positions[j]) 
                                  for j in range(self.n_agents)}
                
                coalition_adj_matrix, signals = self._determine_coalition_membership(
                    coalition_info, temp_next_states)
                signal = signals[i] if (coalition_info['active'] and 
                                      coalition_info['members'][i]) else 0
                
                # Calculate individual loss components
                intra_loss = self._calculate_intra_group_loss(i, next_pos, temp_next_states)
                coalition_loss = self._calculate_coalition_loss(i, next_pos, temp_next_states, coalition_adj_matrix)
                external_loss = self._calculate_external_loss(next_pos, predicted_adv_pos, current_pos=agent.pos)
                communication_loss = self._calculate_communication_loss(i, next_pos, communication_constraints)
                
                # Combine losses
                total_loss = (intra_loss + 
                            signal * coalition_loss + 
                            external_loss * 50 + 
                            communication_loss)

                if total_loss < min_total_loss:
                    min_total_loss = total_loss
                    best_action = action_name
            
            agent_actions[i] = best_action
            new_positions[i] = np.clip(agent.pos + self.actions[best_action], 0, 100)

        # Update planning step
        self.planning_system.plan_step += 1
        
        # Calculate step losses for tracking
        for i, agent in enumerate(self.agents):
            temp_next_states = {j: new_positions[j] for j in range(self.n_agents)}
            coalition_adj_matrix, signals = self._determine_coalition_membership(coalition_info, temp_next_states)
            
            step_losses['intra_group'] += self._calculate_intra_group_loss(i, new_positions[i], temp_next_states)
            step_losses['coalition'] += self._calculate_coalition_loss(i, new_positions[i], temp_next_states, coalition_adj_matrix)
            step_losses['external'] += self._calculate_external_loss(new_positions[i], predicted_adv_pos, current_pos=agent.pos)
            step_losses['communication'] += self._calculate_communication_loss(i, new_positions[i], 
                                                                              self.agents[i].current_constraints)

        # Phase 4: Position updates and constraint checking
        
        # Check interference line violations
        if self.scenario_type == "interference":
            for i, agent in enumerate(self.agents):
                if crosses_interference_line(agent.pos, new_positions[i], 
                                           self.interference_line_start, 
                                           self.interference_line_end):
                    self.has_crossed_line = True
                    print(f"🚩 VIOLATION: Agent {i+1} crossed interference line at step {step_count}")

        # Update agent positions
        for i, agent in enumerate(self.agents):
            agent.update_position(new_positions[i])
        
        # Check for proximity violations
        self.check_exact_collisions(step_count)
        
        # Update loss tracking
        for loss_type, value in step_losses.items():
            self.detailed_losses[loss_type].append(value)
        
        total_loss = sum(step_losses.values())
        self.detailed_losses['total'].append(total_loss)
        
        # Send clear signals for resolved conflicts
        for agent1_id, agent2_id in conflicts:
            distance_now = np.linalg.norm(new_positions[agent1_id] - new_positions[agent2_id])
            if distance_now >= self.safety_zone.warning_radius:
                self.communication_manager.send_message(
                    agent1_id, agent2_id, MessageType.CLEAR_SIGNAL, {'conflict_resolved': True}
                )
                self.communication_manager.send_message(
                    agent2_id, agent1_id, MessageType.CLEAR_SIGNAL, {'conflict_resolved': True}
                )
                self.communication_manager.stats['conflicts_resolved'] += 1

    def check_exact_collisions(self, step_count: int):
        """Enhanced collision detection with communication awareness"""
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                distance = np.linalg.norm(self.agents[i].pos - self.agents[j].pos)
                
                if np.array_equal(self.agents[i].pos, self.agents[j].pos):
                    self.exact_collision_count += 1
                    print(f"🚨 EXACT COLLISION: Agents {i+1} and {j+1} at step {step_count}")
                elif distance < self.safety_zone.collision_radius:
                    self.proximity_violation_count += 1
                    print(f"🚨 PROXIMITY VIOLATION: Agents {i+1} and {j+1} "
                          f"(distance: {distance:.2f}) at step {step_count}")

    def check_target_reached(self, tolerance: float = 5.0) -> bool:
        """Check if all agents reached target"""
        if self.target_position is None:
            return False
        return all(np.linalg.norm(agent.pos - self.target_position) <= tolerance 
                  for agent in self.agents)

    def check_formation_constraints(self, tolerance: float = 2.0) -> bool:
        """Check if formation is maintained"""
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if self.adj_matrix[i, j] == 1:
                    actual_distance = np.linalg.norm(self.agents[i].pos - self.agents[j].pos)
                    if abs(actual_distance - self.rho) > tolerance:
                        return False
        return True

    def check_forbidden_areas(self) -> bool:
        """Check if forbidden areas were violated"""
        if self.scenario_type == "interference" and self.has_crossed_line:
            return False
        elif self.scenario_type == "obstacle" and self.use_forward_prediction:
            obstacle_center = np.array([60, 35])
            obstacle_radius = 8.5
            for agent in self.agents:
                if np.linalg.norm(agent.pos - obstacle_center) < obstacle_radius:
                    return False
        return True

    def check_agent_proximity_collisions(self, min_distance: float = 2.0) -> bool:
        """Check if agents maintain minimum distance"""
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.agents[i].pos - self.agents[j].pos) < min_distance:
                    return False
        return True

    def check_success_conditions(self) -> bool:
        """Check if all success conditions are met"""
        return (self.check_target_reached() and 
                self.check_formation_constraints() and 
                self.check_forbidden_areas() and 
                self.check_agent_proximity_collisions())

    def get_comprehensive_metrics(self):
        """Get all requested metrics in a structured format"""
        comm_stats = self.communication_manager.stats
        loss_summary = self.get_loss_summary()
        
        # Calculate resolution success rate
        if comm_stats['conflicts_detected'] > 0:
            resolution_rate = (comm_stats['conflicts_resolved'] / comm_stats['conflicts_detected']) * 100
        else:
            resolution_rate = 0.0
        
        # Get planning metrics
        total_adherence_rate = self.planning_system.get_total_plan_adherence_rate()
        avg_agents_following = self.planning_system.get_average_agents_following_plan()
        
        return {
            'proximity_violations': self.proximity_violation_count,
            'exact_collisions': self.exact_collision_count,
            'messages_sent': comm_stats['messages_sent'],
            'conflicts_resolved': comm_stats['conflicts_resolved'],
            'behavior_changes': comm_stats['behavior_changes_due_to_communication'],
            'resolution_success_rate': resolution_rate,
            'total_loss': loss_summary['total']['total'] if 'error' not in loss_summary else 0.0,
            'plan_adherence_rate': total_adherence_rate,
            'avg_agents_following_plan': avg_agents_following
        }

    def run_simulation_with_planning(self, max_steps: int = None):
        """Run complete simulation with full simulation planning"""
        if max_steps is None:
            max_steps = self.max_steps
            
        pred_status = "with prediction" if self.use_forward_prediction else "without prediction"
        print(f"🚀 Starting SACR-enhanced {self.scenario_type} scenario ({pred_status}) with Full Simulation Planning...")
        print(f"📡 Communication Protocol: Active")
        print(f"🧠 Planning System: Active (Full Simulation: {max_steps} steps)")
        
        # Get full simulation planning
        planning_prompt = self.planning_system.generate_full_simulation_planning_request()
        print("\n" + "="*80)
        print("🧠 FULL SIMULATION PLANNING REQUEST - Copy this to your chatbot:")
        print("="*80)
        print(planning_prompt)
        print("="*80)
        
        # Get planning response from user
        print(f"\nPlease paste the complete {max_steps}-step planning response here:")
        planning_response = input()
        
        # Parse planning response
        parsed_plan = self.planning_system.parse_planning_response(planning_response)
        if parsed_plan is not None:
            self.planning_system.current_plan = parsed_plan['plan']
            self.planning_system.plan_step = 0
            print(f"✅ Full simulation plan loaded successfully!")
            print(f"📊 Confidence: {parsed_plan.get('confidence', 'N/A')}")
            print(f"📝 Reasoning: {parsed_plan.get('reasoning', 'N/A')}")
            print(f"🎯 Plan covers all {max_steps} steps for all {HyperParameters.N_AGENTS} agents")
        else:
            print("❌ Failed to parse planning response. Proceeding with reactive behavior only.")
        
        print(f"\n🎯 Starting simulation execution...")
        
        actual_steps_executed = 0
        
        for step_count in range(1, max_steps + 1):
            self.run_step_with_planning(step_count)
            actual_steps_executed = step_count
            
            # Progress indicator (adjust frequency based on total steps)
            progress_interval = max(10, min(50, max_steps // 10))
            if step_count % progress_interval == 0:
                planning_rate = self.planning_system.get_total_plan_adherence_rate()
                print(f"📊 Step {step_count}/{max_steps} - Plan adherence: {planning_rate:.1f}%")
            
            # Early termination for formation scenarios
            if self.scenario_type == "formation" and self.check_success_conditions():
                print(f"✅ SUCCESS! Formation task completed in {step_count} steps (planned for {max_steps}).")
                break

        # Comprehensive final statistics with all requested metrics
        metrics = self.get_comprehensive_metrics()
        comm_stats = self.communication_manager.stats
        
        print(f"\n📊 SIMULATION RESULTS:")
        print(f"   Steps Executed: {actual_steps_executed}/{max_steps}")
        print(f"   🔹 PROXIMITY VIOLATIONS: {metrics['proximity_violations']}")
        print(f"   🔹 EXACT COLLISIONS: {metrics['exact_collisions']}")
        print(f"   Target Reached: {self.check_target_reached()}")
        print(f"   Formation Maintained: {self.check_formation_constraints()}")
        print(f"   No Forbidden Area Violations: {self.check_forbidden_areas()}")
        
        print(f"\n📡 COMMUNICATION STATISTICS:")
        print(f"   🔹 MESSAGES SENT: {metrics['messages_sent']}")
        print(f"   🔹 CONFLICTS RESOLVED: {metrics['conflicts_resolved']}")
        print(f"   🔹 BEHAVIOR CHANGES: {metrics['behavior_changes']}")
        print(f"   🔹 RESOLUTION SUCCESS RATE: {metrics['resolution_success_rate']:.1f}%")
        print(f"   Negotiations Initiated: {comm_stats['negotiations_initiated']}")
        print(f"   Successful Yields: {comm_stats['successful_yields']}")
        
        print(f"\n🧠 FULL SIMULATION PLANNING STATISTICS:")
        print(f"   Total Plan Adherence Rate: {metrics['plan_adherence_rate']:.1f}%")
        print(f"   Average Agents Following Plan: {metrics['avg_agents_following_plan']:.1f}/{HyperParameters.N_AGENTS}")
        
        print(f"\n💰 LOSS METRICS:")
        print(f"   🔹 TOTAL LOSS: {metrics['total_loss']:.4f}")
        
        return self.agents, self.adversary

    def get_loss_summary(self):
        """Return comprehensive loss statistics"""
        if not self.detailed_losses['total']:
            return {"error": "No loss data recorded"}
        
        summary = {}
        for loss_type in ['intra_group', 'coalition', 'external', 'communication', 'total']:
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
    """Enhanced visualizer for all scenarios with communication indicators"""
    
    def __init__(self, agents, adversary, title, scenario_type, communication_stats=None):
        self.agents = agents
        self.adversary = adversary
        self.title = title
        self.scenario_type = scenario_type
        self.communication_stats = communication_stats
        self.colors = plt.cm.jet(np.linspace(0, 1, len(agents)))

    def plot_trajectories(self, subplot_ax):
        """Plot agent trajectories with communication indicators"""
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
            
            # Mark start and end positions
            subplot_ax.plot(hist[0, 0], hist[0, 1], 'o', color=self.colors[i], 
                          markersize=8)
            subplot_ax.plot(hist[-1, 0], hist[-1, 1], 's', color=self.colors[i], 
                          markersize=8)

        # Add scenario-specific elements
        if self.scenario_type == "interference":
            subplot_ax.plot([50, 20], [35, 0], 'r--', linewidth=2, alpha=0.9, 
                          label="Interference Path")
            target_pos = np.array([80.0, 40.0])
        elif self.scenario_type == "formation":
            target_pos = np.array([42.0, 12.0])
        else:
            target_pos = np.array([80.0, 40.0])
        
        # Mark target position
        if target_pos is not None:
            subplot_ax.plot(target_pos[0], target_pos[1], 'g*', markersize=15, 
                          label="Target")
        
        # Add communication statistics as text
        if self.communication_stats:
            stats_text = (f"Messages: {self.communication_stats['messages_sent']}\n"
                         f"Conflicts: {self.communication_stats['conflicts_detected']}\n"
                         f"Resolved: {self.communication_stats['conflicts_resolved']}")
            subplot_ax.text(0.02, 0.98, stats_text, transform=subplot_ax.transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', 
                          facecolor='wheat', alpha=0.8), fontsize=8)
        
        subplot_ax.grid(True, linestyle='--', alpha=0.6)
        subplot_ax.legend(fontsize='small')

    def plot_obstacle_avoidance_scenario(self, subplot_ax):
        """Plot obstacle scenario with safety zones"""
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
        subplot_ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=20, 
                       label="Goal")
        
        # Plot agent trajectories
        for i, agent in enumerate(self.agents):
            trajectory = np.array(agent.history)
            subplot_ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=self.colors[i], 
                          markersize=5, linewidth=1.5)
            
            # Mark start and end
            subplot_ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', color=self.colors[i], 
                          markersize=8)
            subplot_ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=self.colors[i], 
                          markersize=8)
        
        # Add communication statistics
        if self.communication_stats:
            stats_text = (f"Messages: {self.communication_stats['messages_sent']}\n"
                         f"Conflicts: {self.communication_stats['conflicts_detected']}\n"
                         f"Resolved: {self.communication_stats['conflicts_resolved']}")
            subplot_ax.text(0.02, 0.98, stats_text, transform=subplot_ax.transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', 
                          facecolor='wheat', alpha=0.8), fontsize=8)
        
        subplot_ax.grid(True, linestyle='--', alpha=0.6)
        subplot_ax.legend(fontsize='small')

# ============================================================================
# COMPREHENSIVE ANALYSIS AND REPORTING FUNCTIONS
# ============================================================================

def print_comprehensive_analysis_with_planning(scenario_results):
    """Print detailed analysis including all requested metrics and planning effectiveness"""
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE ANALYSIS: SACR PROTOCOL + FULL SIMULATION PLANNING")
    print("="*80)
    
    # Collect all metrics for comparison
    all_metrics = {}
    
    for scenario_name, (agents, adversary, simulation) in scenario_results.items():
        print(f"\n📊 {scenario_name.upper()}")
        print("-" * 60)
        
        # Get comprehensive metrics
        metrics = simulation.get_comprehensive_metrics()
        all_metrics[scenario_name] = metrics
        
        # Print all requested metrics clearly
        print(f"   🔹 PROXIMITY VIOLATIONS: {metrics['proximity_violations']}")
        print(f"   🔹 EXACT COLLISIONS: {metrics['exact_collisions']}")
        print(f"   🔹 MESSAGES SENT: {metrics['messages_sent']}")
        print(f"   🔹 CONFLICTS RESOLVED: {metrics['conflicts_resolved']}")
        print(f"   🔹 BEHAVIOR CHANGES: {metrics['behavior_changes']}")
        print(f"   🔹 RESOLUTION SUCCESS RATE: {metrics['resolution_success_rate']:.1f}%")
        print(f"   🔹 TOTAL LOSS: {metrics['total_loss']:.4f}")
        
        loss_summary = simulation.get_loss_summary()
        
        if "error" not in loss_summary:
            # Print detailed loss breakdown
            loss_types = [
                ('INTRA-GROUP', 'intra_group'),
                ('COALITION', 'coalition'), 
                ('EXTERNAL', 'external'),
                ('COMMUNICATION', 'communication'),
                ('TOTAL COMBINED', 'total')
            ]
            
            print(f"   🔹 DETAILED LOSS BREAKDOWN:")
            for display_name, loss_key in loss_types:
                data = loss_summary[loss_key]
                print(f"      {display_name}: Total={data['total']:.4f}, Avg={data['average']:.4f}")
        
        # Planning effectiveness metrics
        print(f"   🔹 PLANNING EFFECTIVENESS:")
        print(f"      Plan Adherence Rate: {metrics['plan_adherence_rate']:.1f}%")
        print(f"      Average Agents Following Plan: {metrics['avg_agents_following_plan']:.1f}/{HyperParameters.N_AGENTS}")
        print(f"      Planning Contribution: {'High' if metrics['plan_adherence_rate'] > 70 else 'Medium' if metrics['plan_adherence_rate'] > 40 else 'Low'}")
    
    # Calculate normalized losses for comparison
    all_total_losses = [metrics['total_loss'] for metrics in all_metrics.values()]
    if len(all_total_losses) > 1:
        min_loss = min(all_total_losses)
        max_loss = max(all_total_losses)
        loss_range = max_loss - min_loss if max_loss != min_loss else 1
        
        print(f"\n🔹 NORMALIZED TOTAL LOSSES (0=Best, 1=Worst):")
        for scenario_name, metrics in all_metrics.items():
            normalized_loss = (metrics['total_loss'] - min_loss) / loss_range
            print(f"   {scenario_name:30}: {normalized_loss:.4f}")
            all_metrics[scenario_name]['normalized_total_loss'] = normalized_loss
    
    # Summary rankings
    print("\n" + "="*80)
    print("🏆 PERFORMANCE RANKINGS:")
    print("="*80)
    
    print("\n🎯 TOTAL LOSSES (Lower is Better):")
    for scenario, metrics in sorted(all_metrics.items(), key=lambda x: x[1]['total_loss']):
        print(f"   {scenario:30}: {metrics['total_loss']:12.4f}")
    
    print("\n🎯 PROXIMITY VIOLATIONS (Lower is Better):")
    for scenario, metrics in sorted(all_metrics.items(), key=lambda x: x[1]['proximity_violations']):
        print(f"   {scenario:30}: {metrics['proximity_violations']:12}")
    
    print("\n🎯 RESOLUTION SUCCESS RATE (Higher is Better):")
    for scenario, metrics in sorted(all_metrics.items(), key=lambda x: x[1]['resolution_success_rate'], reverse=True):
        print(f"   {scenario:30}: {metrics['resolution_success_rate']:12.1f}%")
    
    print("\n🧠 PLAN ADHERENCE RATE (Higher is Better):")
    for scenario, metrics in sorted(all_metrics.items(), key=lambda x: x[1]['plan_adherence_rate'], reverse=True):
        print(f"   {scenario:30}: {metrics['plan_adherence_rate']:12.1f}%")
    
    return all_metrics

def plot_comprehensive_metrics_with_planning(scenario_results, all_metrics):
    """Create comprehensive visualization of all metrics including planning"""
    # Create a large figure for all plots with improved layout
    fig = plt.figure(figsize=(20, 26), constrained_layout=False)  # Increased height for proper spacing
    
    # Define the layout: trajectory plots on top, metrics plots below
    num_scenarios = len(scenario_results)
    
    # Create grid layout with balanced spacing
    gs_main = fig.add_gridspec(5, 1, height_ratios=[2.5, 1, 1, 1, 1], 
                              hspace=0.3, top=0.95, bottom=0.05)  # Balanced spacing for axis labels
    
    # Top section: Trajectory plots with moderate spacing
    gs_traj = gs_main[0].subgridspec(2, 3, hspace=0.2, wspace=0.2)  # Moderate spacing
    
    # Plot trajectories
    scenario_index = 0
    for scenario_name, (agents, adversary, simulation) in scenario_results.items():
        row = scenario_index // 3
        col = scenario_index % 3
        
        ax = fig.add_subplot(gs_traj[row, col])
        
        comm_stats = simulation.communication_manager.stats
        
        # Determine scenario type from name
        if "formation" in scenario_name.lower():
            scenario_type = "formation"
        elif "interference" in scenario_name.lower():
            scenario_type = "interference"
        elif "obstacle" in scenario_name.lower():
            scenario_type = "obstacle"
        else:
            scenario_type = "formation"
        
        viz = Visualizer(agents, adversary, scenario_name, scenario_type, comm_stats)
        
        if scenario_type == "obstacle":
            viz.plot_obstacle_avoidance_scenario(ax)
        else:
            viz.plot_trajectories(ax)
            
        scenario_index += 1
    
    # Bottom sections: Metrics plots with reduced spacing
    metrics_names = ['proximity_violations', 'exact_collisions', 'messages_sent', 
                    'conflicts_resolved', 'behavior_changes', 'resolution_success_rate', 
                    'total_loss', 'normalized_total_loss', 'plan_adherence_rate']
    
    metrics_labels = ['Proximity Violations', 'Exact Collisions', 'Messages Sent',
                     'Conflicts Resolved', 'Behavior Changes', 'Resolution Success Rate (%)',
                     'Total Loss', 'Normalized Total Loss', 'Plan Adherence Rate (%)']
    
    # Create four subplot rows for metrics with reduced spacing
    gs_metrics1 = gs_main[1].subgridspec(1, 3, wspace=0.2)
    gs_metrics2 = gs_main[2].subgridspec(1, 3, wspace=0.2) 
    gs_metrics3 = gs_main[3].subgridspec(1, 3, wspace=0.2)
    
    subplot_positions = [
        (gs_metrics1, 0), (gs_metrics1, 1), (gs_metrics1, 2),
        (gs_metrics2, 0), (gs_metrics2, 1), (gs_metrics2, 2),
        (gs_metrics3, 0), (gs_metrics3, 1), (gs_metrics3, 2)
    ]
    
    scenarios = list(all_metrics.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios)))
    
    for i, (metric_name, metric_label) in enumerate(zip(metrics_names, metrics_labels)):
        gs_section, pos = subplot_positions[i]
        ax = fig.add_subplot(gs_section[pos])
        
        # Get values for this metric
        if metric_name == 'normalized_total_loss':
            values = [all_metrics[scenario].get(metric_name, 0.0) for scenario in scenarios]
        else:
            values = [all_metrics[scenario][metric_name] for scenario in scenarios]
        
        # Create bar plot
        bars = ax.bar(range(len(scenarios)), values, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=0.8)
        
        ax.set_title(metric_label, fontsize=11, fontweight='bold', pad=15)  # Increased padding
        ax.set_xticks(range(len(scenarios)))
        
        # Shorter labels for better fit
        short_labels = [s.replace(" (No Pred)", "\n(No P)").replace(" (With Pred)", "\n(With P)") 
                       for s in scenarios]
        ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars with better positioning
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height > 0:  # Only add labels for positive values
                if metric_name in ['resolution_success_rate', 'plan_adherence_rate']:
                    label = f'{value:.1f}%'
                elif metric_name in ['total_loss', 'normalized_total_loss']:
                    label = f'{value:.3f}'
                else:
                    label = f'{int(value)}'
                
                ax.text(bar.get_x() + bar.get_width()/2., height + max(height*0.01, 0.001),
                       label, ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Add main title with proper positioning to avoid overlap
    fig.suptitle("SACR Protocol + Planning: Complete Multi-Agent System Analysis with Comprehensive Metrics", 
                fontsize=16, fontweight='bold', y=0.975)  # Positioned to avoid overlap
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjusted rect for balanced spacing
    plt.show()

def plot_comprehensive_results_with_planning(scenario_results):
    """Create comprehensive visualization of all results with enhanced metrics and planning"""
    # First get all metrics for normalization
    all_metrics = {}
    for scenario_name, (agents, adversary, simulation) in scenario_results.items():
        all_metrics[scenario_name] = simulation.get_comprehensive_metrics()
    
    # Calculate normalized losses
    all_total_losses = [metrics['total_loss'] for metrics in all_metrics.values()]
    if len(all_total_losses) > 1:
        min_loss = min(all_total_losses)
        max_loss = max(all_total_losses)
        loss_range = max_loss - min_loss if max_loss != min_loss else 1
        
        for scenario_name, metrics in all_metrics.items():
            normalized_loss = (metrics['total_loss'] - min_loss) / loss_range
            all_metrics[scenario_name]['normalized_total_loss'] = normalized_loss
    else:
        for scenario_name, metrics in all_metrics.items():
            all_metrics[scenario_name]['normalized_total_loss'] = 0.0
    
    # Create the comprehensive visualization
    plot_comprehensive_metrics_with_planning(scenario_results, all_metrics)

# ============================================================================
# PLANNING SECTION - Main Execution with Planning
# ============================================================================

def run_single_scenario_with_planning(scenario_name: str, use_prediction: bool, scenario_type: str, max_steps: int):
    """Run a single scenario with full simulation planning integration"""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING: {scenario_name}")
    print(f"{'='*80}")
    
    sim = MASSimulationWithSACRAndPlanning(
        use_forward_prediction=use_prediction,
        scenario_type=scenario_type,
        max_steps=max_steps
    )
    
    agents, adversary = sim.run_simulation_with_planning(max_steps)
    
    return agents, adversary, sim

def run_all_scenarios_with_planning():
    """Run all six scenarios with planning integration"""
    print("🚀 Multi-Agent System with SACR Communication Protocol + Planning")
    print("🧠 Enhanced with AI Planning Integration + Comprehensive Metrics")
    print("=" * 80)
    
    # Define all six scenarios
    scenarios = {
        "Formation (No Pred)": {"pred": False, "type": "formation", "steps": HyperParameters.MAX_STEPS},
        "Formation (With Pred)": {"pred": True, "type": "formation", "steps": HyperParameters.MAX_STEPS},
        "Interference (No Pred)": {"pred": False, "type": "interference", "steps": HyperParameters.INTERFERENCE_STEPS},
        "Interference (With Pred)": {"pred": True, "type": "interference", "steps": HyperParameters.INTERFERENCE_STEPS},
        "Obstacle (No Pred)": {"pred": False, "type": "obstacle", "steps": HyperParameters.MAX_STEPS},
        "Obstacle (With Pred)": {"pred": True, "type": "obstacle", "steps": HyperParameters.MAX_STEPS},
    }
    
    scenario_results = {}
    
    for name, params in scenarios.items():
        agents, adversary, sim = run_single_scenario_with_planning(
            scenario_name=name,
            use_prediction=params["pred"],
            scenario_type=params["type"],
            max_steps=params["steps"]
        )
        scenario_results[name] = (agents, adversary, sim)
    
    # Print comprehensive analysis with all metrics
    all_metrics = print_comprehensive_analysis_with_planning(scenario_results)
    
    # Create comprehensive visualization with all metrics
    plot_comprehensive_results_with_planning(scenario_results)
    
    return scenario_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("🚀 Multi-Agent System with SACR Communication Protocol + Full Simulation Planning")
    print("🧠 Enhanced with Complete AI Planning Integration + Comprehensive Metrics")
    print("📊 All Requested Metrics: Proximity Violations, Exact Collisions, Messages, etc.")
    print("=" * 80)
    
    print("\nKey Features:")
    print("• Comprehensive metrics tracking and visualization")
    print("• All requested metrics: proximity violations, exact collisions, messages sent, etc.")
    print("• Normalized total loss calculations for fair comparison")
    print("• Enhanced planning effectiveness analysis")
    print("• Professional-quality bar charts for all metrics")
    
    print("\nChoose execution mode:")
    print("1. Run single scenario with full simulation planning + metrics")
    print("2. Run all scenarios with full simulation planning + comprehensive metrics")
    print("3. Run single scenario without planning (original)")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        print("\nAvailable scenarios:")
        print("1. Formation (No Prediction) - Up to 400 steps")
        print("2. Formation (With Prediction) - Up to 400 steps") 
        print("3. Interference (No Prediction) - Up to 600 steps")
        print("4. Interference (With Prediction) - Up to 600 steps")
        print("5. Obstacle (No Prediction) - Up to 400 steps")
        print("6. Obstacle (With Prediction) - Up to 400 steps")
        
        print("\nNote: Formation scenarios may complete early when objectives are met.")
        print("All requested metrics will be tracked and displayed.")
        
        scenario_choice = input("\nEnter scenario (1-6): ")
        
        scenario_map = {
            "1": ("Formation (No Pred)", False, "formation", HyperParameters.MAX_STEPS),
            "2": ("Formation (With Pred)", True, "formation", HyperParameters.MAX_STEPS),
            "3": ("Interference (No Pred)", False, "interference", HyperParameters.INTERFERENCE_STEPS),
            "4": ("Interference (With Pred)", True, "interference", HyperParameters.INTERFERENCE_STEPS),
            "5": ("Obstacle (No Pred)", False, "obstacle", HyperParameters.MAX_STEPS),
            "6": ("Obstacle (With Pred)", True, "obstacle", HyperParameters.MAX_STEPS),
        }
        
        if scenario_choice in scenario_map:
            name, pred, stype, steps = scenario_map[scenario_choice]
            agents, adversary, sim = run_single_scenario_with_planning(name, pred, stype, steps)
            
            # Show comprehensive metrics
            metrics = sim.get_comprehensive_metrics()
            print(f"\n📊 COMPREHENSIVE METRICS SUMMARY:")
            print(f"   🔹 PROXIMITY VIOLATIONS: {metrics['proximity_violations']}")
            print(f"   🔹 EXACT COLLISIONS: {metrics['exact_collisions']}")
            print(f"   🔹 MESSAGES SENT: {metrics['messages_sent']}")
            print(f"   🔹 CONFLICTS RESOLVED: {metrics['conflicts_resolved']}")
            print(f"   🔹 BEHAVIOR CHANGES: {metrics['behavior_changes']}")
            print(f"   🔹 RESOLUTION SUCCESS RATE: {metrics['resolution_success_rate']:.1f}%")
            print(f"   🔹 TOTAL LOSS: {metrics['total_loss']:.4f}")
            print(f"   🔹 PLAN ADHERENCE RATE: {metrics['plan_adherence_rate']:.1f}%")
            
            # Visualize results
            comm_stats = sim.communication_manager.stats
            viz = Visualizer(agents, adversary, name, stype, comm_stats)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            if stype == "obstacle":
                viz.plot_obstacle_avoidance_scenario(ax)
            else:
                viz.plot_trajectories(ax)
            plt.show()
        else:
            print("Invalid scenario choice!")
    
    elif choice == "2":
        scenario_results = run_all_scenarios_with_planning()
        
        print(f"\n🎉 All scenarios completed!")
        print(f"📊 Comprehensive metrics analysis and visualization generated.")
        print(f"📈 Check the plots for detailed performance comparison across all scenarios.")
    
    elif choice == "3":
        # Original functionality without planning
        print("Running original simulation without planning...")
        print("Note: This would require the original MASSimulationWithSACR class")
        print("Please use the original code file for non-planning simulations.")
    
    else:
        print("Invalid choice!")