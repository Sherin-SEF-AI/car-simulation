"""
AI system for autonomous vehicle behavior and decision making
"""

from PyQt6.QtCore import QObject, pyqtSignal
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from .behavior_tree import BehaviorTree, BehaviorTreeSerializer, create_basic_driving_tree, NodeStatus

logger = logging.getLogger(__name__)

class AIState(Enum):
    IDLE = "idle"
    DRIVING = "driving"
    AVOIDING = "avoiding"
    PARKING = "parking"
    EMERGENCY = "emergency"

@dataclass
class AIDecision:
    vehicle_id: str
    action: str
    parameters: Dict[str, Any]
    confidence: float
    timestamp: float

class AISystem(QObject):
    """Manages AI behaviors and decision making for autonomous vehicles"""
    
    decision_made = pyqtSignal(str, dict)  # vehicle_id, decision
    behavior_changed = pyqtSignal(str, str)  # vehicle_id, new_behavior
    tree_execution_completed = pyqtSignal(str, str)  # vehicle_id, status
    
    def __init__(self):
        super().__init__()
        
        # AI state tracking
        self.vehicle_states = {}  # vehicle_id -> AIState
        self.behavior_trees = {}  # vehicle_id -> BehaviorTree
        self.sensor_data = {}     # vehicle_id -> sensor_readings
        self.vehicle_controls = {}  # vehicle_id -> control_outputs
        
        # Decision making parameters
        self.reaction_time = 0.2  # seconds
        self.safety_margin = 2.0  # meters
        
        # Behavior tree management
        self.tree_serializer = BehaviorTreeSerializer()
    
    def update(self, delta_time: float):
        """Update AI system and process decisions"""
        for vehicle_id in self.vehicle_states:
            self._process_vehicle_ai(vehicle_id, delta_time)
    
    def _process_vehicle_ai(self, vehicle_id: str, delta_time: float):
        """Process AI decision making for a specific vehicle using behavior trees"""
        if vehicle_id not in self.behavior_trees:
            return
        
        behavior_tree = self.behavior_trees[vehicle_id]
        
        # Update blackboard with current sensor data and vehicle state
        blackboard = behavior_tree.get_blackboard()
        
        # Update sensor data
        if vehicle_id in self.sensor_data:
            blackboard.set('sensor_data', self.sensor_data[vehicle_id])
        
        # Update vehicle state information
        blackboard.set('vehicle_id', vehicle_id)
        blackboard.set('ai_state', self.vehicle_states.get(vehicle_id, AIState.IDLE).value)
        blackboard.set('delta_time', delta_time)
        
        # Execute behavior tree
        try:
            status = behavior_tree.execute(delta_time)
            
            # Extract control outputs from blackboard
            controls = {
                'throttle': blackboard.get('throttle', 0.0),
                'brake': blackboard.get('brake', 0.0),
                'steering': blackboard.get('steering', 0.0),
                'target_speed': blackboard.get('target_speed', 0.0)
            }
            
            # Store control outputs
            self.vehicle_controls[vehicle_id] = controls
            
            # Emit decision signal
            decision = {
                'controls': controls,
                'status': status.value,
                'execution_count': behavior_tree.execution_count
            }
            self.decision_made.emit(vehicle_id, decision)
            
            # Emit completion signal if tree finished
            if status != NodeStatus.RUNNING:
                self.tree_execution_completed.emit(vehicle_id, status.value)
                
        except Exception as e:
            logger.error(f"Error executing behavior tree for vehicle {vehicle_id}: {e}")
            self.vehicle_states[vehicle_id] = AIState.EMERGENCY
    
    def register_vehicle(self, vehicle_id: str, behavior_tree: Optional[BehaviorTree] = None):
        """Register a new vehicle with the AI system"""
        self.vehicle_states[vehicle_id] = AIState.IDLE
        self.sensor_data[vehicle_id] = {}
        self.vehicle_controls[vehicle_id] = {
            'throttle': 0.0,
            'brake': 0.0,
            'steering': 0.0,
            'target_speed': 0.0
        }
        
        # Set behavior tree (use default if none provided)
        if behavior_tree is None:
            behavior_tree = create_basic_driving_tree()
        
        self.behavior_trees[vehicle_id] = behavior_tree
        logger.info(f"Registered vehicle {vehicle_id} with behavior tree: {behavior_tree.name}")
    
    def unregister_vehicle(self, vehicle_id: str):
        """Remove a vehicle from the AI system"""
        self.vehicle_states.pop(vehicle_id, None)
        self.behavior_trees.pop(vehicle_id, None)
        self.sensor_data.pop(vehicle_id, None)
        self.vehicle_controls.pop(vehicle_id, None)
        logger.info(f"Unregistered vehicle {vehicle_id}")
    
    def update_sensor_data(self, vehicle_id: str, sensor_readings: Dict[str, Any]):
        """Update sensor data for a vehicle"""
        if vehicle_id in self.sensor_data:
            self.sensor_data[vehicle_id] = sensor_readings
    
    def set_behavior_tree(self, vehicle_id: str, behavior_tree: BehaviorTree):
        """Set behavior tree for a vehicle"""
        if vehicle_id in self.behavior_trees:
            self.behavior_trees[vehicle_id] = behavior_tree
            logger.info(f"Updated behavior tree for vehicle {vehicle_id}: {behavior_tree.name}")
    
    def get_behavior_tree(self, vehicle_id: str) -> Optional[BehaviorTree]:
        """Get behavior tree for a vehicle"""
        return self.behavior_trees.get(vehicle_id)
    
    def create_behavior_tree(self, tree_definition: Dict[str, Any]) -> BehaviorTree:
        """Create a behavior tree from definition"""
        tree = BehaviorTree(tree_definition.get('name', 'CustomTree'))
        tree.from_dict(tree_definition)
        return tree
    
    def save_behavior_tree(self, vehicle_id: str, filepath: str):
        """Save vehicle's behavior tree to file"""
        if vehicle_id in self.behavior_trees:
            self.tree_serializer.save_to_file(self.behavior_trees[vehicle_id], filepath)
    
    def load_behavior_tree(self, vehicle_id: str, filepath: str):
        """Load behavior tree from file for vehicle"""
        tree = self.tree_serializer.load_from_file(filepath)
        self.set_behavior_tree(vehicle_id, tree)
    
    def get_vehicle_state(self, vehicle_id: str) -> AIState:
        """Get current AI state for a vehicle"""
        return self.vehicle_states.get(vehicle_id, AIState.IDLE)
    
    def set_vehicle_state(self, vehicle_id: str, state: AIState):
        """Set AI state for a vehicle"""
        if vehicle_id in self.vehicle_states:
            old_state = self.vehicle_states[vehicle_id]
            self.vehicle_states[vehicle_id] = state
            if old_state != state:
                self.behavior_changed.emit(vehicle_id, state.value)
    
    def get_vehicle_controls(self, vehicle_id: str) -> Dict[str, float]:
        """Get current control outputs for a vehicle"""
        return self.vehicle_controls.get(vehicle_id, {
            'throttle': 0.0,
            'brake': 0.0,
            'steering': 0.0,
            'target_speed': 0.0
        })
    
    def reset_vehicle(self, vehicle_id: str):
        """Reset AI state and behavior tree for a specific vehicle"""
        if vehicle_id in self.behavior_trees:
            self.behavior_trees[vehicle_id].reset()
        
        if vehicle_id in self.vehicle_states:
            self.vehicle_states[vehicle_id] = AIState.IDLE
        
        if vehicle_id in self.vehicle_controls:
            self.vehicle_controls[vehicle_id] = {
                'throttle': 0.0,
                'brake': 0.0,
                'steering': 0.0,
                'target_speed': 0.0
            }
    
    def reset(self):
        """Reset AI system"""
        # Reset all behavior trees
        for tree in self.behavior_trees.values():
            tree.reset()
        
        # Reset all states
        for vehicle_id in self.vehicle_states:
            self.vehicle_states[vehicle_id] = AIState.IDLE
        
        # Reset all controls
        for vehicle_id in self.vehicle_controls:
            self.vehicle_controls[vehicle_id] = {
                'throttle': 0.0,
                'brake': 0.0,
                'steering': 0.0,
                'target_speed': 0.0
            }
        
        logger.info("AI system reset completed")