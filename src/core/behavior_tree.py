"""
Behavior Tree system for autonomous vehicle AI decision making
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json
import time
import logging

logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    """Status returned by behavior tree nodes"""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"

class NodeType(Enum):
    """Types of behavior tree nodes"""
    COMPOSITE = "composite"
    DECORATOR = "decorator"
    CONDITION = "condition"
    ACTION = "action"

@dataclass
class BlackboardData:
    """Shared data storage for behavior tree execution"""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from blackboard"""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set value in blackboard"""
        self.data[key] = value
    
    def has(self, key: str) -> bool:
        """Check if key exists in blackboard"""
        return key in self.data
    
    def clear(self):
        """Clear all blackboard data"""
        self.data.clear()

class BehaviorNode(ABC):
    """Base class for all behavior tree nodes"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.children: List['BehaviorNode'] = []
        self.parent: Optional['BehaviorNode'] = None
        self.status = NodeStatus.FAILURE
        self.last_execution_time = 0.0
    
    @abstractmethod
    def execute(self, blackboard: BlackboardData, delta_time: float) -> NodeStatus:
        """Execute the node and return status"""
        pass
    
    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        """Return the type of this node"""
        pass
    
    def add_child(self, child: 'BehaviorNode'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child: 'BehaviorNode'):
        """Remove a child node"""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
    
    def reset(self):
        """Reset node state"""
        self.status = NodeStatus.FAILURE
        for child in self.children:
            child.reset()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary"""
        return {
            'name': self.name,
            'type': self.node_type.value,
            'parameters': self.parameters,
            'children': [child.to_dict() for child in self.children]
        }

# Composite Nodes
class SequenceNode(BehaviorNode):
    """Executes children in sequence until one fails"""
    
    def __init__(self, name: str = "Sequence", parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.current_child_index = 0
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.COMPOSITE
    
    def execute(self, blackboard: BlackboardData, delta_time: float) -> NodeStatus:
        """Execute children in sequence"""
        while self.current_child_index < len(self.children):
            child = self.children[self.current_child_index]
            status = child.execute(blackboard, delta_time)
            
            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return NodeStatus.RUNNING
            elif status == NodeStatus.FAILURE:
                self.reset()
                self.status = NodeStatus.FAILURE
                return NodeStatus.FAILURE
            else:  # SUCCESS
                self.current_child_index += 1
        
        # All children succeeded
        self.reset()
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS
    
    def reset(self):
        super().reset()
        self.current_child_index = 0

class SelectorNode(BehaviorNode):
    """Executes children until one succeeds"""
    
    def __init__(self, name: str = "Selector", parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.current_child_index = 0
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.COMPOSITE
    
    def execute(self, blackboard: BlackboardData, delta_time: float) -> NodeStatus:
        """Execute children until one succeeds"""
        while self.current_child_index < len(self.children):
            child = self.children[self.current_child_index]
            status = child.execute(blackboard, delta_time)
            
            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return NodeStatus.RUNNING
            elif status == NodeStatus.SUCCESS:
                self.reset()
                self.status = NodeStatus.SUCCESS
                return NodeStatus.SUCCESS
            else:  # FAILURE
                self.current_child_index += 1
        
        # All children failed
        self.reset()
        self.status = NodeStatus.FAILURE
        return NodeStatus.FAILURE
    
    def reset(self):
        super().reset()
        self.current_child_index = 0

class ParallelNode(BehaviorNode):
    """Executes all children simultaneously"""
    
    def __init__(self, name: str = "Parallel", parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.success_threshold = parameters.get('success_threshold', 1) if parameters else 1
        self.failure_threshold = parameters.get('failure_threshold', 1) if parameters else 1
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.COMPOSITE
    
    def execute(self, blackboard: BlackboardData, delta_time: float) -> NodeStatus:
        """Execute all children in parallel"""
        success_count = 0
        failure_count = 0
        running_count = 0
        
        for child in self.children:
            status = child.execute(blackboard, delta_time)
            
            if status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.FAILURE:
                failure_count += 1
            else:  # RUNNING
                running_count += 1
        
        # Check thresholds
        if success_count >= self.success_threshold:
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS
        elif failure_count >= self.failure_threshold:
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE
        else:
            self.status = NodeStatus.RUNNING
            return NodeStatus.RUNNING

# Decorator Nodes
class InverterNode(BehaviorNode):
    """Inverts the result of its child"""
    
    def __init__(self, name: str = "Inverter", parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.DECORATOR
    
    def execute(self, blackboard: BlackboardData, delta_time: float) -> NodeStatus:
        """Invert child result"""
        if not self.children:
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE
        
        child_status = self.children[0].execute(blackboard, delta_time)
        
        if child_status == NodeStatus.SUCCESS:
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE
        elif child_status == NodeStatus.FAILURE:
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS
        else:  # RUNNING
            self.status = NodeStatus.RUNNING
            return NodeStatus.RUNNING

class RepeatNode(BehaviorNode):
    """Repeats its child a specified number of times"""
    
    def __init__(self, name: str = "Repeat", parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.max_repeats = parameters.get('max_repeats', 1) if parameters else 1
        self.current_repeats = 0
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.DECORATOR
    
    def execute(self, blackboard: BlackboardData, delta_time: float) -> NodeStatus:
        """Repeat child execution"""
        if not self.children:
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE
        
        while self.current_repeats < self.max_repeats:
            child_status = self.children[0].execute(blackboard, delta_time)
            
            if child_status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return NodeStatus.RUNNING
            elif child_status == NodeStatus.FAILURE:
                self.reset()
                self.status = NodeStatus.FAILURE
                return NodeStatus.FAILURE
            else:  # SUCCESS
                self.current_repeats += 1
                self.children[0].reset()
        
        # All repeats completed successfully
        self.reset()
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS
    
    def reset(self):
        super().reset()
        self.current_repeats = 0

# Condition Nodes (Perception)
class ConditionNode(BehaviorNode):
    """Base class for condition nodes"""
    
    def __init__(self, name: str, condition_func: Callable[[BlackboardData], bool], parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.condition_func = condition_func
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.CONDITION
    
    def execute(self, blackboard: BlackboardData, delta_time: float) -> NodeStatus:
        """Execute condition check"""
        try:
            if self.condition_func(blackboard):
                self.status = NodeStatus.SUCCESS
                return NodeStatus.SUCCESS
            else:
                self.status = NodeStatus.FAILURE
                return NodeStatus.FAILURE
        except Exception as e:
            logger.error(f"Condition node {self.name} failed with error: {e}")
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE

class ObstacleDetectedCondition(ConditionNode):
    """Checks if obstacles are detected in front of vehicle"""
    
    def __init__(self, name: str = "ObstacleDetected", parameters: Dict[str, Any] = None):
        def check_obstacle(blackboard: BlackboardData) -> bool:
            sensor_data = blackboard.get('sensor_data', {})
            lidar_data = sensor_data.get('lidar', {})
            min_distance = parameters.get('min_distance', 5.0) if parameters else 5.0
            
            # Check if any obstacle is closer than minimum distance
            distances = lidar_data.get('distances', [])
            return any(d < min_distance for d in distances if d > 0)
        
        super().__init__(name, check_obstacle, parameters)

class SpeedLimitCondition(ConditionNode):
    """Checks if vehicle is within speed limit"""
    
    def __init__(self, name: str = "WithinSpeedLimit", parameters: Dict[str, Any] = None):
        def check_speed(blackboard: BlackboardData) -> bool:
            current_speed = blackboard.get('current_speed', 0.0)
            speed_limit = blackboard.get('speed_limit', 50.0)  # km/h
            tolerance = parameters.get('tolerance', 5.0) if parameters else 5.0
            
            return current_speed <= (speed_limit + tolerance)
        
        super().__init__(name, check_speed, parameters)

# Action Nodes
class ActionNode(BehaviorNode):
    """Base class for action nodes"""
    
    def __init__(self, name: str, action_func: Callable[[BlackboardData, float], NodeStatus], parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.action_func = action_func
    
    @property
    def node_type(self) -> NodeType:
        return NodeType.ACTION
    
    def execute(self, blackboard: BlackboardData, delta_time: float) -> NodeStatus:
        """Execute action"""
        try:
            status = self.action_func(blackboard, delta_time)
            self.status = status
            return status
        except Exception as e:
            logger.error(f"Action node {self.name} failed with error: {e}")
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE

class DriveForwardAction(ActionNode):
    """Action to drive forward at specified speed"""
    
    def __init__(self, name: str = "DriveForward", parameters: Dict[str, Any] = None):
        def drive_forward(blackboard: BlackboardData, delta_time: float) -> NodeStatus:
            target_speed = parameters.get('target_speed', 30.0) if parameters else 30.0
            
            # Set vehicle controls
            blackboard.set('throttle', 0.5)
            blackboard.set('brake', 0.0)
            blackboard.set('steering', 0.0)
            blackboard.set('target_speed', target_speed)
            
            return NodeStatus.SUCCESS
        
        super().__init__(name, drive_forward, parameters)

class BrakeAction(ActionNode):
    """Action to apply brakes"""
    
    def __init__(self, name: str = "Brake", parameters: Dict[str, Any] = None):
        def apply_brake(blackboard: BlackboardData, delta_time: float) -> NodeStatus:
            brake_force = parameters.get('brake_force', 0.8) if parameters else 0.8
            
            # Set vehicle controls
            blackboard.set('throttle', 0.0)
            blackboard.set('brake', brake_force)
            blackboard.set('steering', 0.0)
            
            return NodeStatus.SUCCESS
        
        super().__init__(name, apply_brake, parameters)

class SteerAction(ActionNode):
    """Action to steer vehicle"""
    
    def __init__(self, name: str = "Steer", parameters: Dict[str, Any] = None):
        def steer_vehicle(blackboard: BlackboardData, delta_time: float) -> NodeStatus:
            steering_angle = parameters.get('steering_angle', 0.0) if parameters else 0.0
            
            # Set vehicle controls
            blackboard.set('steering', steering_angle)
            
            return NodeStatus.SUCCESS
        
        super().__init__(name, steer_vehicle, parameters)

class BehaviorTree:
    """Main behavior tree class that manages execution"""
    
    def __init__(self, name: str = "BehaviorTree"):
        self.name = name
        self.root: Optional[BehaviorNode] = None
        self.blackboard = BlackboardData()
        self.is_running = False
        self.execution_count = 0
    
    def set_root(self, root_node: BehaviorNode):
        """Set the root node of the behavior tree"""
        self.root = root_node
    
    def execute(self, delta_time: float) -> NodeStatus:
        """Execute the behavior tree"""
        if not self.root:
            return NodeStatus.FAILURE
        
        self.execution_count += 1
        status = self.root.execute(self.blackboard, delta_time)
        
        # Reset tree if completed (success or failure)
        if status != NodeStatus.RUNNING:
            self.root.reset()
        
        return status
    
    def reset(self):
        """Reset the behavior tree"""
        if self.root:
            self.root.reset()
        self.blackboard.clear()
        self.execution_count = 0
    
    def get_blackboard(self) -> BlackboardData:
        """Get the behavior tree's blackboard"""
        return self.blackboard
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize behavior tree to dictionary"""
        return {
            'name': self.name,
            'root': self.root.to_dict() if self.root else None,
            'execution_count': self.execution_count
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Deserialize behavior tree from dictionary"""
        self.name = data.get('name', 'BehaviorTree')
        self.execution_count = data.get('execution_count', 0)
        
        root_data = data.get('root')
        if root_data:
            self.root = self._create_node_from_dict(root_data)
    
    def _create_node_from_dict(self, node_data: Dict[str, Any]) -> BehaviorNode:
        """Create a behavior node from dictionary data"""
        node_type = node_data.get('type')
        name = node_data.get('name')
        parameters = node_data.get('parameters', {})
        
        # Create node based on type
        if node_type == NodeType.COMPOSITE.value:
            if 'Sequence' in name:
                node = SequenceNode(name, parameters)
            elif 'Selector' in name:
                node = SelectorNode(name, parameters)
            elif 'Parallel' in name:
                node = ParallelNode(name, parameters)
            else:
                node = SequenceNode(name, parameters)  # Default
        elif node_type == NodeType.DECORATOR.value:
            if 'Inverter' in name:
                node = InverterNode(name, parameters)
            elif 'Repeat' in name:
                node = RepeatNode(name, parameters)
            else:
                node = InverterNode(name, parameters)  # Default
        else:
            # For condition and action nodes, create generic nodes
            # In a full implementation, you'd have a registry of node types
            node = SequenceNode(name, parameters)
        
        # Add children
        for child_data in node_data.get('children', []):
            child_node = self._create_node_from_dict(child_data)
            node.add_child(child_node)
        
        return node

class BehaviorTreeSerializer:
    """Handles serialization and deserialization of behavior trees"""
    
    @staticmethod
    def save_to_file(behavior_tree: BehaviorTree, filepath: str):
        """Save behavior tree to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(behavior_tree.to_dict(), f, indent=2)
            logger.info(f"Behavior tree saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save behavior tree to {filepath}: {e}")
            raise
    
    @staticmethod
    def load_from_file(filepath: str) -> BehaviorTree:
        """Load behavior tree from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            behavior_tree = BehaviorTree()
            behavior_tree.from_dict(data)
            logger.info(f"Behavior tree loaded from {filepath}")
            return behavior_tree
        except Exception as e:
            logger.error(f"Failed to load behavior tree from {filepath}: {e}")
            raise

def create_basic_driving_tree() -> BehaviorTree:
    """Create a basic driving behavior tree for testing"""
    tree = BehaviorTree("BasicDriving")
    
    # Root selector: choose between obstacle avoidance and normal driving
    root = SelectorNode("MainSelector")
    
    # Obstacle avoidance sequence
    obstacle_sequence = SequenceNode("ObstacleAvoidance")
    obstacle_condition = ObstacleDetectedCondition("CheckObstacle", {'min_distance': 10.0})
    brake_action = BrakeAction("EmergencyBrake", {'brake_force': 1.0})
    
    obstacle_sequence.add_child(obstacle_condition)
    obstacle_sequence.add_child(brake_action)
    
    # Normal driving sequence
    normal_sequence = SequenceNode("NormalDriving")
    speed_condition = SpeedLimitCondition("CheckSpeed", {'tolerance': 2.0})
    drive_action = DriveForwardAction("CruiseForward", {'target_speed': 50.0})
    
    normal_sequence.add_child(speed_condition)
    normal_sequence.add_child(drive_action)
    
    # Add sequences to root
    root.add_child(obstacle_sequence)
    root.add_child(normal_sequence)
    
    tree.set_root(root)
    return tree