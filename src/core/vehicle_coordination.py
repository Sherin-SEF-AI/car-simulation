"""
Vehicle AI Coordination System
Handles inter-vehicle communication, collision avoidance, and traffic behavior patterns
"""

from PyQt6.QtCore import QObject, pyqtSignal
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import math
import time

from .physics_engine import Vector3


class TrafficRule(Enum):
    """Traffic rules that vehicles must follow"""
    SPEED_LIMIT = "speed_limit"
    STOP_SIGN = "stop_sign"
    TRAFFIC_LIGHT = "traffic_light"
    YIELD = "yield"
    NO_OVERTAKING = "no_overtaking"
    LANE_CHANGE = "lane_change"
    EMERGENCY_VEHICLE = "emergency_vehicle"


class CoordinationMessage(Enum):
    """Types of coordination messages between vehicles"""
    COLLISION_WARNING = "collision_warning"
    LANE_CHANGE_REQUEST = "lane_change_request"
    PRIORITY_REQUEST = "priority_request"
    EMERGENCY_BRAKE = "emergency_brake"
    YIELD_REQUEST = "yield_request"
    TRAFFIC_UPDATE = "traffic_update"
    ROUTE_SHARING = "route_sharing"


class BehaviorPattern(Enum):
    """Traffic behavior patterns"""
    AGGRESSIVE = "aggressive"
    CAUTIOUS = "cautious"
    STANDARD = "standard"
    EMERGENCY = "emergency"
    TRAFFIC_FOLLOWING = "traffic_following"


@dataclass
class CoordinationData:
    """Data structure for vehicle coordination"""
    sender_id: str
    receiver_id: str
    message_type: CoordinationMessage
    priority: float
    timestamp: float
    data: Dict[str, Any]
    expires_at: float


@dataclass
class TrafficContext:
    """Current traffic context for decision making"""
    traffic_density: float
    average_speed: float
    weather_conditions: str
    visibility: float
    road_conditions: str
    time_of_day: float
    emergency_vehicles_present: bool


@dataclass
class ConflictResolution:
    """Result of conflict resolution between vehicles"""
    primary_vehicle_id: str
    secondary_vehicle_id: str
    resolution_type: str
    actions: Dict[str, Any]
    confidence: float


class VehicleCoordinator(QObject):
    """Manages coordination between multiple vehicles"""
    
    # Signals for coordination events
    coordination_established = pyqtSignal(str, str)  # vehicle1_id, vehicle2_id
    conflict_resolved = pyqtSignal(object)  # ConflictResolution
    emergency_detected = pyqtSignal(str, str)  # vehicle_id, emergency_type
    traffic_violation = pyqtSignal(str, str)  # vehicle_id, violation_type
    
    def __init__(self):
        super().__init__()
        
        # Message queue for inter-vehicle communication
        self.message_queue: List[CoordinationData] = []
        self.message_history: Dict[str, List[CoordinationData]] = {}
        
        # Traffic rules and context
        self.active_traffic_rules: Dict[str, Any] = {}
        self.traffic_context = TrafficContext(
            traffic_density=0.5,
            average_speed=15.0,
            weather_conditions="clear",
            visibility=100.0,
            road_conditions="dry",
            time_of_day=12.0,
            emergency_vehicles_present=False
        )
        
        # Coordination parameters
        self.communication_range = 50.0  # meters
        self.collision_prediction_time = 3.0  # seconds
        self.priority_decay_rate = 0.1  # per second
        self.message_ttl = 5.0  # seconds
        
        # Behavior patterns
        self.behavior_patterns = {
            BehaviorPattern.AGGRESSIVE: {
                'following_distance': 1.5,
                'lane_change_threshold': 0.3,
                'speed_tolerance': 1.2,
                'priority_assertion': 0.8
            },
            BehaviorPattern.CAUTIOUS: {
                'following_distance': 4.0,
                'lane_change_threshold': 0.8,
                'speed_tolerance': 0.8,
                'priority_assertion': 0.3
            },
            BehaviorPattern.STANDARD: {
                'following_distance': 2.5,
                'lane_change_threshold': 0.5,
                'speed_tolerance': 1.0,
                'priority_assertion': 0.5
            },
            BehaviorPattern.EMERGENCY: {
                'following_distance': 1.0,
                'lane_change_threshold': 0.1,
                'speed_tolerance': 2.0,
                'priority_assertion': 1.0
            }
        }
    
    def send_coordination_message(self, sender_id: str, receiver_id: str, 
                                message_type: CoordinationMessage, data: Dict[str, Any],
                                priority: float = 1.0) -> bool:
        """Send a coordination message between vehicles"""
        current_time = time.time()
        
        message = CoordinationData(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            priority=priority,
            timestamp=current_time,
            data=data,
            expires_at=current_time + self.message_ttl
        )
        
        self.message_queue.append(message)
        
        # Add to history
        if sender_id not in self.message_history:
            self.message_history[sender_id] = []
        self.message_history[sender_id].append(message)
        
        return True
    
    def process_coordination_messages(self, vehicles: Dict[str, Any]) -> List[CoordinationData]:
        """Process all pending coordination messages"""
        current_time = time.time()
        processed_messages = []
        
        # Remove expired messages
        self.message_queue = [msg for msg in self.message_queue if msg.expires_at > current_time]
        
        # Process each message
        for message in self.message_queue[:]:
            if message.receiver_id in vehicles:
                receiver_vehicle = vehicles[message.receiver_id]
                
                # Apply message to receiver
                self._apply_coordination_message(receiver_vehicle, message)
                processed_messages.append(message)
                
                # Remove processed message
                self.message_queue.remove(message)
        
        return processed_messages
    
    def _apply_coordination_message(self, vehicle: Any, message: CoordinationData):
        """Apply coordination message to a vehicle"""
        if message.message_type == CoordinationMessage.COLLISION_WARNING:
            self._handle_collision_warning(vehicle, message)
        elif message.message_type == CoordinationMessage.PRIORITY_REQUEST:
            self._handle_priority_request(vehicle, message)
        elif message.message_type == CoordinationMessage.LANE_CHANGE_REQUEST:
            self._handle_lane_change_request(vehicle, message)
        elif message.message_type == CoordinationMessage.EMERGENCY_BRAKE:
            self._handle_emergency_brake(vehicle, message)
        elif message.message_type == CoordinationMessage.YIELD_REQUEST:
            self._handle_yield_request(vehicle, message)
    
    def _handle_collision_warning(self, vehicle: Any, message: CoordinationData):
        """Handle collision warning message"""
        # Implement emergency braking or evasive maneuvers
        collision_distance = message.data.get('distance', 10.0)
        relative_speed = message.data.get('relative_speed', 0.0)
        
        if collision_distance < 5.0 and relative_speed > 2.0:
            # Emergency situation - apply maximum braking
            vehicle.set_control_input(-1.0, 0.0, 1.0)  # Full brake
            vehicle.behavior_state['emergency_braking'] = True
            vehicle.safety_score -= 5.0
    
    def _handle_priority_request(self, vehicle: Any, message: CoordinationData):
        """Handle priority request from another vehicle"""
        sender_priority = message.data.get('priority', 1.0)
        
        if sender_priority > vehicle.ai_priority:
            # Yield to higher priority vehicle
            vehicle.behavior_state['yielding_to'] = message.sender_id
            vehicle.behavior_state['yield_until'] = time.time() + 5.0  # Yield for 5 seconds
    
    def _handle_lane_change_request(self, vehicle: Any, message: CoordinationData):
        """Handle lane change request"""
        # Check if it's safe to allow lane change
        safe_distance = message.data.get('required_distance', 10.0)
        current_distance = message.data.get('current_distance', 0.0)
        
        if current_distance > safe_distance:
            # Send approval
            self.send_coordination_message(
                vehicle.id, message.sender_id,
                CoordinationMessage.LANE_CHANGE_REQUEST,
                {'approved': True, 'safe_distance': current_distance}
            )
        else:
            # Adjust speed to create space
            vehicle.behavior_state['creating_space_for'] = message.sender_id
    
    def _handle_emergency_brake(self, vehicle: Any, message: CoordinationData):
        """Handle emergency brake message"""
        # Immediately apply emergency braking
        vehicle.set_control_input(-1.0, 0.0, 1.0)
        vehicle.behavior_state['emergency_braking'] = True
        
        # Propagate emergency brake to nearby vehicles
        self.emergency_detected.emit(vehicle.id, "emergency_brake_cascade")
    
    def _handle_yield_request(self, vehicle: Any, message: CoordinationData):
        """Handle yield request"""
        yield_duration = message.data.get('duration', 3.0)
        vehicle.behavior_state['yielding'] = True
        vehicle.behavior_state['yield_until'] = time.time() + yield_duration
    
    def detect_potential_collisions(self, vehicles: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        """Detect potential collisions between vehicles"""
        potential_collisions = []
        vehicle_list = list(vehicles.values())
        
        for i, vehicle1 in enumerate(vehicle_list):
            for vehicle2 in vehicle_list[i+1:]:
                collision_time = self._calculate_collision_time(vehicle1, vehicle2)
                
                if 0 < collision_time < self.collision_prediction_time:
                    potential_collisions.append((vehicle1.id, vehicle2.id, collision_time))
        
        return potential_collisions
    
    def _calculate_collision_time(self, vehicle1: Any, vehicle2: Any) -> float:
        """Calculate time to collision between two vehicles"""
        # Simplified collision time calculation
        relative_position = vehicle2.physics.position - vehicle1.physics.position
        relative_velocity = vehicle2.physics.velocity - vehicle1.physics.velocity
        
        distance = relative_position.magnitude()
        relative_speed = relative_velocity.magnitude()
        
        if relative_speed < 0.1:  # Vehicles moving at similar speeds
            return float('inf')
        
        # Check if vehicles are approaching each other
        dot_product = relative_position.dot(relative_velocity)
        if dot_product > 0:  # Moving away from each other
            return float('inf')
        
        # Simple time to collision calculation
        collision_time = distance / relative_speed
        
        # Check if collision is likely (considering vehicle sizes)
        collision_threshold = 3.0  # meters
        if distance > collision_threshold:
            return collision_time
        else:
            return 0.1  # Imminent collision
    
    def resolve_conflicts(self, vehicles: Dict[str, Any]) -> List[ConflictResolution]:
        """Resolve conflicts between vehicles using priority-based system"""
        conflicts = self.detect_potential_collisions(vehicles)
        resolutions = []
        
        for vehicle1_id, vehicle2_id, collision_time in conflicts:
            vehicle1 = vehicles[vehicle1_id]
            vehicle2 = vehicles[vehicle2_id]
            
            resolution = self._resolve_vehicle_conflict(vehicle1, vehicle2, collision_time)
            if resolution:
                resolutions.append(resolution)
                self.conflict_resolved.emit(resolution)
        
        return resolutions
    
    def _resolve_vehicle_conflict(self, vehicle1: Any, vehicle2: Any, collision_time: float) -> Optional[ConflictResolution]:
        """Resolve conflict between two specific vehicles"""
        # Determine priority
        if vehicle1.ai_priority > vehicle2.ai_priority:
            primary, secondary = vehicle1, vehicle2
        elif vehicle2.ai_priority > vehicle1.ai_priority:
            primary, secondary = vehicle2, vehicle1
        else:
            # Equal priority - use other factors
            # Vehicle with higher speed gets priority (highway merging logic)
            if vehicle1.physics.velocity.magnitude() > vehicle2.physics.velocity.magnitude():
                primary, secondary = vehicle1, vehicle2
            else:
                primary, secondary = vehicle2, vehicle1
        
        # Determine resolution actions
        actions = {}
        
        if collision_time < 1.0:  # Imminent collision
            # Emergency actions
            actions[primary.id] = {'action': 'maintain_course', 'throttle': 0.5}
            actions[secondary.id] = {'action': 'emergency_brake', 'throttle': -1.0, 'brake': 1.0}
            resolution_type = "emergency_brake"
        elif collision_time < 2.0:  # Close collision
            # Evasive actions
            actions[primary.id] = {'action': 'slight_acceleration', 'throttle': 0.3}
            actions[secondary.id] = {'action': 'decelerate', 'throttle': -0.5}
            resolution_type = "speed_adjustment"
        else:  # Predicted collision
            # Preventive actions
            actions[primary.id] = {'action': 'maintain_course'}
            actions[secondary.id] = {'action': 'yield', 'throttle': -0.2}
            resolution_type = "yield"
        
        # Send coordination messages
        self.send_coordination_message(
            primary.id, secondary.id,
            CoordinationMessage.PRIORITY_REQUEST,
            {'priority': primary.ai_priority, 'action': resolution_type}
        )
        
        return ConflictResolution(
            primary_vehicle_id=primary.id,
            secondary_vehicle_id=secondary.id,
            resolution_type=resolution_type,
            actions=actions,
            confidence=0.8  # Confidence in the resolution
        )
    
    def apply_traffic_behavior_patterns(self, vehicle: Any, pattern: BehaviorPattern):
        """Apply traffic behavior pattern to a vehicle"""
        if pattern not in self.behavior_patterns:
            pattern = BehaviorPattern.STANDARD
        
        behavior_config = self.behavior_patterns[pattern]
        
        # Update vehicle behavior parameters
        vehicle.behavior_state['following_distance'] = behavior_config['following_distance']
        vehicle.behavior_state['lane_change_threshold'] = behavior_config['lane_change_threshold']
        vehicle.behavior_state['speed_tolerance'] = behavior_config['speed_tolerance']
        vehicle.behavior_state['priority_assertion'] = behavior_config['priority_assertion']
        vehicle.behavior_state['pattern'] = pattern.value
    
    def enforce_traffic_rules(self, vehicle: Any, rules: List[TrafficRule]) -> List[str]:
        """Enforce traffic rules for a vehicle"""
        violations = []
        
        for rule in rules:
            if rule == TrafficRule.SPEED_LIMIT:
                speed_limit = self.active_traffic_rules.get('speed_limit', 25.0)  # m/s
                current_speed = vehicle.physics.velocity.magnitude()
                
                if current_speed > speed_limit * 1.1:  # 10% tolerance
                    violations.append(f"Speed limit violation: {current_speed:.1f} m/s in {speed_limit:.1f} m/s zone")
                    vehicle.safety_score -= 2.0
                    self.traffic_violation.emit(vehicle.id, "speed_limit")
            
            elif rule == TrafficRule.EMERGENCY_VEHICLE:
                if self.traffic_context.emergency_vehicles_present:
                    # Check if vehicle is yielding to emergency vehicles
                    if 'yielding_to_emergency' not in vehicle.behavior_state:
                        violations.append("Failed to yield to emergency vehicle")
                        vehicle.safety_score -= 10.0
                        self.traffic_violation.emit(vehicle.id, "emergency_vehicle")
        
        return violations
    
    def update_traffic_context(self, vehicles: Dict[str, Any]):
        """Update traffic context based on current vehicle states"""
        if not vehicles:
            return
        
        # Calculate traffic density
        active_vehicles = [v for v in vehicles.values() if hasattr(v, 'state') and v.state.value == 'active']
        self.traffic_context.traffic_density = len(active_vehicles) / 50.0  # Normalized to max vehicles
        
        # Calculate average speed
        if active_vehicles:
            speeds = [v.physics.velocity.magnitude() for v in active_vehicles]
            self.traffic_context.average_speed = sum(speeds) / len(speeds)
        
        # Check for emergency vehicles
        emergency_present = any(v.vehicle_type == 'emergency' for v in active_vehicles)
        self.traffic_context.emergency_vehicles_present = emergency_present
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination system statistics"""
        current_time = time.time()
        
        # Count active messages
        active_messages = len([msg for msg in self.message_queue if msg.expires_at > current_time])
        
        # Count messages by type
        message_types = {}
        for msg in self.message_queue:
            msg_type = msg.message_type.value
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
        
        return {
            'active_messages': active_messages,
            'message_types': message_types,
            'traffic_context': {
                'density': self.traffic_context.traffic_density,
                'average_speed': self.traffic_context.average_speed,
                'emergency_present': self.traffic_context.emergency_vehicles_present
            },
            'communication_range': self.communication_range,
            'collision_prediction_time': self.collision_prediction_time
        }
    
    def reset(self):
        """Reset coordination system"""
        self.message_queue.clear()
        self.message_history.clear()
        self.active_traffic_rules.clear()
        
        # Reset traffic context
        self.traffic_context = TrafficContext(
            traffic_density=0.5,
            average_speed=15.0,
            weather_conditions="clear",
            visibility=100.0,
            road_conditions="dry",
            time_of_day=12.0,
            emergency_vehicles_present=False
        )