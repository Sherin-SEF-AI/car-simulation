"""
Complete AI Integration System
Integrates advanced AI with physics engine and provides comprehensive autonomous vehicle control
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .advanced_ai_system import AdvancedAISystem, DriverProfile, DrivingBehavior
from .complete_physics_engine import CompletePhysicsEngine


class AutonomyLevel(Enum):
    """SAE levels of driving automation"""
    MANUAL = 0          # No automation
    ASSISTED = 1        # Driver assistance
    PARTIAL = 2         # Partial automation
    CONDITIONAL = 3     # Conditional automation
    HIGH = 4           # High automation
    FULL = 5           # Full automation


@dataclass
class AIVehicleConfig:
    """Configuration for AI-controlled vehicle"""
    vehicle_id: str
    autonomy_level: AutonomyLevel
    driver_profile: DriverProfile
    learning_enabled: bool = True
    safety_override: bool = True
    max_speed: float = 120.0  # km/h
    comfort_level: float = 0.8  # 0-1


class CompleteAIIntegration:
    """Complete AI integration system"""
    
    def __init__(self, physics_engine: CompletePhysicsEngine):
        self.physics_engine = physics_engine
        self.ai_system = AdvancedAISystem()
        
        # AI vehicle configurations
        self.ai_vehicles: Dict[str, AIVehicleConfig] = {}
        
        # Control systems
        self.path_planners: Dict[str, 'PathPlanner'] = {}
        self.behavior_trees: Dict[str, 'BehaviorTree'] = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_maneuvers': 0,
            'emergency_interventions': 0,
            'average_reaction_time': 0.0,
            'fuel_efficiency': 0.0
        }
        
        # Threading
        self.running = False
        self.ai_thread = None
        self.update_frequency = 30  # Hz
        self.lock = threading.Lock()
        
        # Safety systems
        self.collision_avoidance_enabled = True
        self.emergency_brake_threshold = 2.0  # seconds to collision
        
    def add_ai_vehicle(self, vehicle_id: str, config: AIVehicleConfig):
        """Add an AI-controlled vehicle"""
        
        with self.lock:
            self.ai_vehicles[vehicle_id] = config
            
            # Add to AI system
            self.ai_system.add_ai_vehicle(vehicle_id, config.driver_profile)
            
            # Create path planner
            self.path_planners[vehicle_id] = PathPlanner(vehicle_id)
            
            # Create behavior tree
            self.behavior_trees[vehicle_id] = BehaviorTree(vehicle_id, config)
            
            print(f"Added AI vehicle {vehicle_id} with autonomy level {config.autonomy_level.name}")
    
    def remove_ai_vehicle(self, vehicle_id: str):
        """Remove an AI-controlled vehicle"""
        
        with self.lock:
            if vehicle_id in self.ai_vehicles:
                del self.ai_vehicles[vehicle_id]
                self.ai_system.remove_ai_vehicle(vehicle_id)
                
                if vehicle_id in self.path_planners:
                    del self.path_planners[vehicle_id]
                
                if vehicle_id in self.behavior_trees:
                    del self.behavior_trees[vehicle_id]
    
    def set_destination(self, vehicle_id: str, destination: Tuple[float, float]):
        """Set destination for AI vehicle"""
        
        if vehicle_id in self.path_planners:
            self.path_planners[vehicle_id].set_destination(destination)
    
    def update_ai_decisions(self):
        """Update AI decisions for all vehicles"""
        
        # Get current world state
        world_state = {
            'vehicles': self.physics_engine.get_all_vehicle_states(),
            'weather': self.physics_engine.weather_conditions,
            'timestamp': time.time()
        }
        
        # Update AI system with world state
        self.ai_system.update_world_state(world_state)
        self.ai_system.update_weather_conditions(world_state['weather'])
        
        # Process each AI vehicle
        for vehicle_id, config in self.ai_vehicles.items():
            try:
                # Get AI decisions
                decisions = self.ai_system.get_vehicle_decisions(vehicle_id)
                
                if decisions:
                    # Apply autonomy level filtering
                    filtered_decisions = self._apply_autonomy_filtering(
                        vehicle_id, decisions, config.autonomy_level
                    )
                    
                    # Apply safety checks
                    safe_decisions = self._apply_safety_checks(
                        vehicle_id, filtered_decisions, world_state
                    )
                    
                    # Update physics engine with control inputs
                    self.physics_engine.update_vehicle_controls(
                        vehicle_id,
                        safe_decisions.get('throttle', 0.0),
                        safe_decisions.get('brake', 0.0),
                        safe_decisions.get('steering', 0.0)
                    )
                    
                    # Update performance metrics
                    self._update_performance_metrics(vehicle_id, safe_decisions)
                    
            except Exception as e:
                print(f"Error processing AI for vehicle {vehicle_id}: {e}")
    
    def _apply_autonomy_filtering(self, vehicle_id: str, decisions: Dict, 
                                level: AutonomyLevel) -> Dict:
        """Filter decisions based on autonomy level"""
        
        if level == AutonomyLevel.MANUAL:
            # No AI control
            return {'throttle': 0.0, 'brake': 0.0, 'steering': 0.0}
        
        elif level == AutonomyLevel.ASSISTED:
            # Only safety interventions
            filtered = {'throttle': 0.0, 'brake': 0.0, 'steering': 0.0}
            
            # Emergency braking only
            if decisions.get('emergency_stop', False):
                filtered['brake'] = 1.0
            
            return filtered
        
        elif level == AutonomyLevel.PARTIAL:
            # Limited speed and steering control
            return {
                'throttle': min(0.5, decisions.get('throttle', 0.0)),
                'brake': decisions.get('brake', 0.0),
                'steering': np.clip(decisions.get('steering', 0.0), -0.5, 0.5)
            }
        
        elif level in [AutonomyLevel.CONDITIONAL, AutonomyLevel.HIGH, AutonomyLevel.FULL]:
            # Full AI control
            return decisions
        
        return decisions
    
    def _apply_safety_checks(self, vehicle_id: str, decisions: Dict, 
                           world_state: Dict) -> Dict:
        """Apply safety checks and override dangerous decisions"""
        
        if not self.collision_avoidance_enabled:
            return decisions
        
        vehicle_state = world_state['vehicles'].get(vehicle_id)
        if not vehicle_state:
            return decisions
        
        # Check for imminent collisions
        collision_risk = self._assess_collision_risk(vehicle_id, world_state)
        
        if collision_risk > 0.8:  # High collision risk
            # Emergency braking
            decisions['throttle'] = 0.0
            decisions['brake'] = 1.0
            decisions['emergency_stop'] = True
            
            self.performance_metrics['emergency_interventions'] += 1
            
            print(f"Emergency intervention for vehicle {vehicle_id}: collision risk {collision_risk:.2f}")
        
        elif collision_risk > 0.5:  # Moderate collision risk
            # Reduce throttle and prepare to brake
            decisions['throttle'] *= 0.5
            decisions['brake'] = max(decisions.get('brake', 0.0), 0.3)
        
        return decisions
    
    def _assess_collision_risk(self, vehicle_id: str, world_state: Dict) -> float:
        """Assess collision risk for a vehicle"""
        
        vehicle_state = world_state['vehicles'].get(vehicle_id)
        if not vehicle_state:
            return 0.0
        
        vehicle_pos = np.array(vehicle_state['position'][:2])
        vehicle_vel = np.array(vehicle_state['velocity'][:2])
        vehicle_speed = np.linalg.norm(vehicle_vel)
        
        max_risk = 0.0
        
        # Check collision risk with other vehicles
        for other_id, other_state in world_state['vehicles'].items():
            if other_id == vehicle_id:
                continue
            
            other_pos = np.array(other_state['position'][:2])
            other_vel = np.array(other_state['velocity'][:2])
            
            # Calculate relative position and velocity
            relative_pos = other_pos - vehicle_pos
            relative_vel = other_vel - vehicle_vel
            
            distance = np.linalg.norm(relative_pos)
            
            # Skip if too far away
            if distance > 50.0:
                continue
            
            # Calculate time to collision
            if np.dot(relative_pos, relative_vel) < 0:  # Approaching
                relative_speed = np.linalg.norm(relative_vel)
                if relative_speed > 0.1:
                    time_to_collision = distance / relative_speed
                    
                    # Calculate risk based on time to collision
                    if time_to_collision < self.emergency_brake_threshold:
                        risk = 1.0 - (time_to_collision / self.emergency_brake_threshold)
                        max_risk = max(max_risk, risk)
        
        return max_risk
    
    def _update_performance_metrics(self, vehicle_id: str, decisions: Dict):
        """Update performance metrics"""
        
        self.performance_metrics['total_decisions'] += 1
        
        # Update reaction time (simplified)
        if hasattr(self, 'last_decision_time'):
            reaction_time = time.time() - self.last_decision_time
            self.performance_metrics['average_reaction_time'] = (
                self.performance_metrics['average_reaction_time'] * 0.9 + 
                reaction_time * 0.1
            )
        
        self.last_decision_time = time.time()
    
    def start_ai_integration(self):
        """Start AI integration system"""
        
        if not self.running:
            self.running = True
            
            # Start AI system
            self.ai_system.start_ai_system()
            
            # Start integration thread
            self.ai_thread = threading.Thread(target=self._integration_loop)
            self.ai_thread.daemon = True
            self.ai_thread.start()
            
            print("AI integration system started")
    
    def stop_ai_integration(self):
        """Stop AI integration system"""
        
        self.running = False
        
        # Stop AI system
        self.ai_system.stop_ai_system()
        
        # Stop integration thread
        if self.ai_thread:
            self.ai_thread.join()
        
        print("AI integration system stopped")
    
    def _integration_loop(self):
        """Main integration loop"""
        
        dt = 1.0 / self.update_frequency
        
        while self.running:
            start_time = time.time()
            
            try:
                self.update_ai_decisions()
            except Exception as e:
                print(f"Error in AI integration loop: {e}")
            
            # Maintain update frequency
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    def get_ai_status(self) -> Dict:
        """Get comprehensive AI system status"""
        
        with self.lock:
            ai_stats = self.ai_system.get_ai_statistics()
            
            return {
                'ai_vehicles': len(self.ai_vehicles),
                'autonomy_levels': {
                    level.name: sum(1 for config in self.ai_vehicles.values() 
                                  if config.autonomy_level == level)
                    for level in AutonomyLevel
                },
                'performance_metrics': self.performance_metrics.copy(),
                'ai_system_stats': ai_stats,
                'safety_systems': {
                    'collision_avoidance': self.collision_avoidance_enabled,
                    'emergency_brake_threshold': self.emergency_brake_threshold
                }
            }
    
    def set_autonomy_level(self, vehicle_id: str, level: AutonomyLevel):
        """Change autonomy level for a vehicle"""
        
        with self.lock:
            if vehicle_id in self.ai_vehicles:
                self.ai_vehicles[vehicle_id].autonomy_level = level
                print(f"Set autonomy level for {vehicle_id} to {level.name}")
    
    def enable_learning(self, vehicle_id: str, enabled: bool = True):
        """Enable/disable learning for a vehicle"""
        
        with self.lock:
            if vehicle_id in self.ai_vehicles:
                self.ai_vehicles[vehicle_id].learning_enabled = enabled
                print(f"Learning {'enabled' if enabled else 'disabled'} for {vehicle_id}")


class PathPlanner:
    """Simple path planner for AI vehicles"""
    
    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.destination = None
        self.current_path = []
        self.path_index = 0
    
    def set_destination(self, destination: Tuple[float, float]):
        """Set destination and plan path"""
        self.destination = destination
        # Simplified: direct path to destination
        self.current_path = [destination]
        self.path_index = 0
    
    def get_next_waypoint(self, current_position: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Get next waypoint on the path"""
        if not self.current_path or self.path_index >= len(self.current_path):
            return None
        
        return self.current_path[self.path_index]


class BehaviorTree:
    """Simple behavior tree for AI vehicles"""
    
    def __init__(self, vehicle_id: str, config: AIVehicleConfig):
        self.vehicle_id = vehicle_id
        self.config = config
    
    def execute(self, world_state: Dict) -> Dict:
        """Execute behavior tree and return decisions"""
        
        # Simplified behavior tree execution
        decisions = {
            'throttle': 0.3,
            'brake': 0.0,
            'steering': 0.0,
            'emergency_stop': False
        }
        
        return decisions