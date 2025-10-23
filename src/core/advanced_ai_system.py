"""
Advanced AI System with Neural Networks, Reinforcement Learning, and Realistic Decision Making
"""

import numpy as np
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import threading
import time
import json


class DrivingBehavior(Enum):
    """Different driving behavior types"""
    AGGRESSIVE = "aggressive"
    NORMAL = "normal"
    CAUTIOUS = "cautious"
    ELDERLY = "elderly"
    INEXPERIENCED = "inexperienced"
    PROFESSIONAL = "professional"
    EMERGENCY = "emergency"


@dataclass
class DriverProfile:
    """Comprehensive driver profile"""
    behavior_type: DrivingBehavior
    reaction_time: float  # seconds
    risk_tolerance: float  # 0-1
    speed_preference: float  # multiplier
    following_distance: float  # seconds
    lane_change_frequency: float  # changes per km
    attention_level: float  # 0-1
    fatigue_level: float  # 0-1
    experience_years: int
    
    # Skill levels (0-1)
    steering_skill: float = 0.8
    braking_skill: float = 0.8
    acceleration_skill: float = 0.8
    situational_awareness: float = 0.8


class NeuralNetwork:
    """Simple neural network for AI decision making"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        self.layers = []
        
        # Initialize weights and biases
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append({
                'weights': np.random.randn(prev_size, hidden_size) * 0.1,
                'biases': np.zeros(hidden_size)
            })
            prev_size = hidden_size
        
        # Output layer
        self.layers.append({
            'weights': np.random.randn(prev_size, output_size) * 0.1,
            'biases': np.zeros(output_size)
        })
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        x = inputs
        
        for i, layer in enumerate(self.layers):
            x = np.dot(x, layer['weights']) + layer['biases']
            
            # Apply activation function (ReLU for hidden layers, tanh for output)
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)  # ReLU
            else:
                x = np.tanh(x)  # Tanh for output
        
        return x
    
    def update_weights(self, learning_rate: float, gradients: List[np.ndarray]):
        """Update network weights (simplified)"""
        for i, gradient in enumerate(gradients):
            if i < len(self.layers):
                self.layers[i]['weights'] -= learning_rate * gradient
class PerceptionSystem:
    """Advanced perception system for AI vehicles"""
    
    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        
        # Sensor ranges and capabilities
        self.vision_range = 150.0  # meters
        self.vision_fov = math.radians(120)  # field of view
        self.radar_range = 200.0
        self.lidar_range = 100.0
        self.lidar_resolution = 0.1  # degrees
        
        # Perception state
        self.detected_vehicles = {}
        self.detected_obstacles = {}
        self.road_boundaries = []
        self.traffic_signs = []
        self.traffic_lights = []
        
        # Uncertainty and noise
        self.position_uncertainty = 0.1  # meters
        self.velocity_uncertainty = 0.05  # m/s
        self.detection_probability = 0.95
        
    def update_perception(self, world_state: Dict, vehicle_position: np.ndarray,
                         vehicle_orientation: float, weather_conditions: Dict) -> Dict:
        """Update perception based on world state"""
        
        perception_data = {
            'vehicles': [],
            'obstacles': [],
            'road_info': {},
            'traffic_info': {},
            'hazards': []
        }
        
        # Weather effects on perception
        visibility_factor = self._calculate_visibility_factor(weather_conditions)
        effective_range = self.vision_range * visibility_factor
        
        # Detect other vehicles
        for other_id, other_vehicle in world_state.get('vehicles', {}).items():
            if other_id == self.vehicle_id:
                continue
            
            other_pos = np.array(other_vehicle['position'][:2])
            distance = np.linalg.norm(other_pos - vehicle_position[:2])
            
            if distance <= effective_range:
                # Check if in field of view
                relative_pos = other_pos - vehicle_position[:2]
                angle_to_vehicle = math.atan2(relative_pos[1], relative_pos[0])
                angle_diff = abs(angle_to_vehicle - vehicle_orientation)
                
                if angle_diff <= self.vision_fov / 2:
                    # Add noise and uncertainty
                    noisy_distance = distance + np.random.normal(0, self.position_uncertainty)
                    noisy_velocity = np.array(other_vehicle['velocity'][:2]) + \
                                   np.random.normal(0, self.velocity_uncertainty, 2)
                    
                    if random.random() < self.detection_probability:
                        perception_data['vehicles'].append({
                            'id': other_id,
                            'position': other_pos + np.random.normal(0, self.position_uncertainty, 2),
                            'velocity': noisy_velocity,
                            'distance': noisy_distance,
                            'relative_angle': angle_to_vehicle,
                            'confidence': min(1.0, effective_range / distance)
                        })
        
        # Detect road boundaries and lanes
        perception_data['road_info'] = self._detect_road_features(
            vehicle_position, vehicle_orientation, effective_range
        )
        
        # Detect traffic signs and lights
        perception_data['traffic_info'] = self._detect_traffic_features(
            vehicle_position, vehicle_orientation, effective_range, world_state
        )
        
        return perception_data
    
    def _calculate_visibility_factor(self, weather: Dict) -> float:
        """Calculate visibility reduction due to weather"""
        base_visibility = 1.0
        
        # Rain effect
        rain_intensity = weather.get('rain_intensity', 0.0)
        base_visibility *= (1.0 - rain_intensity * 0.5)
        
        # Fog effect
        fog_density = weather.get('fog_density', 0.0)
        base_visibility *= (1.0 - fog_density * 0.8)
        
        # Snow effect
        snow_intensity = weather.get('snow_intensity', 0.0)
        base_visibility *= (1.0 - snow_intensity * 0.6)
        
        # Time of day effect
        time_of_day = weather.get('time_of_day', 12.0)
        if time_of_day < 6 or time_of_day > 20:  # Night time
            base_visibility *= 0.7
        elif time_of_day < 8 or time_of_day > 18:  # Dawn/dusk
            base_visibility *= 0.85
        
        return max(0.1, base_visibility)
    
    def _detect_road_features(self, position: np.ndarray, orientation: float, 
                            range_limit: float) -> Dict:
        """Detect road boundaries, lanes, and surface conditions"""
        
        # Simplified road detection
        road_info = {
            'lane_width': 3.5,
            'lane_markings': [],
            'road_curvature': 0.0,
            'surface_condition': 'dry',
            'speed_limit': 50.0  # km/h
        }
        
        # Simulate lane detection
        for i in range(-2, 3):  # 5 lanes
            lane_center_y = i * road_info['lane_width']
            if abs(lane_center_y - position[1]) < range_limit:
                road_info['lane_markings'].append({
                    'type': 'solid' if abs(i) == 2 else 'dashed',
                    'position': lane_center_y,
                    'confidence': 0.9
                })
        
        return road_info
    
    def _detect_traffic_features(self, position: np.ndarray, orientation: float,
                               range_limit: float, world_state: Dict) -> Dict:
        """Detect traffic signs, lights, and other traffic features"""
        
        traffic_info = {
            'traffic_lights': [],
            'stop_signs': [],
            'speed_limit_signs': [],
            'yield_signs': []
        }
        
        # Simulate traffic light detection
        if random.random() < 0.1:  # 10% chance of traffic light ahead
            traffic_info['traffic_lights'].append({
                'distance': random.uniform(50, 150),
                'state': random.choice(['red', 'yellow', 'green']),
                'time_remaining': random.uniform(5, 30),
                'confidence': 0.95
            })
        
        return traffic_info


class DecisionMakingSystem:
    """Advanced decision making system using neural networks"""
    
    def __init__(self, driver_profile: DriverProfile):
        self.profile = driver_profile
        
        # Neural network for decision making
        # Input: [speed, distance_to_lead, relative_speed, lane_position, traffic_light_state, ...]
        # Output: [throttle, brake, steering, lane_change_intent]
        self.decision_network = NeuralNetwork(
            input_size=20,  # Comprehensive sensor input
            hidden_sizes=[64, 32, 16],
            output_size=4
        )
        
        # Behavior parameters based on profile
        self._initialize_behavior_parameters()
        
        # Learning and adaptation
        self.experience_buffer = []
        self.learning_rate = 0.001
        
    def _initialize_behavior_parameters(self):
        """Initialize behavior parameters based on driver profile"""
        
        behavior_configs = {
            DrivingBehavior.AGGRESSIVE: {
                'target_speed_factor': 1.2,
                'following_distance_factor': 0.7,
                'lane_change_threshold': 0.3,
                'risk_acceptance': 0.8
            },
            DrivingBehavior.NORMAL: {
                'target_speed_factor': 1.0,
                'following_distance_factor': 1.0,
                'lane_change_threshold': 0.5,
                'risk_acceptance': 0.4
            },
            DrivingBehavior.CAUTIOUS: {
                'target_speed_factor': 0.9,
                'following_distance_factor': 1.3,
                'lane_change_threshold': 0.7,
                'risk_acceptance': 0.2
            },
            DrivingBehavior.ELDERLY: {
                'target_speed_factor': 0.8,
                'following_distance_factor': 1.5,
                'lane_change_threshold': 0.8,
                'risk_acceptance': 0.1
            }
        }
        
        config = behavior_configs.get(self.profile.behavior_type, 
                                    behavior_configs[DrivingBehavior.NORMAL])
        
        for key, value in config.items():
            setattr(self, key, value)
    
    def make_decision(self, perception_data: Dict, vehicle_state: Dict,
                     route_info: Dict) -> Dict:
        """Make driving decisions based on perception and state"""
        
        # Prepare neural network input
        nn_input = self._prepare_network_input(perception_data, vehicle_state, route_info)
        
        # Get neural network output
        nn_output = self.decision_network.forward(nn_input)
        
        # Post-process network output with behavior modifications
        decisions = self._post_process_decisions(nn_output, perception_data, vehicle_state)
        
        # Apply driver profile modifications
        decisions = self._apply_driver_profile(decisions, perception_data)
        
        # Safety checks and constraints
        decisions = self._apply_safety_constraints(decisions, perception_data, vehicle_state)
        
        return decisions
    
    def _prepare_network_input(self, perception: Dict, state: Dict, route: Dict) -> np.ndarray:
        """Prepare input vector for neural network"""
        
        inputs = np.zeros(20)
        
        # Vehicle state
        inputs[0] = np.linalg.norm(state['velocity'][:2]) / 30.0  # Normalized speed
        inputs[1] = state.get('fuel_level', 1.0)
        inputs[2] = state.get('engine_temperature', 90) / 120.0
        
        # Leading vehicle information
        if perception['vehicles']:
            closest_vehicle = min(perception['vehicles'], key=lambda v: v['distance'])
            inputs[3] = closest_vehicle['distance'] / 100.0  # Normalized distance
            inputs[4] = np.linalg.norm(closest_vehicle['velocity']) / 30.0
            inputs[5] = (closest_vehicle['distance'] - 
                        np.linalg.norm(closest_vehicle['velocity']) * 2) / 50.0  # Time gap
        
        # Road information
        road_info = perception.get('road_info', {})
        inputs[6] = road_info.get('lane_width', 3.5) / 5.0
        inputs[7] = road_info.get('road_curvature', 0.0)
        inputs[8] = road_info.get('speed_limit', 50) / 100.0
        
        # Traffic information
        traffic_info = perception.get('traffic_info', {})
        if traffic_info.get('traffic_lights'):
            light = traffic_info['traffic_lights'][0]
            inputs[9] = light['distance'] / 200.0
            inputs[10] = {'red': 0.0, 'yellow': 0.5, 'green': 1.0}.get(light['state'], 0.5)
            inputs[11] = light['time_remaining'] / 60.0
        
        # Route information
        inputs[12] = route.get('distance_to_destination', 1000) / 10000.0
        inputs[13] = route.get('time_to_destination', 600) / 3600.0
        
        # Environmental factors
        inputs[14] = perception.get('visibility_factor', 1.0)
        inputs[15] = len(perception['vehicles']) / 10.0  # Traffic density
        
        # Driver state
        inputs[16] = self.profile.fatigue_level
        inputs[17] = self.profile.attention_level
        inputs[18] = self.profile.risk_tolerance
        inputs[19] = random.random()  # Random factor for variability
        
        return inputs
    
    def _post_process_decisions(self, nn_output: np.ndarray, perception: Dict, 
                              state: Dict) -> Dict:
        """Post-process neural network output into driving decisions"""
        
        # Convert network output to control commands
        decisions = {
            'throttle': max(0.0, nn_output[0]),  # 0-1
            'brake': max(0.0, -nn_output[0]) if nn_output[0] < 0 else 0.0,  # 0-1
            'steering': np.clip(nn_output[1], -1.0, 1.0),  # -1 to 1
            'lane_change_intent': nn_output[2],  # -1 (left) to 1 (right)
            'gear_change': int(np.clip(nn_output[3] * 6 + 3, 1, 6))  # 1-6
        }
        
        return decisions
    
    def _apply_driver_profile(self, decisions: Dict, perception: Dict) -> Dict:
        """Apply driver profile characteristics to decisions"""
        
        # Speed preference
        decisions['throttle'] *= self.profile.speed_preference
        
        # Reaction time delay (simplified)
        if hasattr(self, 'previous_decisions'):
            blend_factor = 1.0 - self.profile.reaction_time / 2.0
            for key in ['throttle', 'brake', 'steering']:
                decisions[key] = (blend_factor * decisions[key] + 
                               (1 - blend_factor) * self.previous_decisions.get(key, 0))
        
        # Skill level effects
        decisions['steering'] *= self.profile.steering_skill
        decisions['brake'] *= self.profile.braking_skill
        decisions['throttle'] *= self.profile.acceleration_skill
        
        # Fatigue effects
        fatigue_factor = 1.0 - self.profile.fatigue_level * 0.3
        decisions['steering'] *= fatigue_factor
        
        # Attention effects
        if self.profile.attention_level < 0.7:
            # Reduced attention leads to delayed reactions
            for key in ['throttle', 'brake']:
                decisions[key] *= 0.8
        
        self.previous_decisions = decisions.copy()
        return decisions
    
    def _apply_safety_constraints(self, decisions: Dict, perception: Dict, 
                                state: Dict) -> Dict:
        """Apply safety constraints to prevent dangerous maneuvers"""
        
        # Emergency braking for imminent collision
        if perception['vehicles']:
            closest = min(perception['vehicles'], key=lambda v: v['distance'])
            if closest['distance'] < 10.0 and closest['velocity'][0] < state['velocity'][0]:
                decisions['brake'] = 1.0
                decisions['throttle'] = 0.0
        
        # Speed limit enforcement
        current_speed = np.linalg.norm(state['velocity'][:2]) * 3.6  # km/h
        speed_limit = perception.get('road_info', {}).get('speed_limit', 50)
        
        if current_speed > speed_limit * 1.1:  # 10% tolerance
            decisions['throttle'] *= 0.5
            decisions['brake'] = max(decisions['brake'], 0.3)
        
        # Traffic light compliance
        traffic_lights = perception.get('traffic_info', {}).get('traffic_lights', [])
        for light in traffic_lights:
            if light['state'] == 'red' and light['distance'] < 50:
                decisions['brake'] = max(decisions['brake'], 0.8)
                decisions['throttle'] = 0.0
        
        return decisions


class AdvancedAISystem:
    """Advanced AI system managing multiple AI-controlled vehicles"""
    
    def __init__(self):
        self.ai_vehicles: Dict[str, Dict] = {}
        self.world_state = {}
        self.weather_conditions = {}
        
        # AI system parameters
        self.update_frequency = 30  # Hz
        self.learning_enabled = True
        
        # Threading
        self.running = False
        self.ai_thread = None
        self.lock = threading.Lock()
        
    def add_ai_vehicle(self, vehicle_id: str, driver_profile: DriverProfile = None):
        """Add an AI-controlled vehicle"""
        
        if driver_profile is None:
            # Generate random driver profile
            behavior = random.choice(list(DrivingBehavior))
            driver_profile = DriverProfile(
                behavior_type=behavior,
                reaction_time=random.uniform(0.5, 2.0),
                risk_tolerance=random.uniform(0.1, 0.9),
                speed_preference=random.uniform(0.8, 1.3),
                following_distance=random.uniform(1.5, 3.0),
                lane_change_frequency=random.uniform(0.5, 2.0),
                attention_level=random.uniform(0.7, 1.0),
                fatigue_level=random.uniform(0.0, 0.3),
                experience_years=random.randint(1, 40)
            )
        
        with self.lock:
            self.ai_vehicles[vehicle_id] = {
                'profile': driver_profile,
                'perception': PerceptionSystem(vehicle_id),
                'decision_maker': DecisionMakingSystem(driver_profile),
                'route': {},
                'last_update': time.time()
            }
    
    def remove_ai_vehicle(self, vehicle_id: str):
        """Remove an AI-controlled vehicle"""
        
        with self.lock:
            if vehicle_id in self.ai_vehicles:
                del self.ai_vehicles[vehicle_id]
    
    def update_world_state(self, world_state: Dict):
        """Update world state information"""
        
        with self.lock:
            self.world_state = world_state.copy()
    
    def update_weather_conditions(self, weather: Dict):
        """Update weather conditions"""
        
        with self.lock:
            self.weather_conditions = weather.copy()
    
    def get_vehicle_decisions(self, vehicle_id: str) -> Optional[Dict]:
        """Get AI decisions for a specific vehicle"""
        
        with self.lock:
            if vehicle_id not in self.ai_vehicles:
                return None
            
            ai_vehicle = self.ai_vehicles[vehicle_id]
            vehicle_state = self.world_state.get('vehicles', {}).get(vehicle_id, {})
            
            if not vehicle_state:
                return None
            
            # Update perception
            vehicle_position = np.array(vehicle_state['position'])
            vehicle_orientation = vehicle_state.get('orientation', [0, 0, 0])[2]
            
            perception_data = ai_vehicle['perception'].update_perception(
                self.world_state, vehicle_position, vehicle_orientation, 
                self.weather_conditions
            )
            
            # Make decisions
            decisions = ai_vehicle['decision_maker'].make_decision(
                perception_data, vehicle_state, ai_vehicle['route']
            )
            
            ai_vehicle['last_update'] = time.time()
            return decisions
    
    def start_ai_system(self):
        """Start AI system processing"""
        
        if not self.running:
            self.running = True
            self.ai_thread = threading.Thread(target=self._ai_loop)
            self.ai_thread.daemon = True
            self.ai_thread.start()
    
    def stop_ai_system(self):
        """Stop AI system processing"""
        
        self.running = False
        if self.ai_thread:
            self.ai_thread.join()
    
    def _ai_loop(self):
        """Main AI processing loop"""
        
        dt = 1.0 / self.update_frequency
        
        while self.running:
            start_time = time.time()
            
            # Process all AI vehicles
            with self.lock:
                for vehicle_id in list(self.ai_vehicles.keys()):
                    try:
                        decisions = self.get_vehicle_decisions(vehicle_id)
                        if decisions:
                            # Store decisions for retrieval by physics engine
                            self.ai_vehicles[vehicle_id]['current_decisions'] = decisions
                    except Exception as e:
                        print(f"Error processing AI for vehicle {vehicle_id}: {e}")
            
            # Maintain update frequency
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    def get_ai_statistics(self) -> Dict:
        """Get AI system statistics"""
        
        with self.lock:
            stats = {
                'total_ai_vehicles': len(self.ai_vehicles),
                'behavior_distribution': {},
                'average_reaction_time': 0.0,
                'learning_enabled': self.learning_enabled,
                'update_frequency': self.update_frequency
            }
            
            if self.ai_vehicles:
                # Calculate behavior distribution
                behaviors = [ai['profile'].behavior_type.value for ai in self.ai_vehicles.values()]
                for behavior in behaviors:
                    stats['behavior_distribution'][behavior] = behaviors.count(behavior)
                
                # Calculate average reaction time
                reaction_times = [ai['profile'].reaction_time for ai in self.ai_vehicles.values()]
                stats['average_reaction_time'] = sum(reaction_times) / len(reaction_times)
            
            return stats