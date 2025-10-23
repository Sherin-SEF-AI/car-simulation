"""
End-to-End Scenario Testing for Robotic Car Simulation

Tests complete driving scenarios from start to finish, validating that all
system components work together to achieve realistic autonomous driving behaviors.
"""

import unittest
import sys
import os
import time
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from unittest.mock import Mock, patch
from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.application import SimulationApplication
from core.challenge_manager import ChallengeManager
from core.environment import Environment


@dataclass
class ScenarioResult:
    """Result of an end-to-end scenario test"""
    scenario_name: str
    success: bool
    completion_time: float
    safety_score: float
    efficiency_score: float
    rule_compliance_score: float
    error_messages: List[str]
    telemetry_data: Dict[str, Any]


class EndToEndTestBase(unittest.TestCase):
    """Base class for end-to-end scenario tests"""
    
    def setUp(self):
        """Set up test environment"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
        self.simulation_app = SimulationApplication()
        self.challenge_manager = ChallengeManager()
        self.scenario_results: List[ScenarioResult] = []
        
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'simulation_app'):
            self.simulation_app.cleanup()
            
    def run_scenario(self, scenario_config: Dict[str, Any], 
                    timeout: float = 60.0) -> ScenarioResult:
        """Run a complete end-to-end scenario"""
        scenario_name = scenario_config['name']
        print(f"Running scenario: {scenario_name}")
        
        start_time = time.time()
        error_messages = []
        telemetry_data = {}
        
        try:
            # Set up environment
            if 'environment' in scenario_config:
                self.setup_environment(scenario_config['environment'])
                
            # Spawn vehicles
            vehicle_ids = []
            if 'vehicles' in scenario_config:
                vehicle_ids = self.spawn_vehicles(scenario_config['vehicles'])
                
            # Configure AI behaviors
            if 'ai_behaviors' in scenario_config:
                self.configure_ai_behaviors(vehicle_ids, scenario_config['ai_behaviors'])
                
            # Start recording
            self.simulation_app.recording_system.start_recording({
                'sample_rate': 30,
                'capture_vehicle_states': True,
                'capture_sensor_data': True
            })
            
            # Start simulation
            self.simulation_app.start_simulation()
            
            # Run scenario until completion or timeout
            success, completion_data = self.monitor_scenario_execution(
                scenario_config, vehicle_ids, timeout)
                
            completion_time = time.time() - start_time
            
            # Stop simulation and recording
            self.simulation_app.pause_simulation()
            recording_metadata = self.simulation_app.recording_system.stop_recording()
            
            # Analyze results
            telemetry_data = self.collect_telemetry_data(vehicle_ids)
            safety_score = self.calculate_safety_score(telemetry_data)
            efficiency_score = self.calculate_efficiency_score(telemetry_data, completion_data)
            rule_compliance_score = self.calculate_rule_compliance_score(telemetry_data)
            
        except Exception as e:
            success = False
            completion_time = time.time() - start_time
            error_messages.append(str(e))
            safety_score = 0.0
            efficiency_score = 0.0
            rule_compliance_score = 0.0
            
        finally:
            # Clean up vehicles
            for vehicle_id in vehicle_ids:
                try:
                    self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                except:
                    pass
                    
        result = ScenarioResult(
            scenario_name=scenario_name,
            success=success,
            completion_time=completion_time,
            safety_score=safety_score,
            efficiency_score=efficiency_score,
            rule_compliance_score=rule_compliance_score,
            error_messages=error_messages,
            telemetry_data=telemetry_data
        )
        
        self.scenario_results.append(result)
        return result       
 
    def setup_environment(self, env_config: Dict[str, Any]):
        """Set up the simulation environment"""
        environment = self.simulation_app.environment
        
        if 'weather' in env_config:
            weather = env_config['weather']
            environment.set_weather(weather['type'], weather.get('intensity', 0.5))
            
        if 'time_of_day' in env_config:
            environment.set_time_of_day(env_config['time_of_day'])
            
        if 'traffic_lights' in env_config:
            for light_config in env_config['traffic_lights']:
                environment.add_traffic_light(
                    position=light_config['position'],
                    state=light_config['state']
                )
                
        if 'obstacles' in env_config:
            for obstacle_config in env_config['obstacles']:
                environment.add_obstacle(
                    position=obstacle_config['position'],
                    size=obstacle_config['size']
                )
                
    def spawn_vehicles(self, vehicle_configs: List[Dict[str, Any]]) -> List[str]:
        """Spawn vehicles according to configuration"""
        vehicle_ids = []
        
        for i, vehicle_config in enumerate(vehicle_configs):
            vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                vehicle_type=vehicle_config.get('type', 'default_car'),
                position=vehicle_config['position']
            )
            vehicle_ids.append(vehicle_id)
            
        return vehicle_ids
        
    def configure_ai_behaviors(self, vehicle_ids: List[str], 
                             behavior_configs: List[Dict[str, Any]]):
        """Configure AI behaviors for vehicles"""
        ai_system = self.simulation_app.ai_system
        
        for i, (vehicle_id, behavior_config) in enumerate(zip(vehicle_ids, behavior_configs)):
            behavior_tree = ai_system.create_behavior_tree(behavior_config)
            ai_system.set_vehicle_behavior(vehicle_id, behavior_tree)
            
    def monitor_scenario_execution(self, scenario_config: Dict[str, Any], 
                                 vehicle_ids: List[str], timeout: float) -> Tuple[bool, Dict]:
        """Monitor scenario execution and check for completion"""
        success_conditions = scenario_config.get('success_conditions', {})
        start_time = time.time()
        completion_data = {}
        
        while time.time() - start_time < timeout:
            QTest.qWait(100)  # Check every 100ms
            
            # Check success conditions
            if self.check_success_conditions(success_conditions, vehicle_ids):
                completion_data['completion_reason'] = 'success_conditions_met'
                return True, completion_data
                
            # Check failure conditions
            if self.check_failure_conditions(scenario_config.get('failure_conditions', {}), vehicle_ids):
                completion_data['completion_reason'] = 'failure_conditions_met'
                return False, completion_data
                
        # Timeout reached
        completion_data['completion_reason'] = 'timeout'
        return False, completion_data
        
    def check_success_conditions(self, conditions: Dict[str, Any], 
                               vehicle_ids: List[str]) -> bool:
        """Check if success conditions are met"""
        if 'destination_reached' in conditions:
            target_position = conditions['destination_reached']['position']
            tolerance = conditions['destination_reached'].get('tolerance', 2.0)
            
            for vehicle_id in vehicle_ids:
                vehicle_pos = self.simulation_app.physics_engine.get_vehicle_position(vehicle_id)
                distance = self.calculate_distance(vehicle_pos, target_position)
                if distance <= tolerance:
                    return True
                    
        if 'time_limit' in conditions:
            # Success if we reach this point without failure
            return True
            
        return False
        
    def check_failure_conditions(self, conditions: Dict[str, Any], 
                               vehicle_ids: List[str]) -> bool:
        """Check if failure conditions are met"""
        if 'collision_detected' in conditions and conditions['collision_detected']:
            collision_count = self.simulation_app.physics_engine.get_collision_count()
            if collision_count > 0:
                return True
                
        if 'off_road' in conditions and conditions['off_road']:
            for vehicle_id in vehicle_ids:
                if self.simulation_app.environment.is_vehicle_off_road(vehicle_id):
                    return True
                    
        return False
        
    def collect_telemetry_data(self, vehicle_ids: List[str]) -> Dict[str, Any]:
        """Collect telemetry data from all vehicles"""
        telemetry = {}
        
        for vehicle_id in vehicle_ids:
            vehicle_data = self.simulation_app.vehicle_manager.get_vehicle_telemetry(vehicle_id)
            telemetry[vehicle_id] = vehicle_data
            
        return telemetry
        
    def calculate_safety_score(self, telemetry_data: Dict[str, Any]) -> float:
        """Calculate safety score based on telemetry data"""
        safety_score = 100.0
        
        for vehicle_id, data in telemetry_data.items():
            # Penalize for collisions
            if data.get('collision_count', 0) > 0:
                safety_score -= 50.0
                
            # Penalize for excessive speed
            max_speed = max(data.get('speed_history', [0]))
            if max_speed > 30:  # Speed limit
                safety_score -= min(20.0, (max_speed - 30) * 2)
                
            # Penalize for hard braking
            hard_braking_events = len([b for b in data.get('brake_history', []) if b > 0.8])
            safety_score -= hard_braking_events * 5.0
            
        return max(0.0, safety_score)
        
    def calculate_efficiency_score(self, telemetry_data: Dict[str, Any], 
                                 completion_data: Dict[str, Any]) -> float:
        """Calculate efficiency score based on performance"""
        efficiency_score = 100.0
        
        # Penalize for long completion times
        if 'completion_time' in completion_data:
            expected_time = 30.0  # Expected completion time in seconds
            actual_time = completion_data['completion_time']
            if actual_time > expected_time:
                efficiency_score -= min(30.0, (actual_time - expected_time) * 2)
                
        # Reward smooth driving
        for vehicle_id, data in telemetry_data.items():
            acceleration_variance = self.calculate_variance(data.get('acceleration_history', []))
            if acceleration_variance < 1.0:  # Smooth acceleration
                efficiency_score += 10.0
                
        return max(0.0, min(100.0, efficiency_score))
        
    def calculate_rule_compliance_score(self, telemetry_data: Dict[str, Any]) -> float:
        """Calculate rule compliance score"""
        compliance_score = 100.0
        
        for vehicle_id, data in telemetry_data.items():
            # Check speed limit compliance
            speed_violations = len([s for s in data.get('speed_history', []) if s > 25])
            compliance_score -= speed_violations * 2.0
            
            # Check traffic light compliance
            if data.get('ran_red_light', False):
                compliance_score -= 30.0
                
            # Check lane keeping
            lane_departures = data.get('lane_departure_count', 0)
            compliance_score -= lane_departures * 10.0
            
        return max(0.0, compliance_score)
        
    def calculate_distance(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5
        
    def calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean)**2 for x in values) / len(values)


class TestBasicDrivingScenarios(EndToEndTestBase):
    """Test basic autonomous driving scenarios"""
    
    def test_straight_line_driving(self):
        """Test vehicle driving in a straight line"""
        scenario_config = {
            'name': 'straight_line_driving',
            'environment': {
                'weather': {'type': 'clear'},
                'time_of_day': 12.0
            },
            'vehicles': [
                {'type': 'default_car', 'position': (0, 0, 0)}
            ],
            'ai_behaviors': [
                {
                    'type': 'sequence',
                    'children': [
                        {'type': 'action', 'action': 'accelerate', 'parameters': {'throttle': 0.5}},
                        {'type': 'action', 'action': 'maintain_speed', 'parameters': {'target_speed': 20}}
                    ]
                }
            ],
            'success_conditions': {
                'destination_reached': {
                    'position': (100, 0, 0),
                    'tolerance': 5.0
                }
            },
            'failure_conditions': {
                'collision_detected': True,
                'off_road': True
            }
        }
        
        result = self.run_scenario(scenario_config, timeout=30.0)
        
        self.assertTrue(result.success, f"Straight line driving failed: {result.error_messages}")
        self.assertGreater(result.safety_score, 80.0)
        self.assertGreater(result.efficiency_score, 70.0)
        
    def test_lane_following(self):
        """Test vehicle following lane markings"""
        scenario_config = {
            'name': 'lane_following',
            'environment': {
                'weather': {'type': 'clear'},
                'time_of_day': 14.0
            },
            'vehicles': [
                {'type': 'default_car', 'position': (0, -1.5, 0)}  # Slightly off-center
            ],
            'ai_behaviors': [
                {
                    'type': 'sequence',
                    'children': [
                        {'type': 'action', 'action': 'follow_lane'},
                        {'type': 'action', 'action': 'maintain_speed', 'parameters': {'target_speed': 25}}
                    ]
                }
            ],
            'success_conditions': {
                'destination_reached': {
                    'position': (200, 0, 0),
                    'tolerance': 3.0
                }
            },
            'failure_conditions': {
                'collision_detected': True,
                'off_road': True
            }
        }
        
        result = self.run_scenario(scenario_config, timeout=45.0)
        
        self.assertTrue(result.success, f"Lane following failed: {result.error_messages}")
        self.assertGreater(result.rule_compliance_score, 85.0)
        
    def test_curve_navigation(self):
        """Test vehicle navigating through curves"""
        scenario_config = {
            'name': 'curve_navigation',
            'environment': {
                'weather': {'type': 'clear'},
                'time_of_day': 10.0
            },
            'vehicles': [
                {'type': 'default_car', 'position': (0, 0, 0)}
            ],
            'ai_behaviors': [
                {
                    'type': 'sequence',
                    'children': [
                        {'type': 'action', 'action': 'follow_path', 
                         'parameters': {'waypoints': [(0, 0, 0), (50, 25, 0), (100, 50, 0), (150, 25, 0), (200, 0, 0)]}},
                        {'type': 'action', 'action': 'adaptive_speed'}
                    ]
                }
            ],
            'success_conditions': {
                'destination_reached': {
                    'position': (200, 0, 0),
                    'tolerance': 5.0
                }
            },
            'failure_conditions': {
                'collision_detected': True,
                'off_road': True
            }
        }
        
        result = self.run_scenario(scenario_config, timeout=60.0)
        
        self.assertTrue(result.success, f"Curve navigation failed: {result.error_messages}")
        self.assertGreater(result.safety_score, 75.0)


class TestTrafficScenarios(EndToEndTestBase):
    """Test scenarios involving traffic interactions"""
    
    def test_traffic_light_compliance(self):
        """Test vehicle stopping at red traffic lights"""
        scenario_config = {
            'name': 'traffic_light_compliance',
            'environment': {
                'weather': {'type': 'clear'},
                'time_of_day': 12.0,
                'traffic_lights': [
                    {'position': (50, 0, 0), 'state': 'red'}
                ]
            },
            'vehicles': [
                {'type': 'default_car', 'position': (0, 0, 0)}
            ],
            'ai_behaviors': [
                {
                    'type': 'sequence',
                    'children': [
                        {'type': 'condition', 'condition': 'traffic_light_ahead'},
                        {'type': 'selector', 'children': [
                            {'type': 'action', 'action': 'stop_at_red_light'},
                            {'type': 'action', 'action': 'proceed_on_green'}
                        ]}
                    ]
                }
            ],
            'success_conditions': {
                'time_limit': 30.0  # Success if no violations in 30 seconds
            },
            'failure_conditions': {
                'collision_detected': True
            }
        }
        
        result = self.run_scenario(scenario_config, timeout=35.0)
        
        self.assertTrue(result.success, f"Traffic light compliance failed: {result.error_messages}")
        self.assertGreater(result.rule_compliance_score, 90.0)
        
    def test_multi_vehicle_interaction(self):
        """Test interactions between multiple vehicles"""
        scenario_config = {
            'name': 'multi_vehicle_interaction',
            'environment': {
                'weather': {'type': 'clear'},
                'time_of_day': 15.0
            },
            'vehicles': [
                {'type': 'default_car', 'position': (0, 0, 0)},
                {'type': 'default_car', 'position': (10, 0, 0)},
                {'type': 'default_car', 'position': (20, 0, 0)}
            ],
            'ai_behaviors': [
                {
                    'type': 'sequence',
                    'children': [
                        {'type': 'action', 'action': 'follow_leader'},
                        {'type': 'action', 'action': 'maintain_safe_distance'}
                    ]
                },
                {
                    'type': 'sequence',
                    'children': [
                        {'type': 'action', 'action': 'cruise_control', 'parameters': {'speed': 20}},
                        {'type': 'action', 'action': 'avoid_collisions'}
                    ]
                },
                {
                    'type': 'sequence',
                    'children': [
                        {'type': 'action', 'action': 'follow_leader'},
                        {'type': 'action', 'action': 'maintain_safe_distance'}
                    ]
                }
            ],
            'success_conditions': {
                'destination_reached': {
                    'position': (150, 0, 0),
                    'tolerance': 10.0
                }
            },
            'failure_conditions': {
                'collision_detected': True
            }
        }
        
        result = self.run_scenario(scenario_config, timeout=60.0)
        
        self.assertTrue(result.success, f"Multi-vehicle interaction failed: {result.error_messages}")
        self.assertGreater(result.safety_score, 80.0)


class TestWeatherScenarios(EndToEndTestBase):
    """Test scenarios in different weather conditions"""
    
    def test_rain_driving(self):
        """Test vehicle driving in rain conditions"""
        scenario_config = {
            'name': 'rain_driving',
            'environment': {
                'weather': {'type': 'rain', 'intensity': 0.7},
                'time_of_day': 16.0
            },
            'vehicles': [
                {'type': 'default_car', 'position': (0, 0, 0)}
            ],
            'ai_behaviors': [
                {
                    'type': 'sequence',
                    'children': [
                        {'type': 'condition', 'condition': 'weather_check'},
                        {'type': 'action', 'action': 'adjust_speed_for_weather'},
                        {'type': 'action', 'action': 'increase_following_distance'}
                    ]
                }
            ],
            'success_conditions': {
                'destination_reached': {
                    'position': (100, 0, 0),
                    'tolerance': 5.0
                }
            },
            'failure_conditions': {
                'collision_detected': True,
                'off_road': True
            }
        }
        
        result = self.run_scenario(scenario_config, timeout=45.0)
        
        self.assertTrue(result.success, f"Rain driving failed: {result.error_messages}")
        self.assertGreater(result.safety_score, 70.0)  # Lower threshold due to weather
        
    def test_night_driving(self):
        """Test vehicle driving at night"""
        scenario_config = {
            'name': 'night_driving',
            'environment': {
                'weather': {'type': 'clear'},
                'time_of_day': 22.0  # 10 PM
            },
            'vehicles': [
                {'type': 'default_car', 'position': (0, 0, 0)}
            ],
            'ai_behaviors': [
                {
                    'type': 'sequence',
                    'children': [
                        {'type': 'action', 'action': 'enable_headlights'},
                        {'type': 'action', 'action': 'reduce_speed_for_visibility'},
                        {'type': 'action', 'action': 'follow_lane'}
                    ]
                }
            ],
            'success_conditions': {
                'destination_reached': {
                    'position': (80, 0, 0),
                    'tolerance': 5.0
                }
            },
            'failure_conditions': {
                'collision_detected': True,
                'off_road': True
            }
        }
        
        result = self.run_scenario(scenario_config, timeout=50.0)
        
        self.assertTrue(result.success, f"Night driving failed: {result.error_messages}")
        self.assertGreater(result.safety_score, 75.0)


class TestEmergencyScenarios(EndToEndTestBase):
    """Test emergency and edge case scenarios"""
    
    def test_emergency_braking(self):
        """Test emergency braking when obstacle appears"""
        scenario_config = {
            'name': 'emergency_braking',
            'environment': {
                'weather': {'type': 'clear'},
                'time_of_day': 12.0,
                'obstacles': [
                    {'position': (30, 0, 0), 'size': (2, 2, 2)}
                ]
            },
            'vehicles': [
                {'type': 'default_car', 'position': (0, 0, 0)}
            ],
            'ai_behaviors': [
                {
                    'type': 'sequence',
                    'children': [
                        {'type': 'action', 'action': 'accelerate', 'parameters': {'throttle': 0.8}},
                        {'type': 'condition', 'condition': 'obstacle_detected'},
                        {'type': 'action', 'action': 'emergency_brake'}
                    ]
                }
            ],
            'success_conditions': {
                'time_limit': 20.0  # Success if no collision in 20 seconds
            },
            'failure_conditions': {
                'collision_detected': True
            }
        }
        
        result = self.run_scenario(scenario_config, timeout=25.0)
        
        self.assertTrue(result.success, f"Emergency braking failed: {result.error_messages}")
        self.assertGreater(result.safety_score, 85.0)
        
    def test_obstacle_avoidance(self):
        """Test avoiding obstacles by steering around them"""
        scenario_config = {
            'name': 'obstacle_avoidance',
            'environment': {
                'weather': {'type': 'clear'},
                'time_of_day': 14.0,
                'obstacles': [
                    {'position': (40, 0, 0), 'size': (3, 3, 2)},
                    {'position': (80, -2, 0), 'size': (2, 2, 2)}
                ]
            },
            'vehicles': [
                {'type': 'default_car', 'position': (0, 0, 0)}
            ],
            'ai_behaviors': [
                {
                    'type': 'sequence',
                    'children': [
                        {'type': 'action', 'action': 'path_planning'},
                        {'type': 'action', 'action': 'obstacle_avoidance'},
                        {'type': 'action', 'action': 'return_to_lane'}
                    ]
                }
            ],
            'success_conditions': {
                'destination_reached': {
                    'position': (120, 0, 0),
                    'tolerance': 5.0
                }
            },
            'failure_conditions': {
                'collision_detected': True,
                'off_road': True
            }
        }
        
        result = self.run_scenario(scenario_config, timeout=60.0)
        
        self.assertTrue(result.success, f"Obstacle avoidance failed: {result.error_messages}")
        self.assertGreater(result.safety_score, 80.0)
        self.assertGreater(result.efficiency_score, 70.0)


if __name__ == '__main__':
    unittest.main()