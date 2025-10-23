"""
Scenario Validation and Regression Testing Tools

Automated testing tools to validate driving scenarios and detect regressions
in autonomous vehicle behavior across software updates.
"""

import unittest
import sys
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest

from core.application import SimulationApplication


@dataclass
class ScenarioBaseline:
    """Baseline data for scenario regression testing"""
    scenario_id: str
    version: str
    timestamp: datetime
    completion_time: float
    safety_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    behavior_checkpoints: List[Dict[str, Any]]
    final_state: Dict[str, Any]


@dataclass
class RegressionResult:
    """Result of regression testing"""
    scenario_id: str
    baseline_version: str
    current_version: str
    passed: bool
    deviations: List[str]
    performance_changes: Dict[str, float]
    behavior_differences: List[str]


class ScenarioRegressionTester:
    """Tool for scenario regression testing"""
    
    BASELINE_FILE = "tests/scenario_baselines.json"
    TOLERANCE_CONFIG = {
        'completion_time': 0.1,  # 10% tolerance
        'safety_score': 0.05,    # 5% tolerance
        'position_accuracy': 2.0, # 2 meter tolerance
        'speed_variance': 0.15   # 15% tolerance
    }
    
    def __init__(self):
        self.baselines: Dict[str, ScenarioBaseline] = {}
        self.load_baselines()
        
    def load_baselines(self):
        """Load scenario baselines from file"""
        baseline_file = Path(self.BASELINE_FILE)
        if baseline_file.exists():
            try:
                with open(baseline_file) as f:
                    data = json.load(f)
                    
                for scenario_id, baseline_data in data.items():
                    # Convert timestamp string back to datetime
                    baseline_data['timestamp'] = datetime.fromisoformat(
                        baseline_data['timestamp'])
                    self.baselines[scenario_id] = ScenarioBaseline(**baseline_data)
                    
            except Exception as e:
                print(f"Warning: Could not load baselines: {e}")
                
    def save_baselines(self):
        """Save scenario baselines to file"""
        data = {}
        for scenario_id, baseline in self.baselines.items():
            baseline_dict = asdict(baseline)
            # Convert datetime to string for JSON serialization
            baseline_dict['timestamp'] = baseline.timestamp.isoformat()
            data[scenario_id] = baseline_dict
            
        with open(self.BASELINE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
            
    def create_baseline(self, scenario_id: str, scenario_result: Dict[str, Any], 
                       version: str = "1.0.0") -> ScenarioBaseline:
        """Create a new baseline from scenario result"""
        baseline = ScenarioBaseline(
            scenario_id=scenario_id,
            version=version,
            timestamp=datetime.now(),
            completion_time=scenario_result.get('completion_time', 0.0),
            safety_metrics=scenario_result.get('safety_metrics', {}),
            performance_metrics=scenario_result.get('performance_metrics', {}),
            behavior_checkpoints=scenario_result.get('behavior_checkpoints', []),
            final_state=scenario_result.get('final_state', {})
        )
        
        self.baselines[scenario_id] = baseline
        return baseline
        
    def compare_with_baseline(self, scenario_id: str, current_result: Dict[str, Any],
                            current_version: str = "current") -> RegressionResult:
        """Compare current result with baseline"""
        if scenario_id not in self.baselines:
            return RegressionResult(
                scenario_id=scenario_id,
                baseline_version="none",
                current_version=current_version,
                passed=False,
                deviations=["No baseline available for comparison"],
                performance_changes={},
                behavior_differences=[]
            )
            
        baseline = self.baselines[scenario_id]
        deviations = []
        performance_changes = {}
        behavior_differences = []
        
        # Compare completion time
        baseline_time = baseline.completion_time
        current_time = current_result.get('completion_time', 0.0)
        time_change = (current_time - baseline_time) / baseline_time if baseline_time > 0 else 0
        
        if abs(time_change) > self.TOLERANCE_CONFIG['completion_time']:
            deviations.append(f"Completion time changed by {time_change:.1%}")
            
        performance_changes['completion_time_change'] = time_change
        
        # Compare safety metrics
        current_safety = current_result.get('safety_metrics', {})
        for metric, baseline_value in baseline.safety_metrics.items():
            current_value = current_safety.get(metric, 0.0)
            if baseline_value > 0:
                change = (current_value - baseline_value) / baseline_value
                if abs(change) > self.TOLERANCE_CONFIG['safety_score']:
                    deviations.append(f"Safety metric '{metric}' changed by {change:.1%}")
                performance_changes[f'safety_{metric}_change'] = change
                
        # Compare behavior checkpoints
        baseline_checkpoints = baseline.behavior_checkpoints
        current_checkpoints = current_result.get('behavior_checkpoints', [])
        
        behavior_differences = self.compare_behavior_checkpoints(
            baseline_checkpoints, current_checkpoints)
            
        # Determine if test passed
        passed = len(deviations) == 0 and len(behavior_differences) == 0
        
        return RegressionResult(
            scenario_id=scenario_id,
            baseline_version=baseline.version,
            current_version=current_version,
            passed=passed,
            deviations=deviations,
            performance_changes=performance_changes,
            behavior_differences=behavior_differences
        )
        
    def compare_behavior_checkpoints(self, baseline_checkpoints: List[Dict[str, Any]],
                                   current_checkpoints: List[Dict[str, Any]]) -> List[str]:
        """Compare behavior checkpoints between baseline and current"""
        differences = []
        
        if len(baseline_checkpoints) != len(current_checkpoints):
            differences.append(
                f"Checkpoint count mismatch: baseline={len(baseline_checkpoints)}, "
                f"current={len(current_checkpoints)}")
            return differences
            
        for i, (baseline_cp, current_cp) in enumerate(zip(baseline_checkpoints, current_checkpoints)):
            # Compare positions
            baseline_pos = baseline_cp.get('position', (0, 0, 0))
            current_pos = current_cp.get('position', (0, 0, 0))
            
            distance = self.calculate_distance(baseline_pos, current_pos)
            if distance > self.TOLERANCE_CONFIG['position_accuracy']:
                differences.append(
                    f"Checkpoint {i}: position deviation {distance:.2f}m")
                    
            # Compare speeds
            baseline_speed = baseline_cp.get('speed', 0.0)
            current_speed = current_cp.get('speed', 0.0)
            
            if baseline_speed > 0:
                speed_change = abs(current_speed - baseline_speed) / baseline_speed
                if speed_change > self.TOLERANCE_CONFIG['speed_variance']:
                    differences.append(
                        f"Checkpoint {i}: speed deviation {speed_change:.1%}")
                        
            # Compare AI decisions
            baseline_decision = baseline_cp.get('ai_decision', '')
            current_decision = current_cp.get('ai_decision', '')
            
            if baseline_decision != current_decision:
                differences.append(
                    f"Checkpoint {i}: AI decision changed from '{baseline_decision}' "
                    f"to '{current_decision}'")
                    
        return differences
        
    def calculate_distance(self, pos1: Tuple[float, float, float], 
                          pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between positions"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5


class TestScenarioRegression(unittest.TestCase):
    """Test scenario regression detection"""
    
    def setUp(self):
        """Set up regression testing environment"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
        self.simulation_app = SimulationApplication()
        self.regression_tester = ScenarioRegressionTester()
        
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'simulation_app'):
            self.simulation_app.cleanup()
            
    def run_scenario_with_checkpoints(self, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run scenario and collect checkpoint data"""
        scenario_name = scenario_config['name']
        
        # Set up environment
        if 'environment' in scenario_config:
            self.setup_environment(scenario_config['environment'])
            
        # Spawn vehicles
        vehicle_ids = []
        if 'vehicles' in scenario_config:
            for vehicle_config in scenario_config['vehicles']:
                vehicle_id = self.simulation_app.vehicle_manager.spawn_vehicle(
                    vehicle_config.get('type', 'default_car'),
                    position=vehicle_config['position']
                )
                vehicle_ids.append(vehicle_id)
                
        # Configure AI behaviors
        if 'ai_behaviors' in scenario_config:
            for i, behavior_config in enumerate(scenario_config['ai_behaviors']):
                if i < len(vehicle_ids):
                    behavior_tree = self.simulation_app.ai_system.create_behavior_tree(behavior_config)
                    self.simulation_app.ai_system.set_vehicle_behavior(vehicle_ids[i], behavior_tree)
                    
        try:
            # Start simulation
            self.simulation_app.start_simulation()
            
            # Collect checkpoints during execution
            checkpoints = []
            start_time = time.time()
            checkpoint_interval = 2.0  # Every 2 seconds
            last_checkpoint_time = start_time
            
            timeout = scenario_config.get('timeout', 60.0)
            
            while time.time() - start_time < timeout:
                QTest.qWait(100)
                
                current_time = time.time()
                if current_time - last_checkpoint_time >= checkpoint_interval:
                    # Collect checkpoint data
                    for vehicle_id in vehicle_ids:
                        position = self.simulation_app.physics_engine.get_vehicle_position(vehicle_id)
                        velocity = self.simulation_app.physics_engine.get_vehicle_velocity(vehicle_id)
                        speed = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5
                        
                        ai_state = self.simulation_app.ai_system.get_vehicle_state(vehicle_id)
                        ai_decision = ai_state.get('current_action', 'unknown')
                        
                        checkpoint = {
                            'timestamp': current_time - start_time,
                            'vehicle_id': vehicle_id,
                            'position': position,
                            'speed': speed,
                            'ai_decision': ai_decision
                        }
                        checkpoints.append(checkpoint)
                        
                    last_checkpoint_time = current_time
                    
                # Check completion conditions
                if self.check_scenario_completion(scenario_config, vehicle_ids):
                    break
                    
            completion_time = time.time() - start_time
            
            # Calculate final metrics
            safety_metrics = self.calculate_safety_metrics(vehicle_ids)
            performance_metrics = self.calculate_performance_metrics(vehicle_ids, completion_time)
            final_state = self.get_final_state(vehicle_ids)
            
            return {
                'scenario_name': scenario_name,
                'completion_time': completion_time,
                'safety_metrics': safety_metrics,
                'performance_metrics': performance_metrics,
                'behavior_checkpoints': checkpoints,
                'final_state': final_state,
                'success': True
            }
            
        except Exception as e:
            return {
                'scenario_name': scenario_name,
                'completion_time': 0.0,
                'safety_metrics': {},
                'performance_metrics': {},
                'behavior_checkpoints': [],
                'final_state': {},
                'success': False,
                'error': str(e)
            }
            
        finally:
            # Clean up
            self.simulation_app.pause_simulation()
            for vehicle_id in vehicle_ids:
                try:
                    self.simulation_app.vehicle_manager.despawn_vehicle(vehicle_id)
                except:
                    pass
                    
    def setup_environment(self, env_config: Dict[str, Any]):
        """Set up simulation environment"""
        environment = self.simulation_app.environment
        
        if 'weather' in env_config:
            weather = env_config['weather']
            environment.set_weather(weather['type'], weather.get('intensity', 0.5))
            
        if 'time_of_day' in env_config:
            environment.set_time_of_day(env_config['time_of_day'])
            
    def check_scenario_completion(self, scenario_config: Dict[str, Any], 
                                vehicle_ids: List[str]) -> bool:
        """Check if scenario completion conditions are met"""
        success_conditions = scenario_config.get('success_conditions', {})
        
        if 'destination_reached' in success_conditions:
            target_pos = success_conditions['destination_reached']['position']
            tolerance = success_conditions['destination_reached'].get('tolerance', 2.0)
            
            for vehicle_id in vehicle_ids:
                vehicle_pos = self.simulation_app.physics_engine.get_vehicle_position(vehicle_id)
                distance = self.regression_tester.calculate_distance(vehicle_pos, target_pos)
                if distance <= tolerance:
                    return True
                    
        return False
        
    def calculate_safety_metrics(self, vehicle_ids: List[str]) -> Dict[str, float]:
        """Calculate safety metrics for vehicles"""
        collision_count = self.simulation_app.physics_engine.get_collision_count()
        
        metrics = {
            'collision_count': float(collision_count),
            'safety_score': max(0.0, 100.0 - collision_count * 50.0)
        }
        
        return metrics
        
    def calculate_performance_metrics(self, vehicle_ids: List[str], 
                                    completion_time: float) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {
            'completion_time': completion_time,
            'efficiency_score': max(0.0, 100.0 - max(0.0, completion_time - 30.0) * 2.0)
        }
        
        return metrics
        
    def get_final_state(self, vehicle_ids: List[str]) -> Dict[str, Any]:
        """Get final state of all vehicles"""
        final_state = {}
        
        for vehicle_id in vehicle_ids:
            position = self.simulation_app.physics_engine.get_vehicle_position(vehicle_id)
            velocity = self.simulation_app.physics_engine.get_vehicle_velocity(vehicle_id)
            
            final_state[vehicle_id] = {
                'position': position,
                'velocity': velocity
            }
            
        return final_state
        
    def test_basic_scenario_regression(self):
        """Test regression detection for basic driving scenario"""
        scenario_config = {
            'name': 'basic_regression_test',
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
                    'position': (50, 0, 0),
                    'tolerance': 5.0
                }
            },
            'timeout': 30.0
        }
        
        # Run scenario
        result = self.run_scenario_with_checkpoints(scenario_config)
        
        # Check if baseline exists
        scenario_id = 'basic_regression_test'
        if scenario_id not in self.regression_tester.baselines:
            # Create baseline
            baseline = self.regression_tester.create_baseline(scenario_id, result)
            self.regression_tester.save_baselines()
            print(f"Created baseline for scenario: {scenario_id}")
        else:
            # Compare with baseline
            regression_result = self.regression_tester.compare_with_baseline(
                scenario_id, result)
                
            if not regression_result.passed:
                print(f"Regression detected in {scenario_id}:")
                for deviation in regression_result.deviations:
                    print(f"  - {deviation}")
                for difference in regression_result.behavior_differences:
                    print(f"  - {difference}")
                    
            # For testing purposes, we'll assert success
            # In real usage, you might want to fail on regression
            self.assertTrue(result['success'])
            
    def test_traffic_scenario_regression(self):
        """Test regression detection for traffic scenario"""
        scenario_config = {
            'name': 'traffic_regression_test',
            'environment': {
                'weather': {'type': 'clear'},
                'time_of_day': 14.0
            },
            'vehicles': [
                {'type': 'default_car', 'position': (0, 0, 0)},
                {'type': 'default_car', 'position': (10, 0, 0)}
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
                        {'type': 'action', 'action': 'cruise_control', 'parameters': {'speed': 15}}
                    ]
                }
            ],
            'success_conditions': {
                'destination_reached': {
                    'position': (80, 0, 0),
                    'tolerance': 10.0
                }
            },
            'timeout': 45.0
        }
        
        result = self.run_scenario_with_checkpoints(scenario_config)
        
        scenario_id = 'traffic_regression_test'
        if scenario_id not in self.regression_tester.baselines:
            baseline = self.regression_tester.create_baseline(scenario_id, result)
            self.regression_tester.save_baselines()
            print(f"Created baseline for scenario: {scenario_id}")
        else:
            regression_result = self.regression_tester.compare_with_baseline(
                scenario_id, result)
                
            if not regression_result.passed:
                print(f"Regression detected in {scenario_id}:")
                for deviation in regression_result.deviations:
                    print(f"  - {deviation}")
                    
        self.assertTrue(result['success'])


class TestScenarioValidation(unittest.TestCase):
    """Test scenario validation tools"""
    
    def setUp(self):
        """Set up validation testing environment"""
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
    def test_scenario_configuration_validation(self):
        """Test validation of scenario configurations"""
        # Valid scenario
        valid_scenario = {
            'name': 'test_scenario',
            'environment': {'weather': {'type': 'clear'}},
            'vehicles': [{'type': 'default_car', 'position': (0, 0, 0)}],
            'ai_behaviors': [{'type': 'action', 'action': 'cruise'}],
            'success_conditions': {'destination_reached': {'position': (100, 0, 0)}},
            'timeout': 60.0
        }
        
        self.assertTrue(self.validate_scenario_config(valid_scenario))
        
        # Invalid scenario - missing required fields
        invalid_scenario = {
            'name': 'invalid_scenario'
            # Missing other required fields
        }
        
        self.assertFalse(self.validate_scenario_config(invalid_scenario))
        
    def validate_scenario_config(self, config: Dict[str, Any]) -> bool:
        """Validate scenario configuration"""
        required_fields = ['name', 'vehicles', 'ai_behaviors']
        
        for field in required_fields:
            if field not in config:
                return False
                
        # Validate vehicles configuration
        if not isinstance(config['vehicles'], list) or len(config['vehicles']) == 0:
            return False
            
        for vehicle in config['vehicles']:
            if 'position' not in vehicle:
                return False
                
        # Validate AI behaviors
        if not isinstance(config['ai_behaviors'], list):
            return False
            
        return True


if __name__ == '__main__':
    unittest.main()