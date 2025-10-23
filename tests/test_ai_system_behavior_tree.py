"""
Unit tests for AI system behavior tree integration
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import os

from src.core.ai_system import AISystem, AIState, AIDecision
from src.core.behavior_tree import BehaviorTree, create_basic_driving_tree, NodeStatus

class TestAISystemBehaviorTreeIntegration(unittest.TestCase):
    """Test AI system integration with behavior trees"""
    
    def setUp(self):
        self.ai_system = AISystem()
        self.vehicle_id = "test_vehicle_001"
    
    def test_register_vehicle_with_default_tree(self):
        """Test registering vehicle creates default behavior tree"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Check vehicle is registered
        self.assertIn(self.vehicle_id, self.ai_system.vehicle_states)
        self.assertEqual(self.ai_system.vehicle_states[self.vehicle_id], AIState.IDLE)
        
        # Check behavior tree is created
        self.assertIn(self.vehicle_id, self.ai_system.behavior_trees)
        tree = self.ai_system.behavior_trees[self.vehicle_id]
        self.assertIsInstance(tree, BehaviorTree)
        self.assertEqual(tree.name, "BasicDriving")
    
    def test_register_vehicle_with_custom_tree(self):
        """Test registering vehicle with custom behavior tree"""
        custom_tree = BehaviorTree("CustomTree")
        self.ai_system.register_vehicle(self.vehicle_id, custom_tree)
        
        # Check custom tree is used
        tree = self.ai_system.behavior_trees[self.vehicle_id]
        self.assertEqual(tree.name, "CustomTree")
    
    def test_unregister_vehicle(self):
        """Test unregistering vehicle removes all data"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Verify vehicle is registered
        self.assertIn(self.vehicle_id, self.ai_system.vehicle_states)
        self.assertIn(self.vehicle_id, self.ai_system.behavior_trees)
        self.assertIn(self.vehicle_id, self.ai_system.vehicle_controls)
        
        # Unregister vehicle
        self.ai_system.unregister_vehicle(self.vehicle_id)
        
        # Verify all data is removed
        self.assertNotIn(self.vehicle_id, self.ai_system.vehicle_states)
        self.assertNotIn(self.vehicle_id, self.ai_system.behavior_trees)
        self.assertNotIn(self.vehicle_id, self.ai_system.vehicle_controls)
    
    def test_update_sensor_data(self):
        """Test updating sensor data for vehicle"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        sensor_data = {
            'lidar': {'distances': [10.0, 15.0, 20.0]},
            'camera': {'objects': []},
            'gps': {'lat': 40.7128, 'lon': -74.0060}
        }
        
        self.ai_system.update_sensor_data(self.vehicle_id, sensor_data)
        
        # Verify sensor data is stored
        self.assertEqual(self.ai_system.sensor_data[self.vehicle_id], sensor_data)
    
    def test_behavior_tree_execution_during_update(self):
        """Test behavior tree execution during AI update"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Setup sensor data for normal driving
        sensor_data = {
            'lidar': {'distances': [20.0, 25.0, 30.0, 15.0, 18.0]}
        }
        self.ai_system.update_sensor_data(self.vehicle_id, sensor_data)
        
        # Mock the decision_made signal
        decision_made_mock = Mock()
        self.ai_system.decision_made.connect(decision_made_mock)
        
        # Update AI system
        self.ai_system.update(0.1)
        
        # Verify decision was made
        decision_made_mock.assert_called_once()
        call_args = decision_made_mock.call_args
        self.assertEqual(call_args[0][0], self.vehicle_id)  # vehicle_id
        
        # Check decision contains control data
        decision = call_args[0][1]
        self.assertIn('controls', decision)
        self.assertIn('status', decision)
        self.assertIn('execution_count', decision)
        
        # Verify controls are set
        controls = decision['controls']
        self.assertIn('throttle', controls)
        self.assertIn('brake', controls)
        self.assertIn('steering', controls)
        self.assertIn('target_speed', controls)
    
    def test_obstacle_avoidance_behavior(self):
        """Test obstacle avoidance behavior through behavior tree"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Setup sensor data with close obstacle
        sensor_data = {
            'lidar': {'distances': [3.0, 2.5, 4.0, 10.0, 15.0]}
        }
        self.ai_system.update_sensor_data(self.vehicle_id, sensor_data)
        
        # Update AI system
        self.ai_system.update(0.1)
        
        # Check that braking was applied
        controls = self.ai_system.get_vehicle_controls(self.vehicle_id)
        self.assertEqual(controls['brake'], 1.0)  # Emergency brake
        self.assertEqual(controls['throttle'], 0.0)
    
    def test_normal_driving_behavior(self):
        """Test normal driving behavior when no obstacles"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Setup sensor data with no obstacles
        sensor_data = {
            'lidar': {'distances': [20.0, 25.0, 30.0, 15.0, 18.0]}
        }
        self.ai_system.update_sensor_data(self.vehicle_id, sensor_data)
        
        # Set speed and limit for normal driving condition
        tree = self.ai_system.get_behavior_tree(self.vehicle_id)
        blackboard = tree.get_blackboard()
        blackboard.set('current_speed', 45.0)
        blackboard.set('speed_limit', 50.0)
        
        # Update AI system
        self.ai_system.update(0.1)
        
        # Check that normal driving was applied
        controls = self.ai_system.get_vehicle_controls(self.vehicle_id)
        self.assertEqual(controls['throttle'], 0.5)
        self.assertEqual(controls['target_speed'], 50.0)
    
    def test_set_custom_behavior_tree(self):
        """Test setting custom behavior tree for vehicle"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Create custom tree
        custom_tree = BehaviorTree("CustomBehavior")
        
        # Set custom tree
        self.ai_system.set_behavior_tree(self.vehicle_id, custom_tree)
        
        # Verify custom tree is set
        tree = self.ai_system.get_behavior_tree(self.vehicle_id)
        self.assertEqual(tree.name, "CustomBehavior")
    
    def test_get_behavior_tree(self):
        """Test getting behavior tree for vehicle"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        tree = self.ai_system.get_behavior_tree(self.vehicle_id)
        self.assertIsInstance(tree, BehaviorTree)
        
        # Test non-existent vehicle
        non_existent_tree = self.ai_system.get_behavior_tree("non_existent")
        self.assertIsNone(non_existent_tree)
    
    def test_create_behavior_tree_from_definition(self):
        """Test creating behavior tree from definition"""
        tree_definition = {
            'name': 'TestTree',
            'root': {
                'name': 'Root',
                'type': 'composite',
                'parameters': {},
                'children': []
            }
        }
        
        tree = self.ai_system.create_behavior_tree(tree_definition)
        self.assertIsInstance(tree, BehaviorTree)
        self.assertEqual(tree.name, 'TestTree')
    
    def test_vehicle_state_management(self):
        """Test vehicle AI state management"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Test initial state
        state = self.ai_system.get_vehicle_state(self.vehicle_id)
        self.assertEqual(state, AIState.IDLE)
        
        # Test state change
        behavior_changed_mock = Mock()
        self.ai_system.behavior_changed.connect(behavior_changed_mock)
        
        self.ai_system.set_vehicle_state(self.vehicle_id, AIState.DRIVING)
        
        # Verify state changed
        state = self.ai_system.get_vehicle_state(self.vehicle_id)
        self.assertEqual(state, AIState.DRIVING)
        
        # Verify signal was emitted
        behavior_changed_mock.assert_called_once_with(self.vehicle_id, AIState.DRIVING.value)
    
    def test_get_vehicle_controls(self):
        """Test getting vehicle control outputs"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Test initial controls
        controls = self.ai_system.get_vehicle_controls(self.vehicle_id)
        expected_controls = {
            'throttle': 0.0,
            'brake': 0.0,
            'steering': 0.0,
            'target_speed': 0.0
        }
        self.assertEqual(controls, expected_controls)
        
        # Test non-existent vehicle
        non_existent_controls = self.ai_system.get_vehicle_controls("non_existent")
        self.assertEqual(non_existent_controls, expected_controls)
    
    def test_reset_vehicle(self):
        """Test resetting individual vehicle"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Set some state
        self.ai_system.set_vehicle_state(self.vehicle_id, AIState.DRIVING)
        tree = self.ai_system.get_behavior_tree(self.vehicle_id)
        tree.execution_count = 5
        
        # Reset vehicle
        self.ai_system.reset_vehicle(self.vehicle_id)
        
        # Verify reset
        state = self.ai_system.get_vehicle_state(self.vehicle_id)
        self.assertEqual(state, AIState.IDLE)
        
        controls = self.ai_system.get_vehicle_controls(self.vehicle_id)
        self.assertEqual(controls['throttle'], 0.0)
        self.assertEqual(controls['brake'], 0.0)
    
    def test_reset_all_vehicles(self):
        """Test resetting entire AI system"""
        # Register multiple vehicles
        vehicle_ids = ["vehicle_1", "vehicle_2", "vehicle_3"]
        for vid in vehicle_ids:
            self.ai_system.register_vehicle(vid)
            self.ai_system.set_vehicle_state(vid, AIState.DRIVING)
        
        # Reset system
        self.ai_system.reset()
        
        # Verify all vehicles are reset
        for vid in vehicle_ids:
            state = self.ai_system.get_vehicle_state(vid)
            self.assertEqual(state, AIState.IDLE)
            
            controls = self.ai_system.get_vehicle_controls(vid)
            self.assertEqual(controls['throttle'], 0.0)
    
    def test_behavior_tree_serialization_integration(self):
        """Test behavior tree save/load integration"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            # Save behavior tree
            self.ai_system.save_behavior_tree(self.vehicle_id, temp_filepath)
            
            # Verify file exists
            self.assertTrue(os.path.exists(temp_filepath))
            
            # Create new vehicle and load tree
            new_vehicle_id = "new_vehicle"
            self.ai_system.register_vehicle(new_vehicle_id)
            self.ai_system.load_behavior_tree(new_vehicle_id, temp_filepath)
            
            # Verify trees are similar
            original_tree = self.ai_system.get_behavior_tree(self.vehicle_id)
            loaded_tree = self.ai_system.get_behavior_tree(new_vehicle_id)
            
            self.assertEqual(original_tree.name, loaded_tree.name)
            
        finally:
            # Clean up
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def test_error_handling_in_behavior_tree_execution(self):
        """Test error handling during behavior tree execution"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Create a mock tree that raises an exception
        mock_tree = Mock(spec=BehaviorTree)
        mock_tree.execute.side_effect = Exception("Test exception")
        mock_tree.get_blackboard.return_value = Mock()
        mock_tree.execution_count = 0
        
        self.ai_system.behavior_trees[self.vehicle_id] = mock_tree
        
        # Update should handle the exception gracefully
        self.ai_system.update(0.1)
        
        # Vehicle should be in emergency state
        state = self.ai_system.get_vehicle_state(self.vehicle_id)
        self.assertEqual(state, AIState.EMERGENCY)

class TestAISystemSignals(unittest.TestCase):
    """Test AI system signal emissions"""
    
    def setUp(self):
        self.ai_system = AISystem()
        self.vehicle_id = "test_vehicle"
    
    def test_decision_made_signal(self):
        """Test decision_made signal emission"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Connect mock to signal
        decision_mock = Mock()
        self.ai_system.decision_made.connect(decision_mock)
        
        # Update AI system
        self.ai_system.update(0.1)
        
        # Verify signal was emitted
        decision_mock.assert_called_once()
        
        # Check signal arguments
        args = decision_mock.call_args[0]
        self.assertEqual(args[0], self.vehicle_id)
        self.assertIsInstance(args[1], dict)
    
    def test_behavior_changed_signal(self):
        """Test behavior_changed signal emission"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Connect mock to signal
        behavior_mock = Mock()
        self.ai_system.behavior_changed.connect(behavior_mock)
        
        # Change vehicle state
        self.ai_system.set_vehicle_state(self.vehicle_id, AIState.DRIVING)
        
        # Verify signal was emitted
        behavior_mock.assert_called_once_with(self.vehicle_id, AIState.DRIVING.value)
    
    def test_tree_execution_completed_signal(self):
        """Test tree_execution_completed signal emission"""
        self.ai_system.register_vehicle(self.vehicle_id)
        
        # Connect mock to signal
        completion_mock = Mock()
        self.ai_system.tree_execution_completed.connect(completion_mock)
        
        # Update AI system (should complete tree execution)
        self.ai_system.update(0.1)
        
        # Verify signal was emitted (basic driving tree should complete)
        completion_mock.assert_called_once()
        
        # Check signal arguments
        args = completion_mock.call_args[0]
        self.assertEqual(args[0], self.vehicle_id)
        self.assertIn(args[1], [NodeStatus.SUCCESS.value, NodeStatus.FAILURE.value])

if __name__ == '__main__':
    unittest.main()