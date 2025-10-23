"""
Unit tests for behavior tree system
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch

from src.core.behavior_tree import (
    BehaviorTree, BehaviorNode, BlackboardData, NodeStatus, NodeType,
    SequenceNode, SelectorNode, ParallelNode, InverterNode, RepeatNode,
    ConditionNode, ActionNode, ObstacleDetectedCondition, SpeedLimitCondition,
    DriveForwardAction, BrakeAction, SteerAction, BehaviorTreeSerializer,
    create_basic_driving_tree
)

class TestBlackboardData(unittest.TestCase):
    """Test blackboard data storage"""
    
    def setUp(self):
        self.blackboard = BlackboardData()
    
    def test_set_and_get(self):
        """Test setting and getting values"""
        self.blackboard.set('test_key', 'test_value')
        self.assertEqual(self.blackboard.get('test_key'), 'test_value')
    
    def test_get_default(self):
        """Test getting with default value"""
        self.assertEqual(self.blackboard.get('nonexistent', 'default'), 'default')
    
    def test_has_key(self):
        """Test checking if key exists"""
        self.blackboard.set('existing_key', 'value')
        self.assertTrue(self.blackboard.has('existing_key'))
        self.assertFalse(self.blackboard.has('nonexistent_key'))
    
    def test_clear(self):
        """Test clearing blackboard"""
        self.blackboard.set('key1', 'value1')
        self.blackboard.set('key2', 'value2')
        self.blackboard.clear()
        self.assertFalse(self.blackboard.has('key1'))
        self.assertFalse(self.blackboard.has('key2'))

class TestSequenceNode(unittest.TestCase):
    """Test sequence node behavior"""
    
    def setUp(self):
        self.blackboard = BlackboardData()
        self.sequence = SequenceNode("TestSequence")
    
    def test_empty_sequence_success(self):
        """Test empty sequence returns success"""
        status = self.sequence.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.SUCCESS)
    
    def test_all_children_success(self):
        """Test sequence with all successful children"""
        # Create mock children that return SUCCESS
        child1 = Mock(spec=BehaviorNode)
        child1.execute.return_value = NodeStatus.SUCCESS
        child1.reset = Mock()
        
        child2 = Mock(spec=BehaviorNode)
        child2.execute.return_value = NodeStatus.SUCCESS
        child2.reset = Mock()
        
        self.sequence.add_child(child1)
        self.sequence.add_child(child2)
        
        status = self.sequence.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.SUCCESS)
        
        # Verify both children were executed
        child1.execute.assert_called_once()
        child2.execute.assert_called_once()
    
    def test_first_child_failure(self):
        """Test sequence fails when first child fails"""
        child1 = Mock(spec=BehaviorNode)
        child1.execute.return_value = NodeStatus.FAILURE
        child1.reset = Mock()
        
        child2 = Mock(spec=BehaviorNode)
        child2.execute.return_value = NodeStatus.SUCCESS
        child2.reset = Mock()
        
        self.sequence.add_child(child1)
        self.sequence.add_child(child2)
        
        status = self.sequence.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.FAILURE)
        
        # Verify only first child was executed
        child1.execute.assert_called_once()
        child2.execute.assert_not_called()
    
    def test_running_child(self):
        """Test sequence returns running when child is running"""
        child1 = Mock(spec=BehaviorNode)
        child1.execute.return_value = NodeStatus.RUNNING
        child1.reset = Mock()
        
        self.sequence.add_child(child1)
        
        status = self.sequence.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.RUNNING)

class TestSelectorNode(unittest.TestCase):
    """Test selector node behavior"""
    
    def setUp(self):
        self.blackboard = BlackboardData()
        self.selector = SelectorNode("TestSelector")
    
    def test_empty_selector_failure(self):
        """Test empty selector returns failure"""
        status = self.selector.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.FAILURE)
    
    def test_first_child_success(self):
        """Test selector succeeds when first child succeeds"""
        child1 = Mock(spec=BehaviorNode)
        child1.execute.return_value = NodeStatus.SUCCESS
        child1.reset = Mock()
        
        child2 = Mock(spec=BehaviorNode)
        child2.execute.return_value = NodeStatus.FAILURE
        child2.reset = Mock()
        
        self.selector.add_child(child1)
        self.selector.add_child(child2)
        
        status = self.selector.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.SUCCESS)
        
        # Verify only first child was executed
        child1.execute.assert_called_once()
        child2.execute.assert_not_called()
    
    def test_all_children_failure(self):
        """Test selector fails when all children fail"""
        child1 = Mock(spec=BehaviorNode)
        child1.execute.return_value = NodeStatus.FAILURE
        child1.reset = Mock()
        
        child2 = Mock(spec=BehaviorNode)
        child2.execute.return_value = NodeStatus.FAILURE
        child2.reset = Mock()
        
        self.selector.add_child(child1)
        self.selector.add_child(child2)
        
        status = self.selector.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.FAILURE)
        
        # Verify both children were executed
        child1.execute.assert_called_once()
        child2.execute.assert_called_once()

class TestParallelNode(unittest.TestCase):
    """Test parallel node behavior"""
    
    def setUp(self):
        self.blackboard = BlackboardData()
        self.parallel = ParallelNode("TestParallel", {'success_threshold': 2, 'failure_threshold': 2})
    
    def test_success_threshold_met(self):
        """Test parallel succeeds when success threshold is met"""
        child1 = Mock(spec=BehaviorNode)
        child1.execute.return_value = NodeStatus.SUCCESS
        
        child2 = Mock(spec=BehaviorNode)
        child2.execute.return_value = NodeStatus.SUCCESS
        
        child3 = Mock(spec=BehaviorNode)
        child3.execute.return_value = NodeStatus.FAILURE
        
        self.parallel.add_child(child1)
        self.parallel.add_child(child2)
        self.parallel.add_child(child3)
        
        status = self.parallel.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.SUCCESS)
    
    def test_failure_threshold_met(self):
        """Test parallel fails when failure threshold is met"""
        child1 = Mock(spec=BehaviorNode)
        child1.execute.return_value = NodeStatus.FAILURE
        
        child2 = Mock(spec=BehaviorNode)
        child2.execute.return_value = NodeStatus.FAILURE
        
        child3 = Mock(spec=BehaviorNode)
        child3.execute.return_value = NodeStatus.SUCCESS
        
        self.parallel.add_child(child1)
        self.parallel.add_child(child2)
        self.parallel.add_child(child3)
        
        status = self.parallel.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.FAILURE)

class TestInverterNode(unittest.TestCase):
    """Test inverter decorator node"""
    
    def setUp(self):
        self.blackboard = BlackboardData()
        self.inverter = InverterNode("TestInverter")
    
    def test_invert_success(self):
        """Test inverter converts success to failure"""
        child = Mock(spec=BehaviorNode)
        child.execute.return_value = NodeStatus.SUCCESS
        
        self.inverter.add_child(child)
        
        status = self.inverter.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.FAILURE)
    
    def test_invert_failure(self):
        """Test inverter converts failure to success"""
        child = Mock(spec=BehaviorNode)
        child.execute.return_value = NodeStatus.FAILURE
        
        self.inverter.add_child(child)
        
        status = self.inverter.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.SUCCESS)
    
    def test_running_unchanged(self):
        """Test inverter leaves running status unchanged"""
        child = Mock(spec=BehaviorNode)
        child.execute.return_value = NodeStatus.RUNNING
        
        self.inverter.add_child(child)
        
        status = self.inverter.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.RUNNING)

class TestRepeatNode(unittest.TestCase):
    """Test repeat decorator node"""
    
    def setUp(self):
        self.blackboard = BlackboardData()
        self.repeat = RepeatNode("TestRepeat", {'max_repeats': 3})
    
    def test_successful_repeats(self):
        """Test repeat node executes child multiple times"""
        child = Mock(spec=BehaviorNode)
        child.execute.return_value = NodeStatus.SUCCESS
        child.reset = Mock()
        
        self.repeat.add_child(child)
        
        status = self.repeat.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.SUCCESS)
        
        # Should execute 3 times
        self.assertEqual(child.execute.call_count, 3)
        # Reset is called after each successful execution except the last, plus when repeat node resets
        self.assertTrue(child.reset.call_count >= 2)
    
    def test_child_failure_stops_repeat(self):
        """Test repeat stops on child failure"""
        child = Mock(spec=BehaviorNode)
        child.execute.side_effect = [NodeStatus.SUCCESS, NodeStatus.FAILURE]
        child.reset = Mock()
        
        self.repeat.add_child(child)
        
        status = self.repeat.execute(self.blackboard, 0.1)
        self.assertEqual(status, NodeStatus.FAILURE)
        
        # Should execute only twice (success, then failure)
        self.assertEqual(child.execute.call_count, 2)

class TestConditionNodes(unittest.TestCase):
    """Test condition nodes"""
    
    def setUp(self):
        self.blackboard = BlackboardData()
    
    def test_obstacle_detected_condition(self):
        """Test obstacle detection condition"""
        # Setup sensor data with close obstacle
        sensor_data = {
            'lidar': {
                'distances': [3.0, 4.0, 2.5, 10.0, 15.0]
            }
        }
        self.blackboard.set('sensor_data', sensor_data)
        
        condition = ObstacleDetectedCondition("ObstacleCheck", {'min_distance': 5.0})
        status = condition.execute(self.blackboard, 0.1)
        
        self.assertEqual(status, NodeStatus.SUCCESS)  # Obstacle detected
    
    def test_no_obstacle_condition(self):
        """Test no obstacle detected"""
        sensor_data = {
            'lidar': {
                'distances': [10.0, 15.0, 20.0, 12.0, 8.0]
            }
        }
        self.blackboard.set('sensor_data', sensor_data)
        
        condition = ObstacleDetectedCondition("ObstacleCheck", {'min_distance': 5.0})
        status = condition.execute(self.blackboard, 0.1)
        
        self.assertEqual(status, NodeStatus.FAILURE)  # No obstacle
    
    def test_speed_limit_condition(self):
        """Test speed limit condition"""
        self.blackboard.set('current_speed', 45.0)
        self.blackboard.set('speed_limit', 50.0)
        
        condition = SpeedLimitCondition("SpeedCheck", {'tolerance': 5.0})
        status = condition.execute(self.blackboard, 0.1)
        
        self.assertEqual(status, NodeStatus.SUCCESS)  # Within limit
    
    def test_speed_limit_exceeded(self):
        """Test speed limit exceeded"""
        self.blackboard.set('current_speed', 60.0)
        self.blackboard.set('speed_limit', 50.0)
        
        condition = SpeedLimitCondition("SpeedCheck", {'tolerance': 5.0})
        status = condition.execute(self.blackboard, 0.1)
        
        self.assertEqual(status, NodeStatus.FAILURE)  # Exceeded limit

class TestActionNodes(unittest.TestCase):
    """Test action nodes"""
    
    def setUp(self):
        self.blackboard = BlackboardData()
    
    def test_drive_forward_action(self):
        """Test drive forward action"""
        action = DriveForwardAction("DriveForward", {'target_speed': 60.0})
        status = action.execute(self.blackboard, 0.1)
        
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(self.blackboard.get('throttle'), 0.5)
        self.assertEqual(self.blackboard.get('brake'), 0.0)
        self.assertEqual(self.blackboard.get('steering'), 0.0)
        self.assertEqual(self.blackboard.get('target_speed'), 60.0)
    
    def test_brake_action(self):
        """Test brake action"""
        action = BrakeAction("Brake", {'brake_force': 0.9})
        status = action.execute(self.blackboard, 0.1)
        
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(self.blackboard.get('throttle'), 0.0)
        self.assertEqual(self.blackboard.get('brake'), 0.9)
        self.assertEqual(self.blackboard.get('steering'), 0.0)
    
    def test_steer_action(self):
        """Test steer action"""
        action = SteerAction("Steer", {'steering_angle': 0.3})
        status = action.execute(self.blackboard, 0.1)
        
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(self.blackboard.get('steering'), 0.3)

class TestBehaviorTree(unittest.TestCase):
    """Test behavior tree execution"""
    
    def setUp(self):
        self.tree = BehaviorTree("TestTree")
    
    def test_empty_tree(self):
        """Test behavior tree with no root"""
        status = self.tree.execute(0.1)
        self.assertEqual(status, NodeStatus.FAILURE)
    
    def test_simple_tree_execution(self):
        """Test simple behavior tree execution"""
        root = SequenceNode("Root")
        
        # Add a simple action
        action = DriveForwardAction("Drive", {'target_speed': 30.0})
        root.add_child(action)
        
        self.tree.set_root(root)
        
        status = self.tree.execute(0.1)
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(self.tree.execution_count, 1)
        
        # Check blackboard was updated
        blackboard = self.tree.get_blackboard()
        self.assertEqual(blackboard.get('target_speed'), 30.0)
    
    def test_tree_reset(self):
        """Test behavior tree reset"""
        root = SequenceNode("Root")
        self.tree.set_root(root)
        
        # Execute tree
        self.tree.execute(0.1)
        self.assertEqual(self.tree.execution_count, 1)
        
        # Reset tree
        self.tree.reset()
        self.assertEqual(self.tree.execution_count, 0)

class TestBehaviorTreeSerialization(unittest.TestCase):
    """Test behavior tree serialization"""
    
    def setUp(self):
        self.serializer = BehaviorTreeSerializer()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up temp files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def test_save_and_load_tree(self):
        """Test saving and loading behavior tree"""
        # Create a simple tree
        tree = create_basic_driving_tree()
        
        # Save to file
        filepath = os.path.join(self.temp_dir, 'test_tree.json')
        self.serializer.save_to_file(tree, filepath)
        
        # Verify file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Load from file
        loaded_tree = self.serializer.load_from_file(filepath)
        
        # Verify tree properties
        self.assertEqual(loaded_tree.name, tree.name)
        self.assertIsNotNone(loaded_tree.root)
    
    def test_tree_to_dict(self):
        """Test behavior tree serialization to dictionary"""
        tree = BehaviorTree("TestTree")
        root = SequenceNode("Root")
        action = DriveForwardAction("Drive")
        root.add_child(action)
        tree.set_root(root)
        
        tree_dict = tree.to_dict()
        
        self.assertEqual(tree_dict['name'], 'TestTree')
        self.assertIsNotNone(tree_dict['root'])
        self.assertEqual(tree_dict['root']['name'], 'Root')
        self.assertEqual(len(tree_dict['root']['children']), 1)

class TestBasicDrivingTree(unittest.TestCase):
    """Test the basic driving behavior tree"""
    
    def setUp(self):
        self.tree = create_basic_driving_tree()
        self.blackboard = self.tree.get_blackboard()
    
    def test_obstacle_avoidance_behavior(self):
        """Test obstacle avoidance takes priority"""
        # Setup obstacle detected
        sensor_data = {
            'lidar': {
                'distances': [2.0, 3.0, 1.5, 10.0, 15.0]
            }
        }
        self.blackboard.set('sensor_data', sensor_data)
        self.blackboard.set('current_speed', 40.0)
        self.blackboard.set('speed_limit', 50.0)
        
        status = self.tree.execute(0.1)
        
        # Should brake due to obstacle
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(self.blackboard.get('brake'), 1.0)
        self.assertEqual(self.blackboard.get('throttle'), 0.0)
    
    def test_normal_driving_behavior(self):
        """Test normal driving when no obstacles"""
        # Setup no obstacles
        sensor_data = {
            'lidar': {
                'distances': [20.0, 25.0, 15.0, 30.0, 18.0]
            }
        }
        self.blackboard.set('sensor_data', sensor_data)
        self.blackboard.set('current_speed', 45.0)
        self.blackboard.set('speed_limit', 50.0)
        
        status = self.tree.execute(0.1)
        
        # Should drive forward normally
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(self.blackboard.get('throttle'), 0.5)
        self.assertEqual(self.blackboard.get('target_speed'), 50.0)

if __name__ == '__main__':
    unittest.main()