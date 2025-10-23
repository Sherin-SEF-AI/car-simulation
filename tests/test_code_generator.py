"""
Unit tests for the code generation system.
Tests CodeGenerator, BehaviorTreeDebugger, and RealTimeBehaviorValidator.
"""

import pytest
import json
import ast
from unittest.mock import Mock, patch

from src.ui.code_generator import (
    CodeGenerator, BehaviorTreeDebugger, RealTimeBehaviorValidator,
    ValidationResult, ExecutionState, DebugState
)


class TestValidationResult:
    """Test cases for ValidationResult class"""
    
    def test_validation_result_initialization(self):
        """Test ValidationResult initialization"""
        result = ValidationResult(True)
        assert result.is_valid == True
        assert result.errors == []
        assert result.warnings == []
        
        result_with_errors = ValidationResult(False, ['error1'], ['warning1'])
        assert result_with_errors.is_valid == False
        assert result_with_errors.errors == ['error1']
        assert result_with_errors.warnings == ['warning1']
    
    def test_add_error(self):
        """Test adding errors to validation result"""
        result = ValidationResult(True)
        result.add_error("Test error")
        
        assert result.is_valid == False
        assert "Test error" in result.errors
    
    def test_add_warning(self):
        """Test adding warnings to validation result"""
        result = ValidationResult(True)
        result.add_warning("Test warning")
        
        assert result.is_valid == True  # Warnings don't affect validity
        assert "Test warning" in result.warnings
    
    def test_string_representation(self):
        """Test string representation of validation result"""
        result = ValidationResult(True)
        result.add_error("Error message")
        result.add_warning("Warning message")
        
        str_repr = str(result)
        assert "Valid: False" in str_repr
        assert "Error message" in str_repr
        assert "Warning message" in str_repr


class TestCodeGenerator:
    """Test cases for CodeGenerator class"""
    
    @pytest.fixture
    def code_generator(self):
        return CodeGenerator()
    
    @pytest.fixture
    def simple_tree_data(self):
        return {
            'blocks': [
                {
                    'id': 'sensor-1',
                    'type': 'camera_sensor',
                    'position': {'x': 0, 'y': 0},
                    'data': {
                        'type': 'camera_sensor',
                        'name': 'Camera Sensor',
                        'category': 'sensor',
                        'parameters': {'resolution': '1920x1080', 'fps': 30},
                        'has_input': False,
                        'has_output': True
                    },
                    'connections': ['condition-1']
                },
                {
                    'id': 'condition-1',
                    'type': 'obstacle_detected',
                    'position': {'x': 200, 'y': 0},
                    'data': {
                        'type': 'obstacle_detected',
                        'name': 'Obstacle Detection',
                        'category': 'condition',
                        'parameters': {'distance_threshold': 5.0, 'angle_range': 30.0},
                        'has_input': True,
                        'has_output': True
                    },
                    'connections': ['action-1']
                },
                {
                    'id': 'action-1',
                    'type': 'brake',
                    'position': {'x': 400, 'y': 0},
                    'data': {
                        'type': 'brake',
                        'name': 'Brake',
                        'category': 'action',
                        'parameters': {'brake_force': 0.8},
                        'has_input': True,
                        'has_output': True
                    },
                    'connections': []
                }
            ],
            'connections': []
        }
    
    def test_code_generator_initialization(self, code_generator):
        """Test CodeGenerator initialization"""
        assert code_generator.code_templates is not None
        assert code_generator.validation_rules is not None
        assert code_generator.debug_enabled == False
        assert code_generator.debug_states == {}
        
        # Check that templates exist for key block types
        expected_templates = [
            'camera_sensor', 'lidar_sensor', 'gps_sensor',
            'obstacle_detected', 'speed_check', 'lane_detection',
            'accelerate', 'brake', 'steer', 'follow_path',
            'sequence', 'selector', 'parallel'
        ]
        
        for template_type in expected_templates:
            assert template_type in code_generator.code_templates
    
    def test_validate_behavior_tree_valid(self, code_generator, simple_tree_data):
        """Test validation of a valid behavior tree"""
        result = code_generator.validate_behavior_tree(simple_tree_data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid == True
        assert len(result.errors) == 0
    
    def test_validate_behavior_tree_empty(self, code_generator):
        """Test validation of empty behavior tree"""
        result = code_generator.validate_behavior_tree({})
        
        assert result.is_valid == False
        assert "Empty behavior tree" in result.errors
    
    def test_validate_behavior_tree_no_blocks(self, code_generator):
        """Test validation of behavior tree with no blocks"""
        tree_data = {'blocks': [], 'connections': []}
        result = code_generator.validate_behavior_tree(tree_data)
        
        assert result.is_valid == False
        assert "No blocks in behavior tree" in result.errors
    
    def test_validate_behavior_tree_no_root(self, code_generator):
        """Test validation of behavior tree with no root node (circular dependency)"""
        tree_data = {
            'blocks': [
                {
                    'id': 'block-1',
                    'type': 'accelerate',
                    'data': {'parameters': {'acceleration': 1.0, 'max_speed': 50.0}},
                    'connections': ['block-2']
                },
                {
                    'id': 'block-2',
                    'type': 'brake',
                    'data': {'parameters': {'brake_force': 0.5}},
                    'connections': ['block-1']  # Circular dependency
                }
            ]
        }
        
        result = code_generator.validate_behavior_tree(tree_data)
        assert "No root node found" in result.errors
    
    def test_validate_behavior_tree_multiple_roots(self, code_generator):
        """Test validation of behavior tree with multiple root nodes"""
        tree_data = {
            'blocks': [
                {
                    'id': 'root-1',
                    'type': 'camera_sensor',
                    'data': {'parameters': {'resolution': '1920x1080', 'fps': 30}},
                    'connections': []
                },
                {
                    'id': 'root-2',
                    'type': 'lidar_sensor',
                    'data': {'parameters': {'range': 100.0, 'resolution': 0.1}},
                    'connections': []
                }
            ]
        }
        
        result = code_generator.validate_behavior_tree(tree_data)
        assert "Multiple root nodes found" in result.warnings
    
    def test_validate_block_missing_fields(self, code_generator):
        """Test validation of block with missing required fields"""
        block = {'id': 'test-block'}  # Missing 'type' and 'data'
        result = code_generator._validate_block(block)
        
        assert result.is_valid == False
        assert any("missing required field" in error for error in result.errors)
    
    def test_validate_block_unsupported_type(self, code_generator):
        """Test validation of block with unsupported type"""
        block = {
            'id': 'test-block',
            'type': 'unsupported_type',
            'data': {'parameters': {}}
        }
        result = code_generator._validate_block(block)
        
        assert result.is_valid == False
        assert "Unsupported block type" in result.errors[0]
    
    def test_validation_rules_positive_float(self, code_generator):
        """Test validation rule for positive float parameters"""
        parameters = {'distance_threshold': 5.0}
        rule = 'distance_threshold must be a positive float'
        
        assert code_generator._check_validation_rule(parameters, rule) == True
        
        # Test with invalid values
        parameters_invalid = {'distance_threshold': -1.0}
        assert code_generator._check_validation_rule(parameters_invalid, rule) == False
        
        parameters_invalid2 = {'distance_threshold': 'not_a_number'}
        assert code_generator._check_validation_rule(parameters_invalid2, rule) == False
    
    def test_validation_rules_range(self, code_generator):
        """Test validation rule for value ranges"""
        parameters = {'confidence_threshold': 0.8}
        rule = 'confidence_threshold must be between 0 and 1'
        
        assert code_generator._check_validation_rule(parameters, rule) == True
        
        # Test with invalid values
        parameters_invalid = {'confidence_threshold': 1.5}
        assert code_generator._check_validation_rule(parameters_invalid, rule) == False
    
    def test_find_reachable_nodes(self, code_generator, simple_tree_data):
        """Test finding reachable nodes from root"""
        blocks = simple_tree_data['blocks']
        root_node = blocks[0]  # sensor-1
        
        reachable = code_generator._find_reachable_nodes(root_node, blocks)
        
        expected_reachable = {'sensor-1', 'condition-1', 'action-1'}
        assert reachable == expected_reachable
    
    def test_generate_code_valid_tree(self, code_generator, simple_tree_data):
        """Test code generation for a valid behavior tree"""
        generated_code = code_generator.generate_code(simple_tree_data)
        
        assert isinstance(generated_code, str)
        assert len(generated_code) > 0
        
        # Check that generated code contains expected elements
        assert "class BehaviorTreeExecutor" in generated_code
        assert "def camera_sensor_sensor_1" in generated_code
        assert "def obstacle_detected_condition_1" in generated_code
        assert "def brake_action_1" in generated_code
        assert "def execute_behavior_tree" in generated_code
        
        # Verify the code is syntactically valid Python
        try:
            ast.parse(generated_code)
        except SyntaxError:
            pytest.fail("Generated code has syntax errors")
    
    def test_generate_code_invalid_tree(self, code_generator):
        """Test code generation for an invalid behavior tree"""
        invalid_tree = {'blocks': []}
        
        with pytest.raises(ValueError) as exc_info:
            code_generator.generate_code(invalid_tree)
        
        assert "Invalid behavior tree" in str(exc_info.value)
    
    def test_generate_block_code_camera_sensor(self, code_generator):
        """Test code generation for camera sensor block"""
        block = {
            'id': 'camera-test',
            'type': 'camera_sensor',
            'data': {
                'parameters': {'resolution': '1920x1080', 'fps': 30}
            }
        }
        
        code = code_generator._generate_block_code(block)
        
        assert "def camera_sensor_camera_test" in code
        assert 'resolution = "1920x1080"' in code
        assert "fps = 30" in code
        assert "get_camera_view" in code
    
    def test_generate_block_code_action_block(self, code_generator):
        """Test code generation for action block"""
        block = {
            'id': 'brake-test',
            'type': 'brake',
            'data': {
                'parameters': {'brake_force': 0.8}
            }
        }
        
        code = code_generator._generate_block_code(block)
        
        assert "def brake_brake_test" in code
        assert "brake_force = 0.8" in code
        assert "set_brake" in code
    
    def test_generate_block_code_unsupported_type(self, code_generator):
        """Test code generation for unsupported block type"""
        block = {
            'id': 'unknown-test',
            'type': 'unknown_type',
            'data': {'parameters': {}}
        }
        
        code = code_generator._generate_block_code(block)
        
        assert "Unsupported block type" in code
    
    def test_debugging_enable_disable(self, code_generator):
        """Test enabling and disabling debugging"""
        assert code_generator.debug_enabled == False
        
        code_generator.enable_debugging()
        assert code_generator.debug_enabled == True
        assert code_generator.debug_states == {}
        
        code_generator.disable_debugging()
        assert code_generator.debug_enabled == False
        assert code_generator.debug_states == {}
    
    def test_debug_state_management(self, code_generator):
        """Test debug state management"""
        debug_state = DebugState(
            node_id='test-node',
            node_type='test_type',
            state=ExecutionState.RUNNING,
            input_data={'input': 'test'},
            output_data={'output': 'result'},
            execution_time=0.1
        )
        
        # Initially no state
        assert code_generator.get_debug_state('test-node') is None
        
        # Set state
        code_generator.set_debug_state('test-node', debug_state)
        retrieved_state = code_generator.get_debug_state('test-node')
        
        assert retrieved_state == debug_state
        assert retrieved_state.node_id == 'test-node'
        assert retrieved_state.state == ExecutionState.RUNNING
    
    def test_step_through_execution(self, code_generator, simple_tree_data):
        """Test step-through execution for debugging"""
        debug_steps = code_generator.step_through_execution(simple_tree_data, Mock())
        
        assert isinstance(debug_steps, list)
        assert len(debug_steps) == 3  # Three blocks in simple tree
        
        for step in debug_steps:
            assert isinstance(step, DebugState)
            assert step.state == ExecutionState.READY
            assert step.node_id in ['sensor-1', 'condition-1', 'action-1']


class TestBehaviorTreeDebugger:
    """Test cases for BehaviorTreeDebugger class"""
    
    @pytest.fixture
    def code_generator(self):
        return CodeGenerator()
    
    @pytest.fixture
    def debugger(self, code_generator):
        return BehaviorTreeDebugger(code_generator)
    
    @pytest.fixture
    def simple_tree_data(self):
        return {
            'blocks': [
                {
                    'id': 'sensor-1',
                    'type': 'camera_sensor',
                    'data': {'parameters': {'resolution': '1920x1080', 'fps': 30}},
                    'connections': ['action-1']
                },
                {
                    'id': 'action-1',
                    'type': 'brake',
                    'data': {'parameters': {'brake_force': 0.8}},
                    'connections': []
                }
            ]
        }
    
    def test_debugger_initialization(self, debugger, code_generator):
        """Test BehaviorTreeDebugger initialization"""
        assert debugger.code_generator == code_generator
        assert debugger.breakpoints == set()
        assert debugger.current_step == 0
        assert debugger.execution_history == []
    
    def test_breakpoint_management(self, debugger):
        """Test breakpoint management"""
        # Add breakpoint
        debugger.add_breakpoint('node-1')
        assert 'node-1' in debugger.breakpoints
        
        # Add another breakpoint
        debugger.add_breakpoint('node-2')
        assert len(debugger.breakpoints) == 2
        
        # Remove breakpoint
        debugger.remove_breakpoint('node-1')
        assert 'node-1' not in debugger.breakpoints
        assert 'node-2' in debugger.breakpoints
        
        # Clear all breakpoints
        debugger.clear_breakpoints()
        assert len(debugger.breakpoints) == 0
    
    def test_debug_session_start(self, debugger, simple_tree_data):
        """Test starting a debug session"""
        mock_vehicle_state = Mock()
        
        debug_steps = debugger.start_debug_session(simple_tree_data, mock_vehicle_state)
        
        assert isinstance(debug_steps, list)
        assert len(debug_steps) > 0
        assert debugger.current_step == 0
        assert debugger.execution_history == debug_steps
        assert debugger.code_generator.debug_enabled == True
    
    def test_step_navigation(self, debugger, simple_tree_data):
        """Test stepping forward and backward through execution"""
        mock_vehicle_state = Mock()
        debugger.start_debug_session(simple_tree_data, mock_vehicle_state)
        
        # Step forward
        first_state = debugger.step_forward()
        assert first_state is not None
        assert debugger.current_step == 1
        
        second_state = debugger.step_forward()
        assert second_state is not None
        assert debugger.current_step == 2
        
        # Step backward
        prev_state = debugger.step_backward()
        assert prev_state == first_state
        assert debugger.current_step == 1
        
        # Step backward to beginning
        debugger.step_backward()
        assert debugger.current_step == 0
        
        # Try to step backward beyond beginning
        result = debugger.step_backward()
        assert result is None
        assert debugger.current_step == 0
    
    def test_get_current_state(self, debugger, simple_tree_data):
        """Test getting current debug state"""
        mock_vehicle_state = Mock()
        debugger.start_debug_session(simple_tree_data, mock_vehicle_state)
        
        # Initially at step 0, but current_state should be None until we step
        current_state = debugger.get_current_state()
        assert current_state is not None  # First state in history
        
        # Step forward and check current state
        debugger.step_forward()
        current_state = debugger.get_current_state()
        assert current_state is not None
    
    def test_execution_summary(self, debugger, simple_tree_data):
        """Test getting execution summary"""
        # Initially empty
        summary = debugger.get_execution_summary()
        assert summary['total_steps'] == 0
        assert summary['current_step'] == 0
        
        # After starting session
        mock_vehicle_state = Mock()
        debugger.start_debug_session(simple_tree_data, mock_vehicle_state)
        
        summary = debugger.get_execution_summary()
        assert summary['total_steps'] > 0
        assert summary['current_step'] == 0
        assert 'successful_steps' in summary
        assert 'failed_steps' in summary
        assert 'completion_rate' in summary


class TestRealTimeBehaviorValidator:
    """Test cases for RealTimeBehaviorValidator class"""
    
    @pytest.fixture
    def validator(self):
        return RealTimeBehaviorValidator()
    
    @pytest.fixture
    def simple_tree_data(self):
        return {
            'blocks': [
                {
                    'id': 'sensor-1',
                    'type': 'camera_sensor',
                    'data': {
                        'parameters': {'resolution': '1920x1080', 'fps': 30},
                        'category': 'sensor',
                        'has_input': False
                    },
                    'connections': []
                }
            ]
        }
    
    def test_validator_initialization(self, validator):
        """Test RealTimeBehaviorValidator initialization"""
        assert validator.validation_cache == {}
        assert validator.error_callbacks == []
    
    def test_error_callback_management(self, validator):
        """Test error callback management"""
        callback1 = Mock()
        callback2 = Mock()
        
        validator.add_error_callback(callback1)
        validator.add_error_callback(callback2)
        
        assert len(validator.error_callbacks) == 2
        assert callback1 in validator.error_callbacks
        assert callback2 in validator.error_callbacks
    
    def test_real_time_validation_caching(self, validator, simple_tree_data):
        """Test real-time validation with caching"""
        # First validation should compute result
        result1 = validator.validate_real_time(simple_tree_data)
        assert isinstance(result1, ValidationResult)
        
        # Second validation should use cache
        result2 = validator.validate_real_time(simple_tree_data)
        assert result2 == result1
        
        # Cache should contain the result
        assert len(validator.validation_cache) == 1
    
    def test_error_callback_invocation(self, validator):
        """Test that error callbacks are invoked for invalid trees"""
        callback_mock = Mock()
        validator.add_error_callback(callback_mock)
        
        # Invalid tree (empty)
        invalid_tree = {'blocks': []}
        result = validator.validate_real_time(invalid_tree)
        
        assert result.is_valid == False
        callback_mock.assert_called_once_with(result)
    
    def test_cache_clearing(self, validator, simple_tree_data):
        """Test cache clearing functionality"""
        # Add something to cache
        validator.validate_real_time(simple_tree_data)
        assert len(validator.validation_cache) > 0
        
        # Clear cache
        validator.clear_cache()
        assert len(validator.validation_cache) == 0
    
    def test_syntax_error_checking(self, validator):
        """Test syntax error checking for generated code"""
        # Valid Python code
        valid_code = "def test_function():\n    return True"
        errors = validator.check_syntax_errors(valid_code)
        assert len(errors) == 0
        
        # Invalid Python code
        invalid_code = "def test_function(\n    return True"  # Missing closing parenthesis
        errors = validator.check_syntax_errors(invalid_code)
        assert len(errors) > 0
        assert "Syntax error" in errors[0]
    
    def test_runtime_error_checking(self, validator):
        """Test runtime error checking"""
        # Tree with missing input connections
        tree_with_errors = {
            'blocks': [
                {
                    'id': 'action-1',
                    'type': 'brake',
                    'data': {
                        'category': 'action',
                        'has_input': True,
                        'parameters': {'brake_force': 'not_a_number'}  # Type mismatch
                    },
                    'connections': []
                }
            ]
        }
        
        errors = validator.check_runtime_errors(tree_with_errors)
        assert len(errors) > 0
        
        # Should detect missing input connection
        assert any("no input connection" in error for error in errors)
        
        # Should detect parameter type mismatch
        assert any("should be numeric" in error for error in errors)
    
    def test_create_cache_key(self, validator, simple_tree_data):
        """Test cache key creation"""
        key1 = validator._create_cache_key(simple_tree_data)
        key2 = validator._create_cache_key(simple_tree_data)
        
        # Same data should produce same key
        assert key1 == key2
        
        # Different data should produce different key
        modified_tree = simple_tree_data.copy()
        modified_tree['blocks'][0]['id'] = 'different-id'
        key3 = validator._create_cache_key(modified_tree)
        
        assert key1 != key3


class TestCodeGeneratorIntegration:
    """Integration tests for the complete code generation system"""
    
    @pytest.fixture
    def complete_tree_data(self):
        """A more complex behavior tree for integration testing"""
        return {
            'blocks': [
                {
                    'id': 'camera-sensor',
                    'type': 'camera_sensor',
                    'data': {
                        'parameters': {'resolution': '1920x1080', 'fps': 30},
                        'category': 'sensor',
                        'has_input': False,
                        'has_output': True
                    },
                    'connections': ['obstacle-check']
                },
                {
                    'id': 'lidar-sensor',
                    'type': 'lidar_sensor',
                    'data': {
                        'parameters': {'range': 100.0, 'resolution': 0.1},
                        'category': 'sensor',
                        'has_input': False,
                        'has_output': True
                    },
                    'connections': ['obstacle-check']
                },
                {
                    'id': 'obstacle-check',
                    'type': 'obstacle_detected',
                    'data': {
                        'parameters': {'distance_threshold': 5.0, 'angle_range': 30.0},
                        'category': 'condition',
                        'has_input': True,
                        'has_output': True
                    },
                    'connections': ['emergency-brake']
                },
                {
                    'id': 'emergency-brake',
                    'type': 'brake',
                    'data': {
                        'parameters': {'brake_force': 1.0},
                        'category': 'action',
                        'has_input': True,
                        'has_output': True
                    },
                    'connections': []
                }
            ]
        }
    
    def test_complete_workflow(self, complete_tree_data):
        """Test complete workflow from validation to code generation to debugging"""
        # Step 1: Validate the tree
        code_generator = CodeGenerator()
        validation_result = code_generator.validate_behavior_tree(complete_tree_data)
        
        assert validation_result.is_valid == True
        assert len(validation_result.errors) == 0
        
        # Step 2: Generate code
        generated_code = code_generator.generate_code(complete_tree_data)
        
        assert isinstance(generated_code, str)
        assert len(generated_code) > 0
        
        # Verify code contains all expected functions
        expected_functions = [
            'camera_sensor_camera_sensor',
            'lidar_sensor_lidar_sensor',
            'obstacle_detected_obstacle_check',
            'brake_emergency_brake'
        ]
        
        for func_name in expected_functions:
            assert func_name in generated_code
        
        # Step 3: Check syntax
        validator = RealTimeBehaviorValidator()
        syntax_errors = validator.check_syntax_errors(generated_code)
        assert len(syntax_errors) == 0
        
        # Step 4: Test debugging
        debugger = BehaviorTreeDebugger(code_generator)
        mock_vehicle_state = Mock()
        
        debug_steps = debugger.start_debug_session(complete_tree_data, mock_vehicle_state)
        assert len(debug_steps) == 4  # Four blocks
        
        # Step through execution
        step1 = debugger.step_forward()
        step2 = debugger.step_forward()
        
        assert step1 is not None
        assert step2 is not None
        assert step1 != step2
        
        # Get execution summary
        summary = debugger.get_execution_summary()
        assert summary['total_steps'] == 4
        assert summary['current_step'] == 2
    
    def test_real_time_validation_workflow(self, complete_tree_data):
        """Test real-time validation workflow"""
        validator = RealTimeBehaviorValidator()
        
        # Set up error callback
        error_callback = Mock()
        validator.add_error_callback(error_callback)
        
        # Validate valid tree
        result = validator.validate_real_time(complete_tree_data)
        assert result.is_valid == True
        error_callback.assert_not_called()
        
        # Validate invalid tree
        invalid_tree = {'blocks': []}
        result = validator.validate_real_time(invalid_tree)
        assert result.is_valid == False
        error_callback.assert_called_once()
        
        # Check runtime errors
        runtime_errors = validator.check_runtime_errors(complete_tree_data)
        # Should be no runtime errors for this valid tree
        assert len(runtime_errors) == 0
    
    def test_parameter_validation_edge_cases(self):
        """Test parameter validation with edge cases"""
        code_generator = CodeGenerator()
        
        # Test edge case parameters
        edge_case_tree = {
            'blocks': [
                {
                    'id': 'speed-check',
                    'type': 'speed_check',
                    'data': {
                        'parameters': {
                            'min_speed': 0.0,  # Edge case: zero minimum
                            'max_speed': 0.0   # Edge case: same min and max
                        }
                    }
                }
            ]
        }
        
        result = code_generator.validate_behavior_tree(edge_case_tree)
        # This should fail because max_speed must be greater than min_speed
        assert result.is_valid == False
        assert any("must be greater than" in error for error in result.errors)


if __name__ == '__main__':
    pytest.main([__file__])