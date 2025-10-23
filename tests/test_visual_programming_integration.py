"""
Integration tests for the complete visual programming interface.
Tests the integration between BehaviorEditor and CodeGenerator systems.
"""

import pytest
import json
from unittest.mock import Mock, patch
from PyQt6.QtWidgets import QApplication

from src.ui.behavior_editor import BehaviorEditor, BehaviorBlock, BlockLibrary
from src.ui.code_generator import CodeGenerator, ValidationResult


class TestVisualProgrammingIntegration:
    """Integration tests for the complete visual programming system"""
    
    @pytest.fixture
    def app(self):
        """Create QApplication for integration tests"""
        return QApplication.instance() or QApplication([])
    
    @pytest.fixture
    def behavior_editor(self, app):
        editor = BehaviorEditor()
        return editor
    
    def test_complete_visual_programming_workflow(self, behavior_editor):
        """Test complete workflow from visual programming to code generation"""
        
        # Step 1: Create blocks using the visual interface
        camera_block = behavior_editor.create_behavior_block('camera_sensor')
        obstacle_block = behavior_editor.create_behavior_block('obstacle_detected')
        brake_block = behavior_editor.create_behavior_block('brake')
        
        # Step 2: Add blocks to canvas
        behavior_editor.canvas.scene.addItem(camera_block)
        behavior_editor.canvas.scene.addItem(obstacle_block)
        behavior_editor.canvas.scene.addItem(brake_block)
        
        # Position blocks
        camera_block.setPos(0, 100)
        obstacle_block.setPos(200, 100)
        brake_block.setPos(400, 100)
        
        # Step 3: Connect blocks
        behavior_editor.connect_blocks(camera_block, obstacle_block)
        behavior_editor.connect_blocks(obstacle_block, brake_block)
        
        # Step 4: Get behavior tree data
        tree_data = behavior_editor.canvas.get_behavior_tree_data()
        
        assert len(tree_data['blocks']) == 3
        
        # Step 5: Validate the tree
        validation_result = behavior_editor.validator.validate_real_time(tree_data)
        assert validation_result.is_valid == True
        
        # Step 6: Generate code
        try:
            generated_code = behavior_editor.code_generator.generate_code(tree_data)
            assert isinstance(generated_code, str)
            assert len(generated_code) > 0
            
            # Verify code contains expected functions
            assert "class BehaviorTreeExecutor" in generated_code
            assert "camera_sensor" in generated_code
            assert "obstacle_detected" in generated_code
            assert "brake" in generated_code
            
        except ValueError as e:
            pytest.fail(f"Code generation failed: {e}")
        
        # Step 7: Test debugging
        debug_steps = behavior_editor.debugger.start_debug_session(tree_data, Mock())
        assert len(debug_steps) == 3
        
        # Step through execution
        step1 = behavior_editor.debugger.step_forward()
        step2 = behavior_editor.debugger.step_forward()
        
        assert step1 is not None
        assert step2 is not None
        assert step1.node_id != step2.node_id
    
    def test_block_library_integration(self, behavior_editor):
        """Test integration between block library and behavior editor"""
        
        # Test that all block types from library can be created
        for category, blocks in behavior_editor.block_library.block_definitions.items():
            for block_def in blocks:
                block = behavior_editor.create_behavior_block(block_def['type'])
                
                assert isinstance(block, BehaviorBlock)
                assert block.block_type == block_def['type']
                assert block.block_data['category'] == block_def['category']
    
    def test_real_time_validation_integration(self, behavior_editor):
        """Test real-time validation integration"""
        
        # Create a valid tree
        sensor_block = behavior_editor.create_behavior_block('camera_sensor')
        action_block = behavior_editor.create_behavior_block('accelerate')
        
        behavior_editor.canvas.scene.addItem(sensor_block)
        behavior_editor.canvas.scene.addItem(action_block)
        
        # Connect blocks
        behavior_editor.connect_blocks(sensor_block, action_block)
        
        # Test validation
        tree_data = behavior_editor.canvas.get_behavior_tree_data()
        result = behavior_editor.validator.validate_real_time(tree_data)
        
        assert result.is_valid == True
        
        # Test that validation is cached
        result2 = behavior_editor.validator.validate_real_time(tree_data)
        assert result == result2
        assert len(behavior_editor.validator.validation_cache) == 1
    
    def test_error_handling_integration(self, behavior_editor):
        """Test error handling across the integrated system"""
        
        # Test validation error callback
        error_callback_called = False
        
        def mock_callback(validation_result):
            nonlocal error_callback_called
            error_callback_called = True
            behavior_editor.on_validation_error(validation_result)
        
        # Clear existing callbacks and add our mock
        behavior_editor.validator.error_callbacks.clear()
        behavior_editor.validator.add_error_callback(mock_callback)
        
        # Create invalid tree (empty)
        empty_tree = {'blocks': []}
        result = behavior_editor.validator.validate_real_time(empty_tree)
        
        assert result.is_valid == False
        assert error_callback_called == True
    
    def test_code_generation_with_different_block_types(self, behavior_editor):
        """Test code generation with various block types"""
        
        # Create a complex tree with different block types
        blocks_to_test = [
            ('camera_sensor', {'resolution': '1920x1080', 'fps': 30}),
            ('lidar_sensor', {'range': 100.0, 'resolution': 0.1}),
            ('obstacle_detected', {'distance_threshold': 5.0, 'angle_range': 30.0}),
            ('speed_check', {'min_speed': 0.0, 'max_speed': 50.0}),
            ('accelerate', {'acceleration': 2.0, 'max_speed': 60.0}),
            ('brake', {'brake_force': 0.8}),
            ('steer', {'steering_angle': 0.5, 'duration': 2.0}),
            ('sequence', {}),
            ('selector', {}),
            ('parallel', {'success_count': 2})
        ]
        
        created_blocks = []
        for block_type, params in blocks_to_test:
            block = behavior_editor.create_behavior_block(block_type)
            
            # Update parameters
            if params:
                block.block_data['parameters'] = params
            
            behavior_editor.canvas.scene.addItem(block)
            created_blocks.append(block)
        
        # Create a simple chain connection
        for i in range(len(created_blocks) - 1):
            behavior_editor.connect_blocks(created_blocks[i], created_blocks[i + 1])
        
        # Get tree data and generate code
        tree_data = behavior_editor.canvas.get_behavior_tree_data()
        
        try:
            generated_code = behavior_editor.code_generator.generate_code(tree_data)
            
            # Verify all block types are represented in generated code
            for block_type, _ in blocks_to_test:
                assert block_type in generated_code
            
            # Verify code is syntactically valid
            import ast
            ast.parse(generated_code)
            
        except Exception as e:
            pytest.fail(f"Code generation failed for complex tree: {e}")
    
    def test_debugging_integration(self, behavior_editor):
        """Test debugging integration with visual programming"""
        
        # Create a simple behavior tree
        sensor_block = behavior_editor.create_behavior_block('camera_sensor')
        condition_block = behavior_editor.create_behavior_block('obstacle_detected')
        action_block = behavior_editor.create_behavior_block('brake')
        
        behavior_editor.canvas.scene.addItem(sensor_block)
        behavior_editor.canvas.scene.addItem(condition_block)
        behavior_editor.canvas.scene.addItem(action_block)
        
        # Connect blocks
        behavior_editor.connect_blocks(sensor_block, condition_block)
        behavior_editor.connect_blocks(condition_block, action_block)
        
        # Start debug session
        tree_data = behavior_editor.canvas.get_behavior_tree_data()
        mock_vehicle_state = Mock()
        
        debug_steps = behavior_editor.debugger.start_debug_session(tree_data, mock_vehicle_state)
        
        assert len(debug_steps) == 3
        assert behavior_editor.debugger.current_step == 0
        
        # Test stepping through execution
        step1 = behavior_editor.step_forward()
        assert step1 is not None
        assert behavior_editor.debugger.current_step == 1
        
        step2 = behavior_editor.step_forward()
        assert step2 is not None
        assert behavior_editor.debugger.current_step == 2
        
        # Test stepping backward
        prev_step = behavior_editor.step_backward()
        assert prev_step == step1
        assert behavior_editor.debugger.current_step == 1
        
        # Test execution summary
        summary = behavior_editor.debugger.get_execution_summary()
        assert summary['total_steps'] == 3
        assert summary['current_step'] == 1
    
    def test_json_serialization_integration(self, behavior_editor):
        """Test JSON serialization/deserialization integration"""
        
        # Create a behavior tree
        blocks = [
            behavior_editor.create_behavior_block('camera_sensor'),
            behavior_editor.create_behavior_block('obstacle_detected'),
            behavior_editor.create_behavior_block('brake')
        ]
        
        for i, block in enumerate(blocks):
            block.setPos(i * 200, 100)
            behavior_editor.canvas.scene.addItem(block)
        
        # Connect blocks
        for i in range(len(blocks) - 1):
            behavior_editor.connect_blocks(blocks[i], blocks[i + 1])
        
        # Serialize to JSON
        json_str = behavior_editor.get_behavior_tree_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Verify JSON is valid
        tree_data = json.loads(json_str)
        assert 'blocks' in tree_data
        assert 'connections' in tree_data
        assert len(tree_data['blocks']) == 3
        
        # Clear canvas and reload
        behavior_editor.clear_canvas()
        assert len(behavior_editor.canvas.scene.items()) == 0
        
        # Deserialize from JSON
        behavior_editor.load_behavior_tree_json(json_str)
        
        # Verify reload worked
        reloaded_items = behavior_editor.canvas.scene.items()
        reloaded_blocks = [item for item in reloaded_items if isinstance(item, BehaviorBlock)]
        assert len(reloaded_blocks) == 3
        
        # Verify the reloaded tree can still be validated and generate code
        reloaded_tree_data = behavior_editor.canvas.get_behavior_tree_data()
        validation_result = behavior_editor.validator.validate_real_time(reloaded_tree_data)
        assert validation_result.is_valid == True
        
        generated_code = behavior_editor.code_generator.generate_code(reloaded_tree_data)
        assert isinstance(generated_code, str)
        assert len(generated_code) > 0
    
    def test_parameter_validation_integration(self, behavior_editor):
        """Test parameter validation integration"""
        
        # Create blocks with various parameter types
        test_cases = [
            ('camera_sensor', {'resolution': '1920x1080', 'fps': 30}),  # Valid
            ('lidar_sensor', {'range': -10.0, 'resolution': 0.1}),     # Invalid: negative range
            ('speed_check', {'min_speed': 10.0, 'max_speed': 5.0}),    # Invalid: max < min
            ('brake', {'brake_force': 'invalid'}),                      # Invalid: string instead of number
        ]
        
        for block_type, params in test_cases:
            block = behavior_editor.create_behavior_block(block_type)
            block.block_data['parameters'] = params
            behavior_editor.canvas.scene.addItem(block)
        
        # Get tree data and validate
        tree_data = behavior_editor.canvas.get_behavior_tree_data()
        validation_result = behavior_editor.validator.validate_real_time(tree_data)
        
        # Should have validation errors due to invalid parameters
        assert validation_result.is_valid == False
        assert len(validation_result.errors) > 0
        
        # Check runtime errors
        runtime_errors = behavior_editor.validator.check_runtime_errors(tree_data)
        assert len(runtime_errors) > 0
    
    def test_performance_with_large_trees(self, behavior_editor):
        """Test performance with larger behavior trees"""
        
        # Create a larger behavior tree (50 blocks)
        blocks = []
        for i in range(50):
            block_type = ['camera_sensor', 'obstacle_detected', 'brake'][i % 3]
            block = behavior_editor.create_behavior_block(block_type)
            block.setPos(i * 50, (i % 10) * 50)
            behavior_editor.canvas.scene.addItem(block)
            blocks.append(block)
        
        # Connect blocks in a chain
        for i in range(len(blocks) - 1):
            behavior_editor.connect_blocks(blocks[i], blocks[i + 1])
        
        # Test that operations still work with larger trees
        tree_data = behavior_editor.canvas.get_behavior_tree_data()
        assert len(tree_data['blocks']) == 50
        
        # Validation should still work (though may have errors due to structure)
        validation_result = behavior_editor.validator.validate_real_time(tree_data)
        assert isinstance(validation_result, ValidationResult)
        
        # JSON serialization should work
        json_str = behavior_editor.get_behavior_tree_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Deserialization should work
        behavior_editor.clear_canvas()
        behavior_editor.load_behavior_tree_json(json_str)
        
        reloaded_items = behavior_editor.canvas.scene.items()
        reloaded_blocks = [item for item in reloaded_items if isinstance(item, BehaviorBlock)]
        assert len(reloaded_blocks) == 50


if __name__ == '__main__':
    pytest.main([__file__])