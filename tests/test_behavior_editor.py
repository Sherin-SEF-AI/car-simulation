"""
Unit tests for the visual programming interface behavior editor.
Tests the BehaviorEditor widget, BlockLibrary, and related components.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QPointF, QRectF, QMimeData
from PyQt6.QtGui import QDrag
from PyQt6.QtTest import QTest

from src.ui.behavior_editor import (
    BehaviorEditor, BehaviorBlock, BlockLibrary, BehaviorCanvas, 
    ConnectionLine
)


class TestBehaviorBlock:
    """Test cases for BehaviorBlock class"""
    
    @pytest.fixture
    def sample_block_data(self):
        return {
            'type': 'accelerate',
            'name': 'Accelerate',
            'category': 'action',
            'description': 'Increase vehicle speed',
            'parameters': {'acceleration': 'float', 'max_speed': 'float'},
            'has_input': True,
            'has_output': True
        }
    
    @pytest.fixture
    def behavior_block(self, sample_block_data):
        return BehaviorBlock('accelerate', sample_block_data)
    
    def test_behavior_block_initialization(self, behavior_block, sample_block_data):
        """Test BehaviorBlock initialization"""
        assert behavior_block.block_type == 'accelerate'
        assert behavior_block.block_data == sample_block_data
        assert behavior_block.block_id is not None
        assert len(behavior_block.block_id) > 0
        assert behavior_block.connections == []
        assert behavior_block.input_connections == []
        assert behavior_block.output_connections == []
    
    def test_behavior_block_bounding_rect(self, behavior_block):
        """Test BehaviorBlock bounding rectangle"""
        rect = behavior_block.boundingRect()
        assert isinstance(rect, QRectF)
        assert rect.width() == behavior_block.width
        assert rect.height() == behavior_block.height
    
    def test_behavior_block_connection_points(self, behavior_block):
        """Test BehaviorBlock connection point calculations"""
        # Set position for testing
        behavior_block.setPos(100, 100)
        
        input_point = behavior_block.get_input_point()
        output_point = behavior_block.get_output_point()
        
        assert isinstance(input_point, QPointF)
        assert isinstance(output_point, QPointF)
        
        # Output point should be to the right of input point
        assert output_point.x() > input_point.x()
    
    def test_behavior_block_type_colors(self, behavior_block):
        """Test that different block types have different colors"""
        # Test action block color
        assert 'action' in behavior_block.type_colors
        
        # Test other categories
        condition_data = {'category': 'condition'}
        condition_block = BehaviorBlock('test_condition', condition_data)
        
        sensor_data = {'category': 'sensor'}
        sensor_block = BehaviorBlock('test_sensor', sensor_data)
        
        # Colors should be different
        action_color = behavior_block.type_colors['action']
        condition_color = condition_block.type_colors['condition']
        sensor_color = sensor_block.type_colors['sensor']
        
        assert action_color != condition_color
        assert condition_color != sensor_color
        assert action_color != sensor_color
    
    def test_behavior_block_flags(self, behavior_block):
        """Test that BehaviorBlock has correct item flags"""
        flags = behavior_block.flags()
        
        # Should be movable and selectable
        assert flags & behavior_block.GraphicsItemFlag.ItemIsMovable
        assert flags & behavior_block.GraphicsItemFlag.ItemIsSelectable
        assert flags & behavior_block.GraphicsItemFlag.ItemSendsGeometryChanges


class TestConnectionLine:
    """Test cases for ConnectionLine class"""
    
    @pytest.fixture
    def start_block(self):
        data = {'type': 'start', 'category': 'action', 'has_output': True}
        block = BehaviorBlock('start', data)
        block.setPos(0, 0)
        return block
    
    @pytest.fixture
    def end_block(self):
        data = {'type': 'end', 'category': 'action', 'has_input': True}
        block = BehaviorBlock('end', data)
        block.setPos(200, 100)
        return block
    
    @pytest.fixture
    def connection_line(self, start_block, end_block):
        return ConnectionLine(start_block, end_block)
    
    def test_connection_line_initialization(self, connection_line, start_block, end_block):
        """Test ConnectionLine initialization"""
        assert connection_line.start_block == start_block
        assert connection_line.end_block == end_block
        assert connection_line.pen is not None
        assert connection_line.selected_pen is not None
    
    def test_connection_line_bounding_rect(self, connection_line):
        """Test ConnectionLine bounding rectangle calculation"""
        rect = connection_line.boundingRect()
        assert isinstance(rect, QRectF)
        assert rect.width() > 0
        assert rect.height() >= 0  # Could be 0 for horizontal lines


class TestBlockLibrary:
    """Test cases for BlockLibrary class"""
    
    @pytest.fixture
    def block_library(self):
        # Create QApplication if it doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        library = BlockLibrary()
        return library
    
    def test_block_library_initialization(self, block_library):
        """Test BlockLibrary initialization"""
        assert block_library.block_definitions is not None
        assert len(block_library.block_definitions) > 0
        
        # Check that all expected categories exist
        expected_categories = ['Sensors', 'Conditions', 'Actions', 'Control']
        for category in expected_categories:
            assert category in block_library.block_definitions
    
    def test_block_library_sensor_blocks(self, block_library):
        """Test that sensor blocks are properly defined"""
        sensors = block_library.block_definitions['Sensors']
        assert len(sensors) > 0
        
        # Check for specific sensor types
        sensor_types = [block['type'] for block in sensors]
        assert 'camera_sensor' in sensor_types
        assert 'lidar_sensor' in sensor_types
        assert 'gps_sensor' in sensor_types
        
        # Verify sensor blocks have correct properties
        for sensor in sensors:
            assert sensor['category'] == 'sensor'
            assert sensor['has_input'] == False  # Sensors don't have inputs
            assert sensor['has_output'] == True   # Sensors have outputs
    
    def test_block_library_action_blocks(self, block_library):
        """Test that action blocks are properly defined"""
        actions = block_library.block_definitions['Actions']
        assert len(actions) > 0
        
        # Check for specific action types
        action_types = [block['type'] for block in actions]
        assert 'accelerate' in action_types
        assert 'brake' in action_types
        assert 'steer' in action_types
        
        # Verify action blocks have correct properties
        for action in actions:
            assert action['category'] == 'action'
            assert 'parameters' in action
    
    def test_block_library_condition_blocks(self, block_library):
        """Test that condition blocks are properly defined"""
        conditions = block_library.block_definitions['Conditions']
        assert len(conditions) > 0
        
        # Check for specific condition types
        condition_types = [block['type'] for block in conditions]
        assert 'obstacle_detected' in condition_types
        assert 'speed_check' in condition_types
        
        # Verify condition blocks have correct properties
        for condition in conditions:
            assert condition['category'] == 'condition'
    
    def test_block_library_control_blocks(self, block_library):
        """Test that control flow blocks are properly defined"""
        control_blocks = block_library.block_definitions['Control']
        assert len(control_blocks) > 0
        
        # Check for specific control types
        control_types = [block['type'] for block in control_blocks]
        assert 'sequence' in control_types
        assert 'selector' in control_types
        assert 'parallel' in control_types
        
        # Verify control blocks have multiple children capability
        for control in control_blocks:
            assert control['category'] == 'composite'
            if control['type'] in ['sequence', 'selector', 'parallel']:
                assert control.get('multiple_children', False) == True
    
    def test_get_category_color(self, block_library):
        """Test category color mapping"""
        # Test all defined categories have colors
        colors = {
            'sensor': block_library.get_category_color('sensor'),
            'condition': block_library.get_category_color('condition'),
            'action': block_library.get_category_color('action'),
            'composite': block_library.get_category_color('composite'),
            'control': block_library.get_category_color('control')
        }
        
        # All colors should be valid hex colors
        for category, color in colors.items():
            assert color.startswith('#')
            assert len(color) == 7  # #RRGGBB format
        
        # Colors should be different
        unique_colors = set(colors.values())
        assert len(unique_colors) >= 4  # At least 4 different colors
    
    def test_create_block_widget(self, block_library):
        """Test block widget creation"""
        sample_block = {
            'type': 'test_block',
            'name': 'Test\nBlock',
            'category': 'action',
            'description': 'A test block for unit testing',
            'parameters': {'param1': 'float'},
            'has_input': True,
            'has_output': True
        }
        
        widget = block_library.create_block_widget(sample_block)
        assert widget is not None
        assert hasattr(widget, 'block_def')
        assert widget.block_def == sample_block


class TestBehaviorCanvas:
    """Test cases for BehaviorCanvas class"""
    
    @pytest.fixture
    def behavior_canvas(self):
        # Create QApplication if it doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        canvas = BehaviorCanvas()
        return canvas
    
    def test_behavior_canvas_initialization(self, behavior_canvas):
        """Test BehaviorCanvas initialization"""
        assert behavior_canvas.scene is not None
        assert behavior_canvas.acceptDrops() == True
        assert behavior_canvas.connecting_mode == False
        assert behavior_canvas.connection_start_block is None
    
    def test_behavior_canvas_drag_drop(self, behavior_canvas, qtbot):
        """Test drag and drop functionality"""
        # Create mock drop event
        block_def = {
            'type': 'test_block',
            'name': 'Test Block',
            'category': 'action',
            'description': 'Test block',
            'parameters': {},
            'has_input': True,
            'has_output': True
        }
        
        # Simulate drop event
        mime_data = QMimeData()
        mime_data.setText(json.dumps(block_def))
        
        # Mock the drop event
        with patch.object(behavior_canvas, 'mapToScene') as mock_map:
            mock_map.return_value = QPointF(100, 100)
            
            # Create a mock drop event
            mock_event = Mock()
            mock_event.mimeData.return_value = mime_data
            mock_event.position.return_value.toPoint.return_value = Mock()
            
            # Test drop handling
            behavior_canvas.dropEvent(mock_event)
            
            # Verify block was added to scene
            items = behavior_canvas.scene.items()
            behavior_blocks = [item for item in items if isinstance(item, BehaviorBlock)]
            assert len(behavior_blocks) == 1
            assert behavior_blocks[0].block_type == 'test_block'
    
    def test_get_behavior_tree_data(self, behavior_canvas):
        """Test behavior tree data extraction"""
        # Add some test blocks
        block1_data = {'type': 'block1', 'category': 'action'}
        block1 = BehaviorBlock('block1', block1_data)
        block1.setPos(0, 0)
        behavior_canvas.scene.addItem(block1)
        
        block2_data = {'type': 'block2', 'category': 'condition'}
        block2 = BehaviorBlock('block2', block2_data)
        block2.setPos(100, 100)
        behavior_canvas.scene.addItem(block2)
        
        # Get tree data
        tree_data = behavior_canvas.get_behavior_tree_data()
        
        assert 'blocks' in tree_data
        assert 'connections' in tree_data
        assert len(tree_data['blocks']) == 2
        
        # Verify block data structure
        for block_data in tree_data['blocks']:
            assert 'id' in block_data
            assert 'type' in block_data
            assert 'position' in block_data
            assert 'data' in block_data
            assert 'connections' in block_data


class TestBehaviorEditor:
    """Test cases for BehaviorEditor main widget"""
    
    @pytest.fixture
    def behavior_editor(self):
        # Create QApplication if it doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        editor = BehaviorEditor()
        return editor
    
    def test_behavior_editor_initialization(self, behavior_editor):
        """Test BehaviorEditor initialization"""
        assert behavior_editor.block_library is not None
        assert behavior_editor.canvas is not None
        assert isinstance(behavior_editor.block_library, BlockLibrary)
        assert isinstance(behavior_editor.canvas, BehaviorCanvas)
    
    def test_create_behavior_block(self, behavior_editor):
        """Test behavior block creation"""
        # Test creating a known block type
        block = behavior_editor.create_behavior_block('accelerate')
        assert isinstance(block, BehaviorBlock)
        assert block.block_type == 'accelerate'
        assert block.block_data['category'] == 'action'
        
        # Test creating an unknown block type (should create default)
        unknown_block = behavior_editor.create_behavior_block('unknown_type')
        assert isinstance(unknown_block, BehaviorBlock)
        assert unknown_block.block_type == 'unknown_type'
        assert 'Custom unknown_type block' in unknown_block.block_data['description']
    
    def test_connect_blocks(self, behavior_editor):
        """Test connecting two behavior blocks"""
        # Create two blocks
        source_block = behavior_editor.create_behavior_block('accelerate')
        target_block = behavior_editor.create_behavior_block('brake')
        
        # Add blocks to canvas
        behavior_editor.canvas.scene.addItem(source_block)
        behavior_editor.canvas.scene.addItem(target_block)
        
        # Connect blocks
        behavior_editor.connect_blocks(source_block, target_block)
        
        # Verify connection
        assert target_block in source_block.output_connections
        assert source_block in target_block.input_connections
        
        # Verify connection line was created
        connection_lines = [item for item in behavior_editor.canvas.scene.items() 
                          if isinstance(item, ConnectionLine)]
        assert len(connection_lines) == 1
    
    def test_clear_canvas(self, behavior_editor):
        """Test clearing the canvas"""
        # Add some blocks
        block1 = behavior_editor.create_behavior_block('accelerate')
        block2 = behavior_editor.create_behavior_block('brake')
        behavior_editor.canvas.scene.addItem(block1)
        behavior_editor.canvas.scene.addItem(block2)
        
        # Verify blocks are present
        assert len(behavior_editor.canvas.scene.items()) == 2
        
        # Clear canvas
        behavior_editor.clear_canvas()
        
        # Verify canvas is empty
        assert len(behavior_editor.canvas.scene.items()) == 0
    
    def test_get_behavior_tree_json(self, behavior_editor):
        """Test JSON serialization of behavior tree"""
        # Add a block
        block = behavior_editor.create_behavior_block('accelerate')
        behavior_editor.canvas.scene.addItem(block)
        
        # Get JSON
        json_str = behavior_editor.get_behavior_tree_json()
        
        # Verify it's valid JSON
        tree_data = json.loads(json_str)
        assert 'blocks' in tree_data
        assert 'connections' in tree_data
        assert len(tree_data['blocks']) == 1
    
    def test_load_behavior_tree_json(self, behavior_editor):
        """Test JSON deserialization of behavior tree"""
        # Create test JSON data
        test_data = {
            'blocks': [
                {
                    'id': 'test-id-1',
                    'type': 'accelerate',
                    'position': {'x': 100, 'y': 100},
                    'data': {
                        'type': 'accelerate',
                        'name': 'Accelerate',
                        'category': 'action',
                        'description': 'Increase vehicle speed',
                        'parameters': {'acceleration': 'float'},
                        'has_input': True,
                        'has_output': True
                    },
                    'connections': []
                }
            ],
            'connections': []
        }
        
        json_str = json.dumps(test_data)
        
        # Load JSON
        behavior_editor.load_behavior_tree_json(json_str)
        
        # Verify block was loaded
        items = behavior_editor.canvas.scene.items()
        behavior_blocks = [item for item in items if isinstance(item, BehaviorBlock)]
        assert len(behavior_blocks) == 1
        assert behavior_blocks[0].block_type == 'accelerate'
        assert behavior_blocks[0].block_id == 'test-id-1'
    
    def test_behavior_changed_signal(self, behavior_editor):
        """Test that behavior_changed signal is emitted correctly"""
        # Connect signal to a mock
        signal_mock = Mock()
        behavior_editor.behavior_changed.connect(signal_mock)
        
        # Add a block (should trigger signal)
        block = behavior_editor.create_behavior_block('accelerate')
        behavior_editor.canvas.scene.addItem(block)
        behavior_editor.on_block_added(block)
        
        # Verify signal was emitted
        signal_mock.assert_called_once()
        
        # Verify signal contains tree data
        call_args = signal_mock.call_args[0]
        assert len(call_args) == 1
        tree_data = call_args[0]
        assert 'blocks' in tree_data
        assert 'connections' in tree_data
    
    def test_validate_behavior_tree(self, behavior_editor):
        """Test behavior tree validation"""
        # Add some blocks
        block1 = behavior_editor.create_behavior_block('camera_sensor')
        block2 = behavior_editor.create_behavior_block('obstacle_detected')
        block3 = behavior_editor.create_behavior_block('brake')
        
        behavior_editor.canvas.scene.addItem(block1)
        behavior_editor.canvas.scene.addItem(block2)
        behavior_editor.canvas.scene.addItem(block3)
        
        # Connect signal to mock
        signal_mock = Mock()
        behavior_editor.behavior_changed.connect(signal_mock)
        
        # Validate tree
        behavior_editor.validate_behavior_tree()
        
        # Verify validation triggered behavior_changed signal
        signal_mock.assert_called_once()


class TestBehaviorEditorIntegration:
    """Integration tests for the complete behavior editor system"""
    
    @pytest.fixture
    def app(self):
        """Create QApplication for integration tests"""
        return QApplication.instance() or QApplication([])
    
    @pytest.fixture
    def behavior_editor(self, app):
        editor = BehaviorEditor()
        editor.show()
        return editor
    
    def test_complete_workflow(self, behavior_editor):
        """Test complete workflow from block creation to tree generation"""
        # Step 1: Create blocks
        sensor_block = behavior_editor.create_behavior_block('camera_sensor')
        condition_block = behavior_editor.create_behavior_block('obstacle_detected')
        action_block = behavior_editor.create_behavior_block('brake')
        
        # Step 2: Add blocks to canvas
        behavior_editor.canvas.scene.addItem(sensor_block)
        behavior_editor.canvas.scene.addItem(condition_block)
        behavior_editor.canvas.scene.addItem(action_block)
        
        # Position blocks
        sensor_block.setPos(0, 100)
        condition_block.setPos(200, 100)
        action_block.setPos(400, 100)
        
        # Step 3: Connect blocks
        behavior_editor.connect_blocks(sensor_block, condition_block)
        behavior_editor.connect_blocks(condition_block, action_block)
        
        # Step 4: Validate tree structure
        tree_data = behavior_editor.canvas.get_behavior_tree_data()
        
        assert len(tree_data['blocks']) == 3
        
        # Find blocks by type
        blocks_by_type = {block['type']: block for block in tree_data['blocks']}
        
        assert 'camera_sensor' in blocks_by_type
        assert 'obstacle_detected' in blocks_by_type
        assert 'brake' in blocks_by_type
        
        # Verify connections
        condition_connections = blocks_by_type['obstacle_detected']['connections']
        sensor_connections = blocks_by_type['camera_sensor']['connections']
        
        # Camera sensor should connect to obstacle detection
        assert len(sensor_connections) == 1
        # Obstacle detection should connect to brake
        assert len(condition_connections) == 1
        
        # Step 5: Test JSON serialization/deserialization
        json_str = behavior_editor.get_behavior_tree_json()
        assert json_str is not None
        assert len(json_str) > 0
        
        # Clear and reload
        behavior_editor.clear_canvas()
        assert len(behavior_editor.canvas.scene.items()) == 0
        
        behavior_editor.load_behavior_tree_json(json_str)
        
        # Verify reload worked
        reloaded_items = behavior_editor.canvas.scene.items()
        reloaded_blocks = [item for item in reloaded_items if isinstance(item, BehaviorBlock)]
        assert len(reloaded_blocks) == 3
    
    def test_drag_drop_simulation(self, behavior_editor):
        """Test simulated drag and drop operation"""
        # Get a block definition from the library
        sensor_blocks = behavior_editor.block_library.block_definitions['Sensors']
        camera_block_def = next(block for block in sensor_blocks 
                               if block['type'] == 'camera_sensor')
        
        # Simulate drag and drop
        mime_data = QMimeData()
        mime_data.setText(json.dumps(camera_block_def))
        
        # Mock drop event
        with patch.object(behavior_editor.canvas, 'mapToScene') as mock_map:
            mock_map.return_value = QPointF(150, 150)
            
            mock_event = Mock()
            mock_event.mimeData.return_value = mime_data
            mock_event.position.return_value.toPoint.return_value = Mock()
            mock_event.acceptProposedAction = Mock()
            
            # Perform drop
            behavior_editor.canvas.dropEvent(mock_event)
            
            # Verify block was created
            items = behavior_editor.canvas.scene.items()
            camera_blocks = [item for item in items 
                           if isinstance(item, BehaviorBlock) and item.block_type == 'camera_sensor']
            assert len(camera_blocks) == 1
            
            # Verify position
            camera_block = camera_blocks[0]
            # Position should be close to drop point (exact match depends on implementation)
            assert abs(camera_block.pos().x() - 150) < 50
            assert abs(camera_block.pos().y() - 150) < 50


if __name__ == '__main__':
    pytest.main([__file__])