"""
Visual Programming Interface - Behavior Editor
Provides a Scratch-like visual programming interface for creating autonomous vehicle behaviors.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, 
                            QScrollArea, QGraphicsView, QGraphicsScene, 
                            QGraphicsItem, QGraphicsProxyWidget, QPushButton,
                            QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                            QLineEdit, QGroupBox, QTreeWidget, QTreeWidgetItem,
                            QTabWidget, QTextEdit, QFrame)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QMimeData, QTimer
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QDrag, QPalette
from typing import Dict, List, Any, Optional, Tuple
import json
import uuid
from .code_generator import CodeGenerator, BehaviorTreeDebugger, RealTimeBehaviorValidator

class BehaviorBlock(QGraphicsItem):
    """Individual behavior block that can be connected to other blocks"""
    
    def __init__(self, block_type: str, block_data: Dict[str, Any]):
        super().__init__()
        self.block_type = block_type
        self.block_data = block_data
        self.block_id = str(uuid.uuid4())
        self.connections = []  # List of connected blocks
        self.input_connections = []  # Blocks connected to this block's input
        self.output_connections = []  # Blocks this block connects to
        
        # Visual properties
        self.width = 120
        self.height = 60
        self.corner_radius = 8
        self.selected_pen = QPen(QColor(255, 255, 255), 3)
        self.normal_pen = QPen(QColor(100, 100, 100), 2)
        
        # Block type colors
        self.type_colors = {
            'condition': QColor(100, 150, 255),  # Blue
            'action': QColor(100, 255, 150),     # Green
            'composite': QColor(255, 150, 100),  # Orange
            'sensor': QColor(255, 255, 100),     # Yellow
            'control': QColor(255, 100, 255),    # Magenta
        }
        
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        
    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)
    
    def paint(self, painter: QPainter, option, widget):
        # Get block color based on type
        color = self.type_colors.get(self.block_data.get('category', 'action'), 
                                   QColor(150, 150, 150))
        
        # Draw block background
        if self.isSelected():
            painter.setPen(self.selected_pen)
        else:
            painter.setPen(self.normal_pen)
            
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(self.boundingRect(), self.corner_radius, self.corner_radius)
        
        # Draw block text
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        text_rect = self.boundingRect().adjusted(5, 5, -5, -5)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, 
                        self.block_data.get('name', self.block_type))
        
        # Draw connection points
        self.draw_connection_points(painter)
    
    def draw_connection_points(self, painter: QPainter):
        """Draw input and output connection points"""
        point_radius = 6
        
        # Input connection point (left side)
        if self.block_data.get('has_input', True):
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawEllipse(QPointF(-point_radius, self.height/2), 
                              point_radius, point_radius)
        
        # Output connection point (right side)
        if self.block_data.get('has_output', True):
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawEllipse(QPointF(self.width, self.height/2), 
                              point_radius, point_radius)
    
    def get_input_point(self) -> QPointF:
        """Get the position of the input connection point"""
        return self.mapToScene(QPointF(0, self.height/2))
    
    def get_output_point(self) -> QPointF:
        """Get the position of the output connection point"""
        return self.mapToScene(QPointF(self.width, self.height/2))
    
    def itemChange(self, change, value):
        """Handle item changes, particularly position changes"""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Update connections when block moves
            scene = self.scene()
            if scene and hasattr(scene, 'update_connections'):
                scene.update_connections(self)
        return super().itemChange(change, value)


class ConnectionLine(QGraphicsItem):
    """Visual connection line between behavior blocks"""
    
    def __init__(self, start_block: BehaviorBlock, end_block: BehaviorBlock):
        super().__init__()
        self.start_block = start_block
        self.end_block = end_block
        self.pen = QPen(QColor(100, 100, 100), 3)
        self.selected_pen = QPen(QColor(255, 255, 255), 4)
        
    def boundingRect(self) -> QRectF:
        start_point = self.start_block.get_output_point()
        end_point = self.end_block.get_input_point()
        
        return QRectF(start_point, end_point).normalized().adjusted(-5, -5, 5, 5)
    
    def paint(self, painter: QPainter, option, widget):
        start_point = self.mapFromScene(self.start_block.get_output_point())
        end_point = self.mapFromScene(self.end_block.get_input_point())
        
        if self.isSelected():
            painter.setPen(self.selected_pen)
        else:
            painter.setPen(self.pen)
            
        # Draw curved connection line
        control_offset = abs(end_point.x() - start_point.x()) * 0.5
        control1 = QPointF(start_point.x() + control_offset, start_point.y())
        control2 = QPointF(end_point.x() - control_offset, end_point.y())
        
        path = painter.path()
        path.moveTo(start_point)
        path.cubicTo(control1, control2, end_point)
        painter.drawPath(path)


class BlockLibrary(QWidget):
    """Library of available behavior blocks for drag-and-drop"""
    
    block_selected = pyqtSignal(str, dict)  # block_type, block_data
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_block_definitions()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Block Library")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Category tabs
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Search functionality could be added here
        
    def load_block_definitions(self):
        """Load predefined behavior blocks for autonomous driving"""
        self.block_definitions = {
            'Sensors': [
                {
                    'type': 'camera_sensor',
                    'name': 'Camera\nSensor',
                    'category': 'sensor',
                    'description': 'Captures visual data from the environment',
                    'parameters': {'resolution': 'string', 'fps': 'int'},
                    'has_input': False,
                    'has_output': True
                },
                {
                    'type': 'lidar_sensor',
                    'name': 'LIDAR\nSensor',
                    'category': 'sensor',
                    'description': 'Provides distance measurements in 360 degrees',
                    'parameters': {'range': 'float', 'resolution': 'float'},
                    'has_input': False,
                    'has_output': True
                },
                {
                    'type': 'gps_sensor',
                    'name': 'GPS\nSensor',
                    'category': 'sensor',
                    'description': 'Provides global position information',
                    'parameters': {'accuracy': 'float'},
                    'has_input': False,
                    'has_output': True
                }
            ],
            'Conditions': [
                {
                    'type': 'obstacle_detected',
                    'name': 'Obstacle\nDetected',
                    'category': 'condition',
                    'description': 'Checks if an obstacle is detected ahead',
                    'parameters': {'distance_threshold': 'float', 'angle_range': 'float'},
                    'has_input': True,
                    'has_output': True
                },
                {
                    'type': 'speed_check',
                    'name': 'Speed\nCheck',
                    'category': 'condition',
                    'description': 'Checks if vehicle speed meets criteria',
                    'parameters': {'min_speed': 'float', 'max_speed': 'float'},
                    'has_input': True,
                    'has_output': True
                },
                {
                    'type': 'lane_detection',
                    'name': 'Lane\nDetection',
                    'category': 'condition',
                    'description': 'Detects lane markings and position',
                    'parameters': {'confidence_threshold': 'float'},
                    'has_input': True,
                    'has_output': True
                }
            ],
            'Actions': [
                {
                    'type': 'accelerate',
                    'name': 'Accelerate',
                    'category': 'action',
                    'description': 'Increase vehicle speed',
                    'parameters': {'acceleration': 'float', 'max_speed': 'float'},
                    'has_input': True,
                    'has_output': True
                },
                {
                    'type': 'brake',
                    'name': 'Brake',
                    'category': 'action',
                    'description': 'Decrease vehicle speed',
                    'parameters': {'brake_force': 'float'},
                    'has_input': True,
                    'has_output': True
                },
                {
                    'type': 'steer',
                    'name': 'Steer',
                    'category': 'action',
                    'description': 'Change vehicle direction',
                    'parameters': {'steering_angle': 'float', 'duration': 'float'},
                    'has_input': True,
                    'has_output': True
                },
                {
                    'type': 'follow_path',
                    'name': 'Follow\nPath',
                    'category': 'action',
                    'description': 'Follow a predefined path',
                    'parameters': {'path_id': 'string', 'speed': 'float'},
                    'has_input': True,
                    'has_output': True
                }
            ],
            'Control': [
                {
                    'type': 'sequence',
                    'name': 'Sequence',
                    'category': 'composite',
                    'description': 'Execute children in order until one fails',
                    'parameters': {},
                    'has_input': True,
                    'has_output': True,
                    'multiple_children': True
                },
                {
                    'type': 'selector',
                    'name': 'Selector',
                    'category': 'composite',
                    'description': 'Execute children until one succeeds',
                    'parameters': {},
                    'has_input': True,
                    'has_output': True,
                    'multiple_children': True
                },
                {
                    'type': 'parallel',
                    'name': 'Parallel',
                    'category': 'composite',
                    'description': 'Execute all children simultaneously',
                    'parameters': {'success_count': 'int'},
                    'has_input': True,
                    'has_output': True,
                    'multiple_children': True
                }
            ]
        }
        
        self.create_category_tabs()
    
    def create_category_tabs(self):
        """Create tabs for each block category"""
        for category, blocks in self.block_definitions.items():
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            scroll_area = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            
            for block_def in blocks:
                block_widget = self.create_block_widget(block_def)
                scroll_layout.addWidget(block_widget)
            
            scroll_layout.addStretch()
            scroll_area.setWidget(scroll_widget)
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area)
            
            self.tab_widget.addTab(tab, category)
    
    def create_block_widget(self, block_def: Dict[str, Any]) -> QWidget:
        """Create a draggable widget for a block definition"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        widget.setFixedHeight(80)
        widget.setStyleSheet("""
            QFrame {
                border: 2px solid #666;
                border-radius: 8px;
                background-color: #f0f0f0;
                margin: 2px;
            }
            QFrame:hover {
                background-color: #e0e0e0;
                border-color: #888;
            }
        """)
        
        layout = QHBoxLayout(widget)
        
        # Block preview (simplified visual)
        preview = QLabel()
        preview.setFixedSize(60, 40)
        preview.setStyleSheet(f"""
            background-color: {self.get_category_color(block_def['category'])};
            border: 1px solid #333;
            border-radius: 4px;
        """)
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setText(block_def['name'].replace('\n', ' '))
        preview.setFont(QFont("Arial", 8))
        layout.addWidget(preview)
        
        # Block info
        info_layout = QVBoxLayout()
        name_label = QLabel(block_def['name'].replace('\n', ' '))
        name_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        info_layout.addWidget(name_label)
        
        desc_label = QLabel(block_def['description'])
        desc_label.setFont(QFont("Arial", 8))
        desc_label.setWordWrap(True)
        info_layout.addWidget(desc_label)
        
        layout.addLayout(info_layout)
        
        # Store block definition for drag operations
        widget.block_def = block_def
        widget.mousePressEvent = lambda event: self.start_drag(event, block_def)
        
        return widget
    
    def get_category_color(self, category: str) -> str:
        """Get color for block category"""
        colors = {
            'sensor': '#FFFF64',    # Yellow
            'condition': '#64AAFF', # Blue
            'action': '#64FF96',    # Green
            'composite': '#FF9664', # Orange
            'control': '#FF64FF'    # Magenta
        }
        return colors.get(category, '#AAAAAA')
    
    def start_drag(self, event, block_def: Dict[str, Any]):
        """Start drag operation for a block"""
        if event.button() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(json.dumps(block_def))
            drag.setMimeData(mime_data)
            drag.exec(Qt.DropAction.CopyAction)


class BehaviorCanvas(QGraphicsView):
    """Canvas for creating and editing behavior trees"""
    
    block_added = pyqtSignal(BehaviorBlock)
    connection_created = pyqtSignal(BehaviorBlock, BehaviorBlock)
    
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Canvas properties
        self.setAcceptDrops(True)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Connection state
        self.connecting_mode = False
        self.connection_start_block = None
        self.temp_connection_line = None
        
        # Grid background
        self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle dropping a block onto the canvas"""
        if event.mimeData().hasText():
            try:
                block_def = json.loads(event.mimeData().text())
                position = self.mapToScene(event.position().toPoint())
                
                # Create new behavior block
                block = BehaviorBlock(block_def['type'], block_def)
                block.setPos(position)
                self.scene.addItem(block)
                
                self.block_added.emit(block)
                event.acceptProposedAction()
                
            except json.JSONDecodeError:
                pass
    
    def mousePressEvent(self, event):
        """Handle mouse press for connection creation"""
        if event.button() == Qt.MouseButton.RightButton:
            # Right click to start connection
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, BehaviorBlock):
                self.start_connection(item)
                return
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for temporary connection line"""
        if self.connecting_mode and self.temp_connection_line:
            end_point = self.mapToScene(event.position().toPoint())
            # Update temporary connection line
            # This would be implemented with a temporary graphics item
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release for connection completion"""
        if self.connecting_mode and event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, BehaviorBlock) and item != self.connection_start_block:
                self.complete_connection(item)
            else:
                self.cancel_connection()
        
        super().mouseReleaseEvent(event)
    
    def start_connection(self, start_block: BehaviorBlock):
        """Start creating a connection from a block"""
        self.connecting_mode = True
        self.connection_start_block = start_block
        self.setCursor(Qt.CursorShape.CrossCursor)
    
    def complete_connection(self, end_block: BehaviorBlock):
        """Complete a connection between two blocks"""
        if self.connection_start_block and end_block:
            # Create connection line
            connection = ConnectionLine(self.connection_start_block, end_block)
            self.scene.addItem(connection)
            
            # Update block connections
            self.connection_start_block.output_connections.append(end_block)
            end_block.input_connections.append(self.connection_start_block)
            
            self.connection_created.emit(self.connection_start_block, end_block)
        
        self.cancel_connection()
    
    def cancel_connection(self):
        """Cancel connection creation"""
        self.connecting_mode = False
        self.connection_start_block = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        
        if self.temp_connection_line:
            self.scene.removeItem(self.temp_connection_line)
            self.temp_connection_line = None
    
    def update_connections(self, moved_block: BehaviorBlock):
        """Update connection lines when a block is moved"""
        # This would update all connection lines connected to the moved block
        # Implementation would iterate through connections and update their positions
        pass
    
    def get_behavior_tree_data(self) -> Dict[str, Any]:
        """Extract behavior tree data from the canvas"""
        blocks = []
        connections = []
        
        for item in self.scene.items():
            if isinstance(item, BehaviorBlock):
                block_data = {
                    'id': item.block_id,
                    'type': item.block_type,
                    'position': {'x': item.pos().x(), 'y': item.pos().y()},
                    'data': item.block_data,
                    'connections': [conn.block_id for conn in item.output_connections]
                }
                blocks.append(block_data)
        
        return {
            'blocks': blocks,
            'connections': connections
        }


class BehaviorEditor(QWidget):
    """Main behavior editor widget with Scratch-like visual programming interface"""
    
    behavior_changed = pyqtSignal(dict)  # Emitted when behavior tree changes
    
    def __init__(self):
        super().__init__()
        self.block_library = BlockLibrary()
        self.canvas = BehaviorCanvas()
        
        # Code generation and debugging components
        self.code_generator = CodeGenerator()
        self.debugger = BehaviorTreeDebugger(self.code_generator)
        self.validator = RealTimeBehaviorValidator()
        
        # Set up real-time validation
        self.validator.add_error_callback(self.on_validation_error)
        
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QHBoxLayout(self)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Block library
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self.block_library)
        
        # Add toolbar for canvas operations
        toolbar = self.create_toolbar()
        left_layout.addWidget(toolbar)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Canvas
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Canvas title and controls
        canvas_header = QHBoxLayout()
        canvas_title = QLabel("Behavior Tree Canvas")
        canvas_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        canvas_header.addWidget(canvas_title)
        canvas_header.addStretch()
        
        # Canvas controls
        clear_btn = QPushButton("Clear Canvas")
        clear_btn.clicked.connect(self.clear_canvas)
        canvas_header.addWidget(clear_btn)
        
        right_layout.addLayout(canvas_header)
        right_layout.addWidget(self.canvas)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])  # Set initial sizes
        
        layout.addWidget(splitter)
    
    def create_toolbar(self) -> QWidget:
        """Create toolbar with canvas operations"""
        toolbar = QFrame()
        toolbar.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout(toolbar)
        
        # Connection mode toggle
        connect_btn = QPushButton("Connection Mode")
        connect_btn.setCheckable(True)
        connect_btn.toggled.connect(self.toggle_connection_mode)
        layout.addWidget(connect_btn)
        
        # Validation button
        validate_btn = QPushButton("Validate Tree")
        validate_btn.clicked.connect(self.validate_behavior_tree)
        layout.addWidget(validate_btn)
        
        # Code generation button
        generate_btn = QPushButton("Generate Code")
        generate_btn.clicked.connect(self.generate_code)
        layout.addWidget(generate_btn)
        
        # Debug controls
        debug_btn = QPushButton("Start Debug")
        debug_btn.clicked.connect(self.start_debug_session)
        layout.addWidget(debug_btn)
        
        # Save/Load buttons
        save_btn = QPushButton("Save Tree")
        save_btn.clicked.connect(self.save_behavior_tree)
        layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load Tree")
        load_btn.clicked.connect(self.load_behavior_tree)
        layout.addWidget(load_btn)
        
        return toolbar
    
    def connect_signals(self):
        """Connect internal signals"""
        self.canvas.block_added.connect(self.on_block_added)
        self.canvas.connection_created.connect(self.on_connection_created)
    
    def toggle_connection_mode(self, enabled: bool):
        """Toggle connection creation mode"""
        # This would enable/disable connection creation mode
        pass
    
    def validate_behavior_tree(self):
        """Validate the current behavior tree"""
        tree_data = self.canvas.get_behavior_tree_data()
        
        # Perform real-time validation
        validation_result = self.validator.validate_real_time(tree_data)
        
        if validation_result.is_valid:
            print("✓ Behavior tree is valid!")
            if validation_result.warnings:
                print("Warnings:")
                for warning in validation_result.warnings:
                    print(f"  - {warning}")
        else:
            print("✗ Behavior tree has errors:")
            for error in validation_result.errors:
                print(f"  - {error}")
        
        self.behavior_changed.emit(tree_data)
    
    def save_behavior_tree(self):
        """Save the current behavior tree"""
        tree_data = self.canvas.get_behavior_tree_data()
        # Save logic would go here
        pass
    
    def load_behavior_tree(self):
        """Load a behavior tree from file"""
        # Load logic would go here
        pass
    
    def generate_code(self):
        """Generate executable code from the current behavior tree"""
        tree_data = self.canvas.get_behavior_tree_data()
        
        try:
            generated_code = self.code_generator.generate_code(tree_data)
            
            # Display generated code (in a real implementation, this might open a code editor)
            print("Generated Code:")
            print("=" * 50)
            print(generated_code)
            print("=" * 50)
            
        except ValueError as e:
            print(f"Code generation failed: {e}")
    
    def start_debug_session(self):
        """Start a debugging session for the current behavior tree"""
        tree_data = self.canvas.get_behavior_tree_data()
        
        if not tree_data.get('blocks'):
            print("No behavior tree to debug")
            return
        
        # Mock vehicle state for debugging
        mock_vehicle_state = type('MockVehicleState', (), {
            'position': type('Position', (), {'x': 0, 'y': 0, 'z': 0})(),
            'velocity': type('Velocity', (), {'magnitude': lambda: 10.0})(),
            'orientation': 0.0
        })()
        
        debug_steps = self.debugger.start_debug_session(tree_data, mock_vehicle_state)
        
        print(f"Debug session started with {len(debug_steps)} steps")
        print("Use step_forward() and step_backward() to navigate")
        
        # Show first step
        first_step = self.debugger.step_forward()
        if first_step:
            print(f"Step 1: {first_step.node_type} ({first_step.node_id}) - {first_step.state.value}")
    
    def on_validation_error(self, validation_result):
        """Handle validation errors from real-time validator"""
        print("Real-time validation error:")
        for error in validation_result.errors:
            print(f"  - {error}")
    
    def step_forward(self):
        """Step forward in debug session"""
        step = self.debugger.step_forward()
        if step:
            print(f"Debug step: {step.node_type} ({step.node_id}) - {step.state.value}")
            return step
        else:
            print("End of debug session")
            return None
    
    def step_backward(self):
        """Step backward in debug session"""
        step = self.debugger.step_backward()
        if step:
            print(f"Debug step: {step.node_type} ({step.node_id}) - {step.state.value}")
            return step
        else:
            print("Beginning of debug session")
            return None
    
    def clear_canvas(self):
        """Clear all blocks from the canvas"""
        self.canvas.scene.clear()
    
    def on_block_added(self, block: BehaviorBlock):
        """Handle when a new block is added to the canvas"""
        tree_data = self.canvas.get_behavior_tree_data()
        self.behavior_changed.emit(tree_data)
    
    def on_connection_created(self, start_block: BehaviorBlock, end_block: BehaviorBlock):
        """Handle when a new connection is created"""
        tree_data = self.canvas.get_behavior_tree_data()
        self.behavior_changed.emit(tree_data)
    
    def create_behavior_block(self, block_type: str) -> BehaviorBlock:
        """Create a new behavior block of the specified type"""
        # Find block definition
        for category_blocks in self.block_library.block_definitions.values():
            for block_def in category_blocks:
                if block_def['type'] == block_type:
                    # Create a copy and ensure parameters are set
                    block_def_copy = block_def.copy()
                    # Always set default parameters to ensure they exist
                    block_def_copy['parameters'] = self._get_default_parameters(block_type)
                    return BehaviorBlock(block_type, block_def_copy)
        
        # Return default block if type not found
        default_params = self._get_default_parameters(block_type)
        default_def = {
            'type': block_type,
            'name': block_type.replace('_', ' ').title(),
            'category': 'action',
            'description': f'Custom {block_type} block',
            'parameters': default_params,
            'has_input': True,
            'has_output': True
        }
        return BehaviorBlock(block_type, default_def)
    
    def _get_default_parameters(self, block_type: str) -> Dict[str, Any]:
        """Get default parameters for a block type"""
        defaults = {
            'camera_sensor': {'resolution': '1920x1080', 'fps': 30},
            'lidar_sensor': {'range': 100.0, 'resolution': 0.1},
            'gps_sensor': {'accuracy': 1.0},
            'obstacle_detected': {'distance_threshold': 5.0, 'angle_range': 30.0},
            'speed_check': {'min_speed': 0.0, 'max_speed': 50.0},
            'lane_detection': {'confidence_threshold': 0.8},
            'accelerate': {'acceleration': 2.0, 'max_speed': 60.0},
            'brake': {'brake_force': 0.8},
            'steer': {'steering_angle': 0.0, 'duration': 1.0},
            'follow_path': {'path_id': 'default_path', 'speed': 30.0},
            'sequence': {},
            'selector': {},
            'parallel': {'success_count': 1}
        }
        return defaults.get(block_type, {})
    
    def connect_blocks(self, source: BehaviorBlock, target: BehaviorBlock):
        """Connect two behavior blocks"""
        if source and target and source != target:
            # Create visual connection
            connection = ConnectionLine(source, target)
            self.canvas.scene.addItem(connection)
            
            # Update block connections
            source.output_connections.append(target)
            target.input_connections.append(source)
            
            # Emit change signal
            tree_data = self.canvas.get_behavior_tree_data()
            self.behavior_changed.emit(tree_data)
    
    def get_behavior_tree_json(self) -> str:
        """Get the current behavior tree as JSON"""
        tree_data = self.canvas.get_behavior_tree_data()
        return json.dumps(tree_data, indent=2)
    
    def load_behavior_tree_json(self, json_data: str):
        """Load a behavior tree from JSON data"""
        try:
            tree_data = json.loads(json_data)
            self.load_tree_data(tree_data)
        except json.JSONDecodeError as e:
            print(f"Error loading behavior tree: {e}")
    
    def load_tree_data(self, tree_data: Dict[str, Any]):
        """Load behavior tree from data dictionary"""
        self.clear_canvas()
        
        # Create blocks first
        block_map = {}
        for block_data in tree_data.get('blocks', []):
            block = BehaviorBlock(block_data['type'], block_data['data'])
            block.setPos(block_data['position']['x'], block_data['position']['y'])
            block.block_id = block_data['id']
            self.canvas.scene.addItem(block)
            block_map[block_data['id']] = block
        
        # Create connections
        for block_data in tree_data.get('blocks', []):
            source_block = block_map.get(block_data['id'])
            if source_block:
                for target_id in block_data.get('connections', []):
                    target_block = block_map.get(target_id)
                    if target_block:
                        self.connect_blocks(source_block, target_block)