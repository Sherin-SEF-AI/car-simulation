"""
Advanced 3D Scene Editor
Interactive 3D environment creation and editing tools
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QPushButton, QComboBox, QSlider, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QSplitter, QTreeWidget, QTreeWidgetItem, QScrollArea,
    QListWidget, QListWidgetItem, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap

import numpy as np
import json
import time
from collections import deque


class SceneObject:
    """3D scene object representation"""
    
    def __init__(self, name, object_type, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        self.name = name
        self.object_type = object_type
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.properties = {}
        self.children = []
        self.parent = None
        self.visible = True
        self.locked = False
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'type': self.object_type,
            'position': self.position,
            'rotation': self.rotation,
            'scale': self.scale,
            'properties': self.properties,
            'visible': self.visible,
            'locked': self.locked
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        obj = cls(
            data['name'],
            data['type'],
            data.get('position', (0, 0, 0)),
            data.get('rotation', (0, 0, 0)),
            data.get('scale', (1, 1, 1))
        )
        obj.properties = data.get('properties', {})
        obj.visible = data.get('visible', True)
        obj.locked = data.get('locked', False)
        return obj


class SceneHierarchy(QTreeWidget):
    """Scene hierarchy tree widget"""
    
    object_selected = pyqtSignal(str)  # object_name
    object_deleted = pyqtSignal(str)   # object_name
    
    def __init__(self):
        super().__init__()
        self.setHeaderLabels(["Name", "Type", "Visible"])
        self.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)
        
        # Context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Selection handling
        self.itemSelectionChanged.connect(self.on_selection_changed)
        
        self.scene_objects = {}
    
    def add_object(self, scene_object):
        """Add object to hierarchy"""
        item = QTreeWidgetItem([
            scene_object.name,
            scene_object.object_type,
            "✓" if scene_object.visible else "✗"
        ])
        item.setData(0, Qt.ItemDataRole.UserRole, scene_object.name)
        
        self.addTopLevelItem(item)
        self.scene_objects[scene_object.name] = scene_object
    
    def remove_object(self, object_name):
        """Remove object from hierarchy"""
        if object_name in self.scene_objects:
            del self.scene_objects[object_name]
        
        # Find and remove item
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if item.data(0, Qt.ItemDataRole.UserRole) == object_name:
                self.takeTopLevelItem(i)
                break
    
    def update_object(self, object_name, scene_object):
        """Update object in hierarchy"""
        if object_name in self.scene_objects:
            self.scene_objects[object_name] = scene_object
            
            # Update item display
            for i in range(self.topLevelItemCount()):
                item = self.topLevelItem(i)
                if item.data(0, Qt.ItemDataRole.UserRole) == object_name:
                    item.setText(0, scene_object.name)
                    item.setText(1, scene_object.object_type)
                    item.setText(2, "✓" if scene_object.visible else "✗")
                    break
    
    def on_selection_changed(self):
        """Handle selection change"""
        current_item = self.currentItem()
        if current_item:
            object_name = current_item.data(0, Qt.ItemDataRole.UserRole)
            self.object_selected.emit(object_name)
    
    def show_context_menu(self, position):
        """Show context menu"""
        # TODO: Implement context menu
        pass


class ObjectProperties(QWidget):
    """Object properties editor"""
    
    property_changed = pyqtSignal(str, str, object)  # object_name, property, value
    
    def __init__(self):
        super().__init__()
        self.current_object = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup properties UI"""
        layout = QVBoxLayout(self)
        
        # Object info
        info_group = QGroupBox("Object Information")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("Name:"), 0, 0)
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self.on_name_changed)
        info_layout.addWidget(self.name_edit, 0, 1)
        
        info_layout.addWidget(QLabel("Type:"), 1, 0)
        self.type_label = QLabel("None")
        info_layout.addWidget(self.type_label, 1, 1)
        
        layout.addWidget(info_group)
        
        # Transform
        transform_group = QGroupBox("Transform")
        transform_layout = QGridLayout(transform_group)
        
        # Position
        transform_layout.addWidget(QLabel("Position:"), 0, 0)
        self.pos_x = QDoubleSpinBox()
        self.pos_x.setRange(-1000, 1000)
        self.pos_x.valueChanged.connect(self.on_transform_changed)
        transform_layout.addWidget(self.pos_x, 0, 1)
        
        self.pos_y = QDoubleSpinBox()
        self.pos_y.setRange(-1000, 1000)
        self.pos_y.valueChanged.connect(self.on_transform_changed)
        transform_layout.addWidget(self.pos_y, 0, 2)
        
        self.pos_z = QDoubleSpinBox()
        self.pos_z.setRange(-1000, 1000)
        self.pos_z.valueChanged.connect(self.on_transform_changed)
        transform_layout.addWidget(self.pos_z, 0, 3)
        
        # Rotation
        transform_layout.addWidget(QLabel("Rotation:"), 1, 0)
        self.rot_x = QDoubleSpinBox()
        self.rot_x.setRange(-360, 360)
        self.rot_x.valueChanged.connect(self.on_transform_changed)
        transform_layout.addWidget(self.rot_x, 1, 1)
        
        self.rot_y = QDoubleSpinBox()
        self.rot_y.setRange(-360, 360)
        self.rot_y.valueChanged.connect(self.on_transform_changed)
        transform_layout.addWidget(self.rot_y, 1, 2)
        
        self.rot_z = QDoubleSpinBox()
        self.rot_z.setRange(-360, 360)
        self.rot_z.valueChanged.connect(self.on_transform_changed)
        transform_layout.addWidget(self.rot_z, 1, 3)
        
        # Scale
        transform_layout.addWidget(QLabel("Scale:"), 2, 0)
        self.scale_x = QDoubleSpinBox()
        self.scale_x.setRange(0.01, 100)
        self.scale_x.setValue(1.0)
        self.scale_x.valueChanged.connect(self.on_transform_changed)
        transform_layout.addWidget(self.scale_x, 2, 1)
        
        self.scale_y = QDoubleSpinBox()
        self.scale_y.setRange(0.01, 100)
        self.scale_y.setValue(1.0)
        self.scale_y.valueChanged.connect(self.on_transform_changed)
        transform_layout.addWidget(self.scale_y, 2, 2)
        
        self.scale_z = QDoubleSpinBox()
        self.scale_z.setRange(0.01, 100)
        self.scale_z.setValue(1.0)
        self.scale_z.valueChanged.connect(self.on_transform_changed)
        transform_layout.addWidget(self.scale_z, 2, 3)
        
        layout.addWidget(transform_group)
        
        # Visibility and locking
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        self.visible_check = QCheckBox("Visible")
        self.visible_check.setChecked(True)
        self.visible_check.toggled.connect(self.on_visibility_changed)
        options_layout.addWidget(self.visible_check)
        
        self.locked_check = QCheckBox("Locked")
        self.locked_check.toggled.connect(self.on_lock_changed)
        options_layout.addWidget(self.locked_check)
        
        layout.addWidget(options_group)
        
        # Custom properties
        custom_group = QGroupBox("Custom Properties")
        custom_layout = QVBoxLayout(custom_group)
        
        self.properties_table = QTableWidget()
        self.properties_table.setColumnCount(2)
        self.properties_table.setHorizontalHeaderLabels(["Property", "Value"])
        custom_layout.addWidget(self.properties_table)
        
        # Add property button
        add_prop_btn = QPushButton("Add Property")
        add_prop_btn.clicked.connect(self.add_custom_property)
        custom_layout.addWidget(add_prop_btn)
        
        layout.addWidget(custom_group)
    
    def set_object(self, scene_object):
        """Set current object"""
        self.current_object = scene_object
        
        if scene_object:
            # Update UI with object data
            self.name_edit.setText(scene_object.name)
            self.type_label.setText(scene_object.object_type)
            
            # Transform
            self.pos_x.setValue(scene_object.position[0])
            self.pos_y.setValue(scene_object.position[1])
            self.pos_z.setValue(scene_object.position[2])
            
            self.rot_x.setValue(scene_object.rotation[0])
            self.rot_y.setValue(scene_object.rotation[1])
            self.rot_z.setValue(scene_object.rotation[2])
            
            self.scale_x.setValue(scene_object.scale[0])
            self.scale_y.setValue(scene_object.scale[1])
            self.scale_z.setValue(scene_object.scale[2])
            
            # Options
            self.visible_check.setChecked(scene_object.visible)
            self.locked_check.setChecked(scene_object.locked)
            
            # Custom properties
            self.update_properties_table()
        else:
            # Clear UI
            self.name_edit.clear()
            self.type_label.setText("None")
            self.properties_table.setRowCount(0)
    
    def update_properties_table(self):
        """Update custom properties table"""
        if not self.current_object:
            return
        
        properties = self.current_object.properties
        self.properties_table.setRowCount(len(properties))
        
        for i, (key, value) in enumerate(properties.items()):
            self.properties_table.setItem(i, 0, QTableWidgetItem(str(key)))
            self.properties_table.setItem(i, 1, QTableWidgetItem(str(value)))
    
    def on_name_changed(self, name):
        """Handle name change"""
        if self.current_object:
            old_name = self.current_object.name
            self.current_object.name = name
            self.property_changed.emit(old_name, "name", name)
    
    def on_transform_changed(self):
        """Handle transform change"""
        if self.current_object:
            self.current_object.position = (
                self.pos_x.value(),
                self.pos_y.value(),
                self.pos_z.value()
            )
            self.current_object.rotation = (
                self.rot_x.value(),
                self.rot_y.value(),
                self.rot_z.value()
            )
            self.current_object.scale = (
                self.scale_x.value(),
                self.scale_y.value(),
                self.scale_z.value()
            )
            self.property_changed.emit(self.current_object.name, "transform", None)
    
    def on_visibility_changed(self, visible):
        """Handle visibility change"""
        if self.current_object:
            self.current_object.visible = visible
            self.property_changed.emit(self.current_object.name, "visible", visible)
    
    def on_lock_changed(self, locked):
        """Handle lock change"""
        if self.current_object:
            self.current_object.locked = locked
            self.property_changed.emit(self.current_object.name, "locked", locked)
    
    def add_custom_property(self):
        """Add custom property"""
        if not self.current_object:
            return
        
        # Simple dialog for property name and value
        from PyQt6.QtWidgets import QInputDialog
        
        prop_name, ok = QInputDialog.getText(self, "Property Name", "Enter property name:")
        if ok and prop_name:
            prop_value, ok = QInputDialog.getText(self, "Property Value", "Enter property value:")
            if ok:
                self.current_object.properties[prop_name] = prop_value
                self.update_properties_table()
                self.property_changed.emit(self.current_object.name, "properties", None)


class SceneEditor(QWidget):
    """Advanced 3D scene editor"""
    
    scene_changed = pyqtSignal()
    object_added = pyqtSignal(str)    # object_name
    object_removed = pyqtSignal(str)  # object_name
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        
        # Scene data
        self.scene_objects = {}
        self.selected_object = None
        self.scene_name = "Untitled Scene"
        
        self.setup_ui()
        self.setup_connections()
        
        print("Scene editor initialized")
    
    def setup_ui(self):
        """Setup scene editor UI"""
        layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        # File operations
        new_scene_btn = QPushButton("New Scene")
        new_scene_btn.clicked.connect(self.new_scene)
        toolbar_layout.addWidget(new_scene_btn)
        
        load_scene_btn = QPushButton("Load Scene")
        load_scene_btn.clicked.connect(self.load_scene)
        toolbar_layout.addWidget(load_scene_btn)
        
        save_scene_btn = QPushButton("Save Scene")
        save_scene_btn.clicked.connect(self.save_scene)
        toolbar_layout.addWidget(save_scene_btn)
        
        toolbar_layout.addStretch()
        
        # Object creation
        toolbar_layout.addWidget(QLabel("Add Object:"))
        
        self.object_type_combo = QComboBox()
        self.object_type_combo.addItems([
            "Road Segment", "Building", "Tree", "Traffic Light",
            "Sign", "Barrier", "Vehicle Spawn", "Waypoint",
            "Obstacle", "Decoration", "Light Source"
        ])
        toolbar_layout.addWidget(self.object_type_combo)
        
        add_object_btn = QPushButton("Add")
        add_object_btn.clicked.connect(self.add_object)
        toolbar_layout.addWidget(add_object_btn)
        
        delete_object_btn = QPushButton("Delete")
        delete_object_btn.clicked.connect(self.delete_selected_object)
        toolbar_layout.addWidget(delete_object_btn)
        
        layout.addLayout(toolbar_layout)
        
        # Main content
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Scene hierarchy and properties
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Scene hierarchy
        hierarchy_group = QGroupBox("Scene Hierarchy")
        hierarchy_layout = QVBoxLayout(hierarchy_group)
        
        self.scene_hierarchy = SceneHierarchy()
        hierarchy_layout.addWidget(self.scene_hierarchy)
        
        left_layout.addWidget(hierarchy_group)
        
        # Object properties
        properties_group = QGroupBox("Object Properties")
        properties_layout = QVBoxLayout(properties_group)
        
        self.object_properties = ObjectProperties()
        properties_layout.addWidget(self.object_properties)
        
        left_layout.addWidget(properties_group)
        
        main_splitter.addWidget(left_panel)
        
        # Right panel - Scene preview and tools
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Scene preview (placeholder)
        preview_group = QGroupBox("Scene Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.scene_preview = QLabel("3D Scene Preview\n(Integration with 3D viewport)")
        self.scene_preview.setMinimumSize(400, 300)
        self.scene_preview.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                color: #cccccc;
                font-size: 14px;
            }
        """)
        self.scene_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.scene_preview)
        
        right_layout.addWidget(preview_group)
        
        # Scene tools
        tools_group = QGroupBox("Scene Tools")
        tools_layout = QGridLayout(tools_group)
        
        # Grid settings
        tools_layout.addWidget(QLabel("Grid Size:"), 0, 0)
        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(1, 100)
        self.grid_size_spin.setValue(10)
        tools_layout.addWidget(self.grid_size_spin, 0, 1)
        
        self.show_grid_check = QCheckBox("Show Grid")
        self.show_grid_check.setChecked(True)
        tools_layout.addWidget(self.show_grid_check, 0, 2)
        
        # Snap settings
        self.snap_to_grid_check = QCheckBox("Snap to Grid")
        tools_layout.addWidget(self.snap_to_grid_check, 1, 0)
        
        self.snap_rotation_check = QCheckBox("Snap Rotation")
        tools_layout.addWidget(self.snap_rotation_check, 1, 1)
        
        # View settings
        tools_layout.addWidget(QLabel("View Mode:"), 2, 0)
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Perspective", "Orthographic", "Top", "Front", "Side"])
        tools_layout.addWidget(self.view_mode_combo, 2, 1)
        
        right_layout.addWidget(tools_group)
        
        # Scene statistics
        stats_group = QGroupBox("Scene Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.scene_stats = {}
        stats_items = [
            ("Objects", "0"),
            ("Vertices", "0"),
            ("Triangles", "0"),
            ("Materials", "0"),
            ("Lights", "0"),
            ("File Size", "0 KB")
        ]
        
        for i, (name, default) in enumerate(stats_items):
            row, col = i // 2, (i % 2) * 2
            
            label = QLabel(f"{name}:")
            stats_layout.addWidget(label, row, col)
            
            value_label = QLabel(default)
            value_label.setStyleSheet("color: #4a90e2; font-weight: bold;")
            stats_layout.addWidget(value_label, row, col + 1)
            
            self.scene_stats[name] = value_label
        
        right_layout.addWidget(stats_group)
        
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([300, 500])
        
        layout.addWidget(main_splitter)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.scene_hierarchy.object_selected.connect(self.on_object_selected)
        self.scene_hierarchy.object_deleted.connect(self.delete_object)
        self.object_properties.property_changed.connect(self.on_property_changed)
    
    def new_scene(self):
        """Create new scene"""
        reply = QMessageBox.question(
            self, "New Scene", 
            "Create a new scene? This will clear the current scene.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.scene_objects.clear()
            self.scene_hierarchy.clear()
            self.scene_hierarchy.scene_objects.clear()
            self.object_properties.set_object(None)
            self.selected_object = None
            self.scene_name = "Untitled Scene"
            self.update_scene_stats()
            
            print("New scene created")
    
    def load_scene(self):
        """Load scene from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Scene", "", 
            "Scene Files (*.scene);;JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    scene_data = json.load(f)
                
                # Clear current scene
                self.new_scene()
                
                # Load objects
                for obj_data in scene_data.get('objects', []):
                    scene_obj = SceneObject.from_dict(obj_data)
                    self.scene_objects[scene_obj.name] = scene_obj
                    self.scene_hierarchy.add_object(scene_obj)
                
                self.scene_name = scene_data.get('name', 'Loaded Scene')
                self.update_scene_stats()
                
                QMessageBox.information(
                    self, "Load Complete", 
                    f"Scene loaded from {filename}\n{len(self.scene_objects)} objects loaded."
                )
                
                print(f"Scene loaded: {filename}")
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Load Error", 
                    f"Failed to load scene:\n{str(e)}"
                )
    
    def save_scene(self):
        """Save scene to file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Scene", f"{self.scene_name}.scene", 
            "Scene Files (*.scene);;JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                scene_data = {
                    'name': self.scene_name,
                    'version': '1.0',
                    'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'objects': [obj.to_dict() for obj in self.scene_objects.values()]
                }
                
                with open(filename, 'w') as f:
                    json.dump(scene_data, f, indent=2)
                
                QMessageBox.information(
                    self, "Save Complete", 
                    f"Scene saved to {filename}\n{len(self.scene_objects)} objects saved."
                )
                
                print(f"Scene saved: {filename}")
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error", 
                    f"Failed to save scene:\n{str(e)}"
                )
    
    def add_object(self):
        """Add new object to scene"""
        object_type = self.object_type_combo.currentText()
        
        # Generate unique name
        base_name = object_type.replace(" ", "_").lower()
        counter = 1
        object_name = f"{base_name}_{counter}"
        
        while object_name in self.scene_objects:
            counter += 1
            object_name = f"{base_name}_{counter}"
        
        # Create object
        scene_obj = SceneObject(object_name, object_type)
        
        # Add some default properties based on type
        if object_type == "Road Segment":
            scene_obj.properties = {
                'width': 3.5,
                'length': 10.0,
                'material': 'asphalt',
                'lanes': 1
            }
        elif object_type == "Building":
            scene_obj.properties = {
                'height': 10.0,
                'width': 8.0,
                'depth': 8.0,
                'floors': 3
            }
        elif object_type == "Traffic Light":
            scene_obj.properties = {
                'cycle_time': 30.0,
                'current_state': 'red',
                'height': 4.0
            }
        
        # Add to scene
        self.scene_objects[object_name] = scene_obj
        self.scene_hierarchy.add_object(scene_obj)
        
        # Select new object
        self.selected_object = object_name
        self.object_properties.set_object(scene_obj)
        
        self.update_scene_stats()
        self.object_added.emit(object_name)
        
        print(f"Added object: {object_name} ({object_type})")
    
    def delete_selected_object(self):
        """Delete currently selected object"""
        if self.selected_object:
            self.delete_object(self.selected_object)
    
    def delete_object(self, object_name):
        """Delete object from scene"""
        if object_name in self.scene_objects:
            reply = QMessageBox.question(
                self, "Delete Object", 
                f"Delete object '{object_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                del self.scene_objects[object_name]
                self.scene_hierarchy.remove_object(object_name)
                
                if self.selected_object == object_name:
                    self.selected_object = None
                    self.object_properties.set_object(None)
                
                self.update_scene_stats()
                self.object_removed.emit(object_name)
                
                print(f"Deleted object: {object_name}")
    
    @pyqtSlot(str)
    def on_object_selected(self, object_name):
        """Handle object selection"""
        self.selected_object = object_name
        
        if object_name in self.scene_objects:
            scene_obj = self.scene_objects[object_name]
            self.object_properties.set_object(scene_obj)
        else:
            self.object_properties.set_object(None)
    
    @pyqtSlot(str, str, object)
    def on_property_changed(self, object_name, property_name, value):
        """Handle property change"""
        if object_name in self.scene_objects:
            # Update hierarchy display if needed
            if property_name in ['name', 'visible']:
                scene_obj = self.scene_objects[object_name]
                self.scene_hierarchy.update_object(object_name, scene_obj)
                
                # Handle name change
                if property_name == 'name' and value != object_name:
                    self.scene_objects[value] = self.scene_objects.pop(object_name)
                    self.selected_object = value
            
            self.scene_changed.emit()
            print(f"Property changed: {object_name}.{property_name} = {value}")
    
    def update_scene_stats(self):
        """Update scene statistics"""
        object_count = len(self.scene_objects)
        
        # Calculate estimated statistics
        vertex_count = object_count * 100  # Rough estimate
        triangle_count = object_count * 50
        material_count = len(set(obj.properties.get('material', 'default') 
                                for obj in self.scene_objects.values()))
        light_count = sum(1 for obj in self.scene_objects.values() 
                         if obj.object_type == "Light Source")
        
        # Estimate file size
        file_size = len(json.dumps([obj.to_dict() for obj in self.scene_objects.values()]))
        file_size_kb = file_size / 1024
        
        # Update displays
        self.scene_stats["Objects"].setText(str(object_count))
        self.scene_stats["Vertices"].setText(str(vertex_count))
        self.scene_stats["Triangles"].setText(str(triangle_count))
        self.scene_stats["Materials"].setText(str(material_count))
        self.scene_stats["Lights"].setText(str(light_count))
        self.scene_stats["File Size"].setText(f"{file_size_kb:.1f} KB")
    
    def get_scene_data(self):
        """Get current scene data"""
        return {
            'name': self.scene_name,
            'objects': {name: obj.to_dict() for name, obj in self.scene_objects.items()}
        }
    
    def apply_scene_to_simulation(self):
        """Apply current scene to simulation"""
        # This would integrate with the main simulation
        scene_data = self.get_scene_data()
        
        # TODO: Apply scene objects to simulation environment
        print(f"Applying scene to simulation: {len(self.scene_objects)} objects")
        
        return scene_data