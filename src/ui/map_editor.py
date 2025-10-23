"""
Integrated map editor for creating and editing simulation environments
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QGroupBox,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QTabWidget,
    QFileDialog, QMessageBox, QSlider, QCheckBox, QSplitter, QFrame,
    QScrollArea, QToolBar, QColorDialog, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect, QTimer, QMimeData
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QIcon, QFont, QDragEnterEvent,
    QDropEvent, QMouseEvent, QPaintEvent, QWheelEvent, QKeyEvent, QAction
)
from typing import Dict, Any, List, Optional, Tuple
import json
import os
import math
import random

from ..core.environment import (
    EnvironmentAsset, Vector3, MapBounds, EnvironmentConfiguration,
    EnvironmentType, SurfaceProperties
)
from ..core.traffic_system import TrafficLight, RoadSign, TrafficLightState


class MapCanvas(QWidget):
    """Interactive canvas for map editing with drag-and-drop support"""
    
    # Signals
    asset_selected = pyqtSignal(object)  # EnvironmentAsset
    asset_moved = pyqtSignal(str, object)  # asset_id, new_position
    asset_deleted = pyqtSignal(str)  # asset_id
    waypoint_added = pyqtSignal(object)  # Vector3
    surface_painted = pyqtSignal(str, object, object)  # surface_type, start_pos, end_pos
    
    def __init__(self):
        super().__init__()
        
        # Canvas properties
        self.scale = 1.0
        self.offset = QPoint(0, 0)
        self.canvas_size = 1000  # Virtual canvas size
        self.grid_size = 10
        self.show_grid = True
        
        # Map data
        self.assets: Dict[str, EnvironmentAsset] = {}
        self.waypoints: List[Vector3] = []
        self.surface_regions: Dict[str, List[Tuple[Vector3, Vector3]]] = {}
        self.map_bounds = MapBounds(-500, 500, -500, 500)
        
        # Editing state
        self.selected_asset_id: Optional[str] = None
        self.dragging_asset = False
        self.drag_start_pos = QPoint()
        self.current_tool = "select"  # "select", "place", "paint", "waypoint"
        self.current_asset_type = "building"
        self.current_surface_type = "asphalt"
        self.painting_surface = False
        self.paint_start_pos = None
        
        # Asset library
        self.asset_library = {
            "building": {
                "color": QColor(100, 100, 100),
                "size": (10, 10),
                "icon": "🏢"
            },
            "tree": {
                "color": QColor(0, 150, 0),
                "size": (5, 5),
                "icon": "🌳"
            },
            "traffic_light": {
                "color": QColor(255, 255, 0),
                "size": (3, 3),
                "icon": "🚦"
            },
            "road_sign": {
                "color": QColor(255, 0, 0),
                "size": (2, 2),
                "icon": "🛑"
            },
            "obstacle": {
                "color": QColor(150, 75, 0),
                "size": (4, 4),
                "icon": "🚧"
            }
        }
        
        # Surface types
        self.surface_types = {
            "asphalt": QColor(80, 80, 80),
            "grass": QColor(0, 120, 0),
            "dirt": QColor(139, 69, 19),
            "gravel": QColor(128, 128, 128),
            "water": QColor(0, 100, 200)
        }
        
        # Setup widget
        self.setMinimumSize(800, 600)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        
        # Update timer for smooth interactions
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(16)  # ~60 FPS
    
    def set_tool(self, tool: str):
        """Set the current editing tool"""
        self.current_tool = tool
        self.selected_asset_id = None
        self.update()
    
    def set_asset_type(self, asset_type: str):
        """Set the current asset type for placement"""
        self.current_asset_type = asset_type
    
    def set_surface_type(self, surface_type: str):
        """Set the current surface type for painting"""
        self.current_surface_type = surface_type
    
    def world_to_screen(self, world_pos: Vector3) -> QPoint:
        """Convert world coordinates to screen coordinates"""
        screen_x = int((world_pos.x + self.canvas_size // 2) * self.scale + self.offset.x())
        screen_y = int((world_pos.z + self.canvas_size // 2) * self.scale + self.offset.y())
        return QPoint(screen_x, screen_y)
    
    def screen_to_world(self, screen_pos: QPoint) -> Vector3:
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_pos.x() - self.offset.x()) / self.scale - self.canvas_size // 2
        world_z = (screen_pos.y() - self.offset.y()) / self.scale - self.canvas_size // 2
        return Vector3(world_x, 0, world_z)
    
    def snap_to_grid(self, pos: Vector3) -> Vector3:
        """Snap position to grid"""
        if self.show_grid:
            snapped_x = round(pos.x / self.grid_size) * self.grid_size
            snapped_z = round(pos.z / self.grid_size) * self.grid_size
            return Vector3(snapped_x, pos.y, snapped_z)
        return pos
    
    def add_asset(self, asset: EnvironmentAsset):
        """Add an asset to the map"""
        self.assets[asset.asset_id] = asset
        self.update()
    
    def remove_asset(self, asset_id: str):
        """Remove an asset from the map"""
        if asset_id in self.assets:
            del self.assets[asset_id]
            if self.selected_asset_id == asset_id:
                self.selected_asset_id = None
            self.update()
    
    def add_waypoint(self, position: Vector3):
        """Add a waypoint to the map"""
        snapped_pos = self.snap_to_grid(position)
        self.waypoints.append(snapped_pos)
        self.waypoint_added.emit(snapped_pos)
        self.update()
    
    def clear_waypoints(self):
        """Clear all waypoints"""
        self.waypoints.clear()
        self.update()
    
    def add_surface_region(self, surface_type: str, start_pos: Vector3, end_pos: Vector3):
        """Add a surface region"""
        if surface_type not in self.surface_regions:
            self.surface_regions[surface_type] = []
        
        # Ensure start is top-left, end is bottom-right
        min_x = min(start_pos.x, end_pos.x)
        max_x = max(start_pos.x, end_pos.x)
        min_z = min(start_pos.z, end_pos.z)
        max_z = max(start_pos.z, end_pos.z)
        
        region = (Vector3(min_x, 0, min_z), Vector3(max_x, 0, max_z))
        self.surface_regions[surface_type].append(region)
        self.surface_painted.emit(surface_type, region[0], region[1])
        self.update()
    
    def get_asset_at_position(self, world_pos: Vector3) -> Optional[str]:
        """Get asset ID at world position"""
        for asset_id, asset in self.assets.items():
            # Simple bounding box check
            asset_info = self.asset_library.get(asset.asset_type, {"size": (5, 5)})
            size_x, size_z = asset_info["size"]
            
            if (abs(world_pos.x - asset.position.x) <= size_x / 2 and
                abs(world_pos.z - asset.position.z) <= size_z / 2):
                return asset_id
        
        return None
    
    def paintEvent(self, event: QPaintEvent):
        """Paint the map canvas"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Clear background
        painter.fillRect(self.rect(), QColor(240, 240, 240))
        
        # Draw grid
        if self.show_grid:
            self._draw_grid(painter)
        
        # Draw surface regions
        self._draw_surface_regions(painter)
        
        # Draw map bounds
        self._draw_map_bounds(painter)
        
        # Draw assets
        self._draw_assets(painter)
        
        # Draw waypoints
        self._draw_waypoints(painter)
        
        # Draw current painting operation
        if self.painting_surface and self.paint_start_pos:
            self._draw_paint_preview(painter)
        
        # Draw tool cursor
        self._draw_tool_cursor(painter)
    
    def _draw_grid(self, painter: QPainter):
        """Draw grid lines"""
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        
        # Calculate visible grid range
        top_left = self.screen_to_world(QPoint(0, 0))
        bottom_right = self.screen_to_world(QPoint(self.width(), self.height()))
        
        # Draw vertical lines
        start_x = int(top_left.x // self.grid_size) * self.grid_size
        end_x = int(bottom_right.x // self.grid_size + 1) * self.grid_size
        
        for x in range(start_x, end_x + 1, self.grid_size):
            screen_start = self.world_to_screen(Vector3(x, 0, top_left.z))
            screen_end = self.world_to_screen(Vector3(x, 0, bottom_right.z))
            painter.drawLine(screen_start, screen_end)
        
        # Draw horizontal lines
        start_z = int(top_left.z // self.grid_size) * self.grid_size
        end_z = int(bottom_right.z // self.grid_size + 1) * self.grid_size
        
        for z in range(start_z, end_z + 1, self.grid_size):
            screen_start = self.world_to_screen(Vector3(top_left.x, 0, z))
            screen_end = self.world_to_screen(Vector3(bottom_right.x, 0, z))
            painter.drawLine(screen_start, screen_end)
    
    def _draw_surface_regions(self, painter: QPainter):
        """Draw surface regions"""
        for surface_type, regions in self.surface_regions.items():
            color = self.surface_types.get(surface_type, QColor(128, 128, 128))
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(), 1))
            
            for start_pos, end_pos in regions:
                screen_start = self.world_to_screen(start_pos)
                screen_end = self.world_to_screen(end_pos)
                
                rect = QRect(screen_start, screen_end).normalized()
                painter.drawRect(rect)
    
    def _draw_map_bounds(self, painter: QPainter):
        """Draw map boundaries"""
        painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
        
        top_left = self.world_to_screen(Vector3(self.map_bounds.min_x, 0, self.map_bounds.min_z))
        bottom_right = self.world_to_screen(Vector3(self.map_bounds.max_x, 0, self.map_bounds.max_z))
        
        rect = QRect(top_left, bottom_right).normalized()
        painter.drawRect(rect)
    
    def _draw_assets(self, painter: QPainter):
        """Draw all assets"""
        for asset_id, asset in self.assets.items():
            asset_info = self.asset_library.get(asset.asset_type, {"color": QColor(128, 128, 128), "size": (5, 5)})
            
            # Get screen position
            screen_pos = self.world_to_screen(asset.position)
            size_x, size_z = asset_info["size"]
            screen_size_x = int(size_x * self.scale)
            screen_size_z = int(size_z * self.scale)
            
            # Draw asset
            color = asset_info["color"]
            if asset_id == self.selected_asset_id:
                color = color.lighter(150)
                painter.setPen(QPen(QColor(255, 255, 0), 3))
            else:
                painter.setPen(QPen(color.darker(), 1))
            
            painter.setBrush(QBrush(color))
            
            rect = QRect(
                screen_pos.x() - screen_size_x // 2,
                screen_pos.y() - screen_size_z // 2,
                screen_size_x,
                screen_size_z
            )
            painter.drawRect(rect)
            
            # Draw asset icon/label
            if self.scale > 0.5:  # Only draw text when zoomed in enough
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.setFont(QFont("Arial", max(8, int(10 * self.scale))))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, asset_info.get("icon", asset.asset_type[:2].upper()))
    
    def _draw_waypoints(self, painter: QPainter):
        """Draw waypoints"""
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        painter.setPen(QPen(QColor(0, 200, 0), 2))
        
        for i, waypoint in enumerate(self.waypoints):
            screen_pos = self.world_to_screen(waypoint)
            radius = max(3, int(5 * self.scale))
            
            painter.drawEllipse(screen_pos, radius, radius)
            
            # Draw waypoint number
            if self.scale > 0.3:
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.setFont(QFont("Arial", max(6, int(8 * self.scale))))
                painter.drawText(screen_pos.x() - 10, screen_pos.y() + 5, str(i + 1))
                painter.setPen(QPen(QColor(0, 200, 0), 2))
        
        # Draw connections between waypoints
        if len(self.waypoints) > 1:
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine))
            for i in range(len(self.waypoints) - 1):
                start_screen = self.world_to_screen(self.waypoints[i])
                end_screen = self.world_to_screen(self.waypoints[i + 1])
                painter.drawLine(start_screen, end_screen)
    
    def _draw_paint_preview(self, painter: QPainter):
        """Draw surface painting preview"""
        if not self.paint_start_pos:
            return
        
        current_pos = self.mapFromGlobal(self.cursor().pos())
        if not self.rect().contains(current_pos):
            return
        
        color = self.surface_types.get(self.current_surface_type, QColor(128, 128, 128))
        painter.setBrush(QBrush(color.lighter(150)))
        painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
        
        rect = QRect(self.paint_start_pos, current_pos).normalized()
        painter.drawRect(rect)
    
    def _draw_tool_cursor(self, painter: QPainter):
        """Draw tool-specific cursor"""
        if self.current_tool == "place":
            # Show preview of asset to be placed
            cursor_pos = self.mapFromGlobal(self.cursor().pos())
            if self.rect().contains(cursor_pos):
                world_pos = self.screen_to_world(cursor_pos)
                snapped_pos = self.snap_to_grid(world_pos)
                screen_pos = self.world_to_screen(snapped_pos)
                
                asset_info = self.asset_library.get(self.current_asset_type, {"color": QColor(128, 128, 128), "size": (5, 5)})
                color = asset_info["color"]
                color.setAlpha(128)
                
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color.darker(), 2, Qt.PenStyle.DashLine))
                
                size_x, size_z = asset_info["size"]
                screen_size_x = int(size_x * self.scale)
                screen_size_z = int(size_z * self.scale)
                
                rect = QRect(
                    screen_pos.x() - screen_size_x // 2,
                    screen_pos.y() - screen_size_z // 2,
                    screen_size_x,
                    screen_size_z
                )
                painter.drawRect(rect)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events"""
        world_pos = self.screen_to_world(event.pos())
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_tool == "select":
                # Select or start dragging asset
                asset_id = self.get_asset_at_position(world_pos)
                if asset_id:
                    self.selected_asset_id = asset_id
                    self.dragging_asset = True
                    self.drag_start_pos = event.pos()
                    self.asset_selected.emit(self.assets[asset_id])
                else:
                    self.selected_asset_id = None
                    self.asset_selected.emit(None)
            
            elif self.current_tool == "place":
                # Place new asset
                self._place_asset(world_pos)
            
            elif self.current_tool == "waypoint":
                # Add waypoint
                self.add_waypoint(world_pos)
            
            elif self.current_tool == "paint":
                # Start surface painting
                self.painting_surface = True
                self.paint_start_pos = event.pos()
        
        elif event.button() == Qt.MouseButton.RightButton:
            if self.current_tool == "select":
                # Delete asset
                asset_id = self.get_asset_at_position(world_pos)
                if asset_id:
                    self.remove_asset(asset_id)
                    self.asset_deleted.emit(asset_id)
        
        self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events"""
        if self.dragging_asset and self.selected_asset_id:
            # Move selected asset
            world_pos = self.screen_to_world(event.pos())
            snapped_pos = self.snap_to_grid(world_pos)
            
            asset = self.assets[self.selected_asset_id]
            asset.position = snapped_pos
            self.asset_moved.emit(self.selected_asset_id, snapped_pos)
        
        self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.painting_surface and self.paint_start_pos:
                # Finish surface painting
                start_world = self.screen_to_world(self.paint_start_pos)
                end_world = self.screen_to_world(event.pos())
                
                start_snapped = self.snap_to_grid(start_world)
                end_snapped = self.snap_to_grid(end_world)
                
                self.add_surface_region(self.current_surface_type, start_snapped, end_snapped)
                
                self.painting_surface = False
                self.paint_start_pos = None
            
            self.dragging_asset = False
        
        self.update()
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 1.0 / 1.1
        
        # Zoom towards mouse position
        mouse_pos = event.position().toPoint()
        world_pos_before = self.screen_to_world(mouse_pos)
        
        self.scale *= zoom_factor
        self.scale = max(0.1, min(5.0, self.scale))  # Limit zoom range
        
        world_pos_after = self.screen_to_world(mouse_pos)
        
        # Adjust offset to keep mouse position stable
        delta = self.world_to_screen(world_pos_before) - self.world_to_screen(world_pos_after)
        self.offset += delta
        
        self.update()
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events"""
        if event.key() == Qt.Key.Key_Delete and self.selected_asset_id:
            # Delete selected asset
            self.remove_asset(self.selected_asset_id)
            self.asset_deleted.emit(self.selected_asset_id)
        
        elif event.key() == Qt.Key.Key_G:
            # Toggle grid
            self.show_grid = not self.show_grid
            self.update()
        
        elif event.key() == Qt.Key.Key_C:
            # Clear waypoints
            self.clear_waypoints()
    
    def _place_asset(self, world_pos: Vector3):
        """Place a new asset at the specified position"""
        snapped_pos = self.snap_to_grid(world_pos)
        
        # Generate unique asset ID
        asset_id = f"{self.current_asset_type}_{len(self.assets)}_{int(snapped_pos.x)}_{int(snapped_pos.z)}"
        
        # Create asset
        asset = EnvironmentAsset(
            asset_id=asset_id,
            asset_type=self.current_asset_type,
            position=snapped_pos,
            rotation=Vector3(0, 0, 0),
            scale=Vector3(1, 1, 1),
            properties={}
        )
        
        self.add_asset(asset)
    
    def zoom_to_fit(self):
        """Zoom to fit all content"""
        if not self.assets and not self.waypoints:
            return
        
        # Calculate bounding box of all content
        min_x = min_z = float('inf')
        max_x = max_z = float('-inf')
        
        for asset in self.assets.values():
            min_x = min(min_x, asset.position.x)
            max_x = max(max_x, asset.position.x)
            min_z = min(min_z, asset.position.z)
            max_z = max(max_z, asset.position.z)
        
        for waypoint in self.waypoints:
            min_x = min(min_x, waypoint.x)
            max_x = max(max_x, waypoint.x)
            min_z = min(min_z, waypoint.z)
            max_z = max(max_z, waypoint.z)
        
        # Add padding
        padding = 50
        min_x -= padding
        max_x += padding
        min_z -= padding
        max_z += padding
        
        # Calculate required scale
        content_width = max_x - min_x
        content_height = max_z - min_z
        
        scale_x = self.width() / content_width if content_width > 0 else 1.0
        scale_z = self.height() / content_height if content_height > 0 else 1.0
        
        self.scale = min(scale_x, scale_z, 2.0)  # Limit maximum zoom
        
        # Center content
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2
        
        screen_center = self.world_to_screen(Vector3(center_x, 0, center_z))
        widget_center = QPoint(self.width() // 2, self.height() // 2)
        
        self.offset = widget_center - screen_center + self.offset
        
        self.update()


class MapEditor(QWidget):
    """Main map editor widget with tools and property panels"""
    
    # Signals
    map_saved = pyqtSignal(str)  # file_path
    map_loaded = pyqtSignal(str)  # file_path
    environment_generated = pyqtSignal(object)  # EnvironmentConfiguration
    
    def __init__(self):
        super().__init__()
        
        # Create main layout
        self.main_layout = QHBoxLayout(self)
        
        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.splitter)
        
        # Create map canvas first (needed by other panels)
        self.canvas = MapCanvas()
        
        # Create tool panel
        self.tool_panel = self._create_tool_panel()
        self.splitter.addWidget(self.tool_panel)
        
        # Add canvas to splitter
        self.splitter.addWidget(self.canvas)
        
        # Create property panel
        self.property_panel = self._create_property_panel()
        self.splitter.addWidget(self.property_panel)
        
        # Set splitter proportions
        self.splitter.setSizes([200, 800, 200])
        
        # Connect signals
        self.canvas.asset_selected.connect(self._on_asset_selected)
        self.canvas.asset_moved.connect(self._on_asset_moved)
        self.canvas.asset_deleted.connect(self._on_asset_deleted)
        self.canvas.waypoint_added.connect(self._on_waypoint_added)
        self.canvas.surface_painted.connect(self._on_surface_painted)
        
        # Current map data
        self.current_map_file = None
        self.map_modified = False
    
    def _create_tool_panel(self) -> QWidget:
        """Create the tool panel with editing tools"""
        panel = QWidget()
        panel.setMaximumWidth(250)
        layout = QVBoxLayout(panel)
        
        # Tool selection
        tools_group = QGroupBox("Tools")
        tools_layout = QVBoxLayout(tools_group)
        
        self.tool_buttons = {}
        tools = [
            ("select", "Select/Move", "🔍"),
            ("place", "Place Asset", "📍"),
            ("paint", "Paint Surface", "🎨"),
            ("waypoint", "Add Waypoint", "📌")
        ]
        
        for tool_id, tool_name, icon in tools:
            btn = QPushButton(f"{icon} {tool_name}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, t=tool_id: self._set_tool(t))
            self.tool_buttons[tool_id] = btn
            tools_layout.addWidget(btn)
        
        # Set default tool
        self.tool_buttons["select"].setChecked(True)
        
        layout.addWidget(tools_group)
        
        # Asset library
        assets_group = QGroupBox("Asset Library")
        assets_layout = QVBoxLayout(assets_group)
        
        self.asset_type_combo = QComboBox()
        self.asset_type_combo.addItems(["building", "tree", "traffic_light", "road_sign", "obstacle"])
        self.asset_type_combo.currentTextChanged.connect(self.canvas.set_asset_type)
        assets_layout.addWidget(QLabel("Asset Type:"))
        assets_layout.addWidget(self.asset_type_combo)
        
        layout.addWidget(assets_group)
        
        # Surface painting
        surface_group = QGroupBox("Surface Types")
        surface_layout = QVBoxLayout(surface_group)
        
        self.surface_type_combo = QComboBox()
        self.surface_type_combo.addItems(["asphalt", "grass", "dirt", "gravel", "water"])
        self.surface_type_combo.currentTextChanged.connect(self.canvas.set_surface_type)
        surface_layout.addWidget(QLabel("Surface Type:"))
        surface_layout.addWidget(self.surface_type_combo)
        
        layout.addWidget(surface_group)
        
        # Map settings
        map_group = QGroupBox("Map Settings")
        map_layout = QVBoxLayout(map_group)
        
        # Grid settings
        grid_checkbox = QCheckBox("Show Grid")
        grid_checkbox.setChecked(True)
        grid_checkbox.toggled.connect(self._toggle_grid)
        map_layout.addWidget(grid_checkbox)
        
        # Grid size
        map_layout.addWidget(QLabel("Grid Size:"))
        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(1, 50)
        self.grid_size_spin.setValue(10)
        self.grid_size_spin.valueChanged.connect(self._set_grid_size)
        map_layout.addWidget(self.grid_size_spin)
        
        layout.addWidget(map_group)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        new_btn = QPushButton("🆕 New Map")
        new_btn.clicked.connect(self.new_map)
        file_layout.addWidget(new_btn)
        
        load_btn = QPushButton("📁 Load Map")
        load_btn.clicked.connect(self.load_map)
        file_layout.addWidget(load_btn)
        
        save_btn = QPushButton("💾 Save Map")
        save_btn.clicked.connect(self.save_map)
        file_layout.addWidget(save_btn)
        
        save_as_btn = QPushButton("💾 Save As...")
        save_as_btn.clicked.connect(self.save_map_as)
        file_layout.addWidget(save_as_btn)
        
        layout.addWidget(file_group)
        
        # Generation tools
        gen_group = QGroupBox("Generation")
        gen_layout = QVBoxLayout(gen_group)
        
        gen_urban_btn = QPushButton("🏙️ Generate Urban")
        gen_urban_btn.clicked.connect(lambda: self._generate_environment("urban"))
        gen_layout.addWidget(gen_urban_btn)
        
        gen_highway_btn = QPushButton("🛣️ Generate Highway")
        gen_highway_btn.clicked.connect(lambda: self._generate_environment("highway"))
        gen_layout.addWidget(gen_highway_btn)
        
        gen_offroad_btn = QPushButton("🌲 Generate Off-road")
        gen_offroad_btn.clicked.connect(lambda: self._generate_environment("offroad"))
        gen_layout.addWidget(gen_offroad_btn)
        
        layout.addWidget(gen_group)
        
        # View controls
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)
        
        zoom_fit_btn = QPushButton("🔍 Zoom to Fit")
        zoom_fit_btn.clicked.connect(self.canvas.zoom_to_fit)
        view_layout.addWidget(zoom_fit_btn)
        
        clear_waypoints_btn = QPushButton("🗑️ Clear Waypoints")
        clear_waypoints_btn.clicked.connect(self.canvas.clear_waypoints)
        view_layout.addWidget(clear_waypoints_btn)
        
        layout.addWidget(view_group)
        
        layout.addStretch()
        
        return panel
    
    def _create_property_panel(self) -> QWidget:
        """Create the property panel for editing selected objects"""
        panel = QWidget()
        panel.setMaximumWidth(250)
        layout = QVBoxLayout(panel)
        
        # Selected asset properties
        self.asset_props_group = QGroupBox("Asset Properties")
        self.asset_props_group.setEnabled(False)
        props_layout = QVBoxLayout(self.asset_props_group)
        
        # Asset ID
        props_layout.addWidget(QLabel("Asset ID:"))
        self.asset_id_label = QLabel("None")
        props_layout.addWidget(self.asset_id_label)
        
        # Asset type
        props_layout.addWidget(QLabel("Type:"))
        self.asset_type_label = QLabel("None")
        props_layout.addWidget(self.asset_type_label)
        
        # Position
        props_layout.addWidget(QLabel("Position:"))
        pos_layout = QHBoxLayout()
        
        self.pos_x_spin = QDoubleSpinBox()
        self.pos_x_spin.setRange(-1000, 1000)
        self.pos_x_spin.valueChanged.connect(self._update_asset_position)
        pos_layout.addWidget(QLabel("X:"))
        pos_layout.addWidget(self.pos_x_spin)
        
        self.pos_z_spin = QDoubleSpinBox()
        self.pos_z_spin.setRange(-1000, 1000)
        self.pos_z_spin.valueChanged.connect(self._update_asset_position)
        pos_layout.addWidget(QLabel("Z:"))
        pos_layout.addWidget(self.pos_z_spin)
        
        props_layout.addLayout(pos_layout)
        
        # Rotation
        props_layout.addWidget(QLabel("Rotation (Y):"))
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(0, 360)
        self.rotation_spin.valueChanged.connect(self._update_asset_rotation)
        props_layout.addWidget(self.rotation_spin)
        
        # Scale
        props_layout.addWidget(QLabel("Scale:"))
        scale_layout = QHBoxLayout()
        
        self.scale_x_spin = QDoubleSpinBox()
        self.scale_x_spin.setRange(0.1, 10.0)
        self.scale_x_spin.setValue(1.0)
        self.scale_x_spin.valueChanged.connect(self._update_asset_scale)
        scale_layout.addWidget(QLabel("X:"))
        scale_layout.addWidget(self.scale_x_spin)
        
        self.scale_z_spin = QDoubleSpinBox()
        self.scale_z_spin.setRange(0.1, 10.0)
        self.scale_z_spin.setValue(1.0)
        self.scale_z_spin.valueChanged.connect(self._update_asset_scale)
        scale_layout.addWidget(QLabel("Z:"))
        scale_layout.addWidget(self.scale_z_spin)
        
        props_layout.addLayout(scale_layout)
        
        layout.addWidget(self.asset_props_group)
        
        # Map statistics
        stats_group = QGroupBox("Map Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_assets_label = QLabel("Assets: 0")
        stats_layout.addWidget(self.stats_assets_label)
        
        self.stats_waypoints_label = QLabel("Waypoints: 0")
        stats_layout.addWidget(self.stats_waypoints_label)
        
        self.stats_surfaces_label = QLabel("Surface Regions: 0")
        stats_layout.addWidget(self.stats_surfaces_label)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel
    
    def _set_tool(self, tool: str):
        """Set the active tool"""
        # Update button states
        for tool_id, button in self.tool_buttons.items():
            button.setChecked(tool_id == tool)
        
        # Set canvas tool
        self.canvas.set_tool(tool)
    
    def _toggle_grid(self, enabled: bool):
        """Toggle grid display"""
        self.canvas.show_grid = enabled
        self.canvas.update()
    
    def _set_grid_size(self, size: int):
        """Set grid size"""
        self.canvas.grid_size = size
        self.canvas.update()
    
    def _on_asset_selected(self, asset: Optional[EnvironmentAsset]):
        """Handle asset selection"""
        if asset:
            self.asset_props_group.setEnabled(True)
            self.asset_id_label.setText(asset.asset_id)
            self.asset_type_label.setText(asset.asset_type)
            
            # Update position spinboxes
            self.pos_x_spin.blockSignals(True)
            self.pos_z_spin.blockSignals(True)
            self.pos_x_spin.setValue(asset.position.x)
            self.pos_z_spin.setValue(asset.position.z)
            self.pos_x_spin.blockSignals(False)
            self.pos_z_spin.blockSignals(False)
            
            # Update rotation
            self.rotation_spin.blockSignals(True)
            self.rotation_spin.setValue(asset.rotation.y)
            self.rotation_spin.blockSignals(False)
            
            # Update scale
            self.scale_x_spin.blockSignals(True)
            self.scale_z_spin.blockSignals(True)
            self.scale_x_spin.setValue(asset.scale.x)
            self.scale_z_spin.setValue(asset.scale.z)
            self.scale_x_spin.blockSignals(False)
            self.scale_z_spin.blockSignals(False)
        else:
            self.asset_props_group.setEnabled(False)
            self.asset_id_label.setText("None")
            self.asset_type_label.setText("None")
        
        self._update_statistics()
    
    def _on_asset_moved(self, asset_id: str, new_position: Vector3):
        """Handle asset movement"""
        self.map_modified = True
        
        # Update property panel if this asset is selected
        if self.canvas.selected_asset_id == asset_id:
            self.pos_x_spin.blockSignals(True)
            self.pos_z_spin.blockSignals(True)
            self.pos_x_spin.setValue(new_position.x)
            self.pos_z_spin.setValue(new_position.z)
            self.pos_x_spin.blockSignals(False)
            self.pos_z_spin.blockSignals(False)
    
    def _on_asset_deleted(self, asset_id: str):
        """Handle asset deletion"""
        self.map_modified = True
        self._update_statistics()
    
    def _on_waypoint_added(self, position: Vector3):
        """Handle waypoint addition"""
        self.map_modified = True
        self._update_statistics()
    
    def _on_surface_painted(self, surface_type: str, start_pos: Vector3, end_pos: Vector3):
        """Handle surface painting"""
        self.map_modified = True
        self._update_statistics()
    
    def _update_asset_position(self):
        """Update selected asset position"""
        if not self.canvas.selected_asset_id:
            return
        
        asset = self.canvas.assets[self.canvas.selected_asset_id]
        asset.position.x = self.pos_x_spin.value()
        asset.position.z = self.pos_z_spin.value()
        
        self.canvas.update()
        self.map_modified = True
    
    def _update_asset_rotation(self):
        """Update selected asset rotation"""
        if not self.canvas.selected_asset_id:
            return
        
        asset = self.canvas.assets[self.canvas.selected_asset_id]
        asset.rotation.y = self.rotation_spin.value()
        
        self.canvas.update()
        self.map_modified = True
    
    def _update_asset_scale(self):
        """Update selected asset scale"""
        if not self.canvas.selected_asset_id:
            return
        
        asset = self.canvas.assets[self.canvas.selected_asset_id]
        asset.scale.x = self.scale_x_spin.value()
        asset.scale.z = self.scale_z_spin.value()
        
        self.canvas.update()
        self.map_modified = True
    
    def _update_statistics(self):
        """Update map statistics display"""
        num_assets = len(self.canvas.assets)
        num_waypoints = len(self.canvas.waypoints)
        num_surfaces = sum(len(regions) for regions in self.canvas.surface_regions.values())
        
        self.stats_assets_label.setText(f"Assets: {num_assets}")
        self.stats_waypoints_label.setText(f"Waypoints: {num_waypoints}")
        self.stats_surfaces_label.setText(f"Surface Regions: {num_surfaces}")
    
    def new_map(self):
        """Create a new map"""
        if self.map_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before creating a new map?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                if not self.save_map():
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                return
        
        # Clear canvas
        self.canvas.assets.clear()
        self.canvas.waypoints.clear()
        self.canvas.surface_regions.clear()
        self.canvas.selected_asset_id = None
        
        # Reset properties
        self.current_map_file = None
        self.map_modified = False
        
        self.canvas.update()
        self._update_statistics()
    
    def load_map(self):
        """Load a map from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Map", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Clear current map
            self.canvas.assets.clear()
            self.canvas.waypoints.clear()
            self.canvas.surface_regions.clear()
            
            # Load assets
            for asset_data in data.get('assets', []):
                asset = EnvironmentAsset(
                    asset_id=asset_data['asset_id'],
                    asset_type=asset_data['asset_type'],
                    position=Vector3(**asset_data['position']),
                    rotation=Vector3(**asset_data['rotation']),
                    scale=Vector3(**asset_data['scale']),
                    properties=asset_data.get('properties', {})
                )
                self.canvas.assets[asset.asset_id] = asset
            
            # Load waypoints
            for waypoint_data in data.get('waypoints', []):
                waypoint = Vector3(**waypoint_data)
                self.canvas.waypoints.append(waypoint)
            
            # Load surface regions
            for surface_type, regions_data in data.get('surface_regions', {}).items():
                regions = []
                for region_data in regions_data:
                    start = Vector3(**region_data['start'])
                    end = Vector3(**region_data['end'])
                    regions.append((start, end))
                self.canvas.surface_regions[surface_type] = regions
            
            # Load map bounds
            if 'map_bounds' in data:
                bounds_data = data['map_bounds']
                self.canvas.map_bounds = MapBounds(
                    bounds_data['min_x'], bounds_data['max_x'],
                    bounds_data['min_z'], bounds_data['max_z']
                )
            
            self.current_map_file = file_path
            self.map_modified = False
            
            self.canvas.update()
            self._update_statistics()
            self.map_loaded.emit(file_path)
            
            QMessageBox.information(self, "Success", f"Map loaded from {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load map: {str(e)}")
    
    def save_map(self) -> bool:
        """Save the current map"""
        if not self.current_map_file:
            return self.save_map_as()
        
        return self._save_to_file(self.current_map_file)
    
    def save_map_as(self) -> bool:
        """Save the current map with a new filename"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Map", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return False
        
        if not file_path.endswith('.json'):
            file_path += '.json'
        
        if self._save_to_file(file_path):
            self.current_map_file = file_path
            return True
        
        return False
    
    def _save_to_file(self, file_path: str) -> bool:
        """Save map data to file"""
        try:
            data = {
                'assets': [
                    {
                        'asset_id': asset.asset_id,
                        'asset_type': asset.asset_type,
                        'position': {'x': asset.position.x, 'y': asset.position.y, 'z': asset.position.z},
                        'rotation': {'x': asset.rotation.x, 'y': asset.rotation.y, 'z': asset.rotation.z},
                        'scale': {'x': asset.scale.x, 'y': asset.scale.y, 'z': asset.scale.z},
                        'properties': asset.properties
                    }
                    for asset in self.canvas.assets.values()
                ],
                'waypoints': [
                    {'x': wp.x, 'y': wp.y, 'z': wp.z}
                    for wp in self.canvas.waypoints
                ],
                'surface_regions': {
                    surface_type: [
                        {
                            'start': {'x': start.x, 'y': start.y, 'z': start.z},
                            'end': {'x': end.x, 'y': end.y, 'z': end.z}
                        }
                        for start, end in regions
                    ]
                    for surface_type, regions in self.canvas.surface_regions.items()
                },
                'map_bounds': {
                    'min_x': self.canvas.map_bounds.min_x,
                    'max_x': self.canvas.map_bounds.max_x,
                    'min_z': self.canvas.map_bounds.min_z,
                    'max_z': self.canvas.map_bounds.max_z
                }
            }
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.map_modified = False
            self.map_saved.emit(file_path)
            
            QMessageBox.information(self, "Success", f"Map saved to {file_path}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save map: {str(e)}")
            return False
    
    def _generate_environment(self, env_type: str):
        """Generate a procedural environment"""
        # This would integrate with the procedural generator from the environment system
        # For now, we'll create a simple example
        
        reply = QMessageBox.question(
            self, "Generate Environment",
            f"This will clear the current map and generate a new {env_type} environment. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Clear current map
        self.canvas.assets.clear()
        self.canvas.waypoints.clear()
        self.canvas.surface_regions.clear()
        
        # Generate based on type
        if env_type == "urban":
            self._generate_urban_environment()
        elif env_type == "highway":
            self._generate_highway_environment()
        elif env_type == "offroad":
            self._generate_offroad_environment()
        
        self.map_modified = True
        self.canvas.update()
        self._update_statistics()
    
    def _generate_urban_environment(self):
        """Generate an urban environment"""
        # Create street grid
        for x in range(-200, 201, 50):
            for z in range(-200, 201, 50):
                # Streets
                self.canvas.add_surface_region("asphalt", Vector3(x-5, 0, -200), Vector3(x+5, 0, 200))
                self.canvas.add_surface_region("asphalt", Vector3(-200, 0, z-5), Vector3(200, 0, z+5))
                
                # Buildings
                if abs(x) > 10 and abs(z) > 10 and random.random() < 0.6:
                    building_x = x + random.uniform(-15, 15)
                    building_z = z + random.uniform(-15, 15)
                    
                    asset_id = f"building_{len(self.canvas.assets)}"
                    asset = EnvironmentAsset(
                        asset_id=asset_id,
                        asset_type="building",
                        position=Vector3(building_x, 0, building_z),
                        rotation=Vector3(0, random.uniform(0, 360), 0),
                        scale=Vector3(random.uniform(0.8, 2.0), 1.0, random.uniform(0.8, 2.0))
                    )
                    self.canvas.assets[asset_id] = asset
        
        # Add some traffic lights at intersections
        for x in range(-200, 201, 100):
            for z in range(-200, 201, 100):
                if x != 0 or z != 0:  # Skip center
                    asset_id = f"traffic_light_{len(self.canvas.assets)}"
                    asset = EnvironmentAsset(
                        asset_id=asset_id,
                        asset_type="traffic_light",
                        position=Vector3(x, 0, z),
                        rotation=Vector3(0, 0, 0),
                        scale=Vector3(1, 1, 1)
                    )
                    self.canvas.assets[asset_id] = asset
    
    def _generate_highway_environment(self):
        """Generate a highway environment"""
        # Main highway
        self.canvas.add_surface_region("asphalt", Vector3(-300, 0, -20), Vector3(300, 0, 20))
        
        # Grass shoulders
        self.canvas.add_surface_region("grass", Vector3(-300, 0, -50), Vector3(300, 0, -20))
        self.canvas.add_surface_region("grass", Vector3(-300, 0, 20), Vector3(300, 0, 50))
        
        # Highway signs
        for x in range(-250, 251, 100):
            if random.random() < 0.4:
                asset_id = f"road_sign_{len(self.canvas.assets)}"
                asset = EnvironmentAsset(
                    asset_id=asset_id,
                    asset_type="road_sign",
                    position=Vector3(x, 0, 35),
                    rotation=Vector3(0, 0, 0),
                    scale=Vector3(1, 1, 1)
                )
                self.canvas.assets[asset_id] = asset
        
        # Add waypoints along highway
        for x in range(-250, 251, 50):
            self.canvas.waypoints.append(Vector3(x, 0, 0))
    
    def _generate_offroad_environment(self):
        """Generate an off-road environment"""
        # Varied terrain
        terrain_types = ["dirt", "grass", "gravel"]
        
        for x in range(-200, 201, 40):
            for z in range(-200, 201, 40):
                terrain = random.choice(terrain_types)
                size = random.uniform(20, 60)
                
                self.canvas.add_surface_region(
                    terrain,
                    Vector3(x - size/2, 0, z - size/2),
                    Vector3(x + size/2, 0, z + size/2)
                )
        
        # Natural obstacles
        for _ in range(30):
            x = random.uniform(-180, 180)
            z = random.uniform(-180, 180)
            
            obstacle_type = random.choice(["tree", "obstacle"])
            asset_id = f"{obstacle_type}_{len(self.canvas.assets)}"
            asset = EnvironmentAsset(
                asset_id=asset_id,
                asset_type=obstacle_type,
                position=Vector3(x, 0, z),
                rotation=Vector3(0, random.uniform(0, 360), 0),
                scale=Vector3(random.uniform(0.5, 2.0), 1.0, random.uniform(0.5, 2.0))
            )
            self.canvas.assets[asset_id] = asset
        
        # Winding path waypoints
        for i in range(20):
            angle = i * 0.3
            radius = 50 + i * 5
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            self.canvas.waypoints.append(Vector3(x, 0, z))
    
    def get_environment_configuration(self) -> EnvironmentConfiguration:
        """Get the current map as an EnvironmentConfiguration"""
        # Convert surface regions to the expected format
        surface_layout = {}
        for surface_type, regions in self.canvas.surface_regions.items():
            surface_layout[surface_type] = [
                (start.x, start.z, end.x, end.z)
                for start, end in regions
            ]
        
        return EnvironmentConfiguration(
            environment_type=EnvironmentType.MIXED,  # User-created maps are mixed type
            map_bounds=self.canvas.map_bounds,
            surface_layout=surface_layout,
            assets=list(self.canvas.assets.values()),
            spawn_points=[Vector3(0, 0, 0)],  # Default spawn point
            waypoints=self.canvas.waypoints.copy(),
            metadata={"created_with": "map_editor", "modified": self.map_modified}
        )