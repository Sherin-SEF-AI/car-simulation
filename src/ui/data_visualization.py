"""
Comprehensive data visualization tools for real-time performance metrics and sensor data
"""

import math
import time
from collections import deque
from typing import Dict, List, Any, Optional, Tuple

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QTabWidget, QGroupBox, QLabel, QPushButton, QComboBox,
                            QCheckBox, QSlider, QSpinBox, QScrollArea, QSplitter,
                            QFrame, QSizePolicy, QProgressBar, QTextEdit)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QFontMetrics, QPalette
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from ..core.application import SimulationApplication


class RealTimeGraph(QWidget):
    """Real-time graph widget for displaying time-series data"""
    
    def __init__(self, title: str = "Graph", max_points: int = 100):
        super().__init__()
        self.title = title
        self.max_points = max_points
        self.data_series = {}  # name -> deque of (time, value) tuples
        self.colors = {}  # name -> QColor
        self.y_range = [0, 100]  # Auto-scaling range
        self.time_range = 60.0  # 60 seconds
        
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Color palette for different series
        self.color_palette = [
            QColor(255, 99, 132),   # Red
            QColor(54, 162, 235),   # Blue
            QColor(255, 205, 86),   # Yellow
            QColor(75, 192, 192),   # Teal
            QColor(153, 102, 255),  # Purple
            QColor(255, 159, 64),   # Orange
            QColor(199, 199, 199),  # Grey
            QColor(83, 102, 255),   # Indigo
        ]
        self.color_index = 0
    
    def add_series(self, name: str, color: Optional[QColor] = None):
        """Add a new data series to the graph"""
        if name not in self.data_series:
            self.data_series[name] = deque(maxlen=self.max_points)
            if color is None:
                color = self.color_palette[self.color_index % len(self.color_palette)]
                self.color_index += 1
            self.colors[name] = color
    
    def add_data_point(self, series_name: str, timestamp: float, value: float):
        """Add a data point to a series"""
        if series_name not in self.data_series:
            self.add_series(series_name)
        
        self.data_series[series_name].append((timestamp, value))
        self.update()  # Trigger repaint
    
    def clear_series(self, series_name: str):
        """Clear all data from a series"""
        if series_name in self.data_series:
            self.data_series[series_name].clear()
            self.update()
    
    def clear_all(self):
        """Clear all data from all series"""
        for series in self.data_series.values():
            series.clear()
        self.update()
    
    def set_y_range(self, min_val: float, max_val: float):
        """Set the Y-axis range"""
        self.y_range = [min_val, max_val]
        self.update()
    
    def auto_scale_y(self):
        """Auto-scale Y-axis based on current data"""
        all_values = []
        for series in self.data_series.values():
            all_values.extend([point[1] for point in series])
        
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            margin = (max_val - min_val) * 0.1  # 10% margin
            self.y_range = [min_val - margin, max_val + margin]
        else:
            self.y_range = [0, 100]
    
    def paintEvent(self, event):
        """Paint the graph"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        margin = 40
        
        # Draw background
        painter.fillRect(self.rect(), QColor(250, 250, 250))
        
        # Draw border
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(0, 0, width - 1, height - 1)
        
        # Calculate plot area
        plot_x = margin
        plot_y = margin
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
        
        if plot_width <= 0 or plot_height <= 0:
            return
        
        # Draw title
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        font = QFont()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(10, 20, self.title)
        
        # Draw axes
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawLine(plot_x, plot_y + plot_height, plot_x + plot_width, plot_y + plot_height)  # X-axis
        painter.drawLine(plot_x, plot_y, plot_x, plot_y + plot_height)  # Y-axis
        
        # Draw grid
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        for i in range(1, 5):
            y = plot_y + (plot_height * i / 5)
            painter.drawLine(plot_x, y, plot_x + plot_width, y)
        
        for i in range(1, 6):
            x = plot_x + (plot_width * i / 6)
            painter.drawLine(x, plot_y, x, plot_y + plot_height)
        
        # Draw Y-axis labels
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        
        y_min, y_max = self.y_range
        for i in range(6):
            y_val = y_min + (y_max - y_min) * (5 - i) / 5
            y_pos = plot_y + (plot_height * i / 5)
            painter.drawText(5, y_pos + 5, f"{y_val:.1f}")
        
        # Get current time for X-axis calculation
        current_time = time.time()
        
        # Draw data series
        for series_name, data_points in self.data_series.items():
            if len(data_points) < 2:
                continue
            
            color = self.colors[series_name]
            painter.setPen(QPen(color, 2))
            
            # Convert data points to screen coordinates
            points = []
            for timestamp, value in data_points:
                # Only show points within time range
                if current_time - timestamp > self.time_range:
                    continue
                
                # Calculate screen coordinates
                x_ratio = 1.0 - (current_time - timestamp) / self.time_range
                y_ratio = (value - y_min) / (y_max - y_min) if y_max != y_min else 0.5
                
                screen_x = plot_x + x_ratio * plot_width
                screen_y = plot_y + (1.0 - y_ratio) * plot_height
                
                points.append((screen_x, screen_y))
            
            # Draw lines between points
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
            
            # Draw points
            painter.setBrush(QBrush(color))
            for x, y in points:
                painter.drawEllipse(int(x - 2), int(y - 2), 4, 4)
        
        # Draw legend
        legend_y = plot_y + 10
        for i, (series_name, color) in enumerate(self.colors.items()):
            legend_x = plot_x + plot_width - 150
            legend_item_y = legend_y + i * 20
            
            # Draw color box
            painter.fillRect(legend_x, legend_item_y - 8, 12, 12, color)
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.drawRect(legend_x, legend_item_y - 8, 12, 12)
            
            # Draw series name
            painter.setPen(QPen(QColor(50, 50, 50), 1))
            painter.drawText(legend_x + 18, legend_item_y + 2, series_name)


class MetricsDisplay(QWidget):
    """Display widget for key performance metrics"""
    
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the metrics display UI"""
        layout = QGridLayout(self)
        layout.setSpacing(10)
        
        # Create metric cards
        self.metric_cards = {}
        
        # Define metrics to display
        metrics_config = [
            ("FPS", "Frames/sec", QColor(76, 175, 80)),
            ("Vehicles", "Count", QColor(33, 150, 243)),
            ("Memory", "MB", QColor(255, 152, 0)),
            ("Physics", "ms", QColor(156, 39, 176)),
            ("Render", "ms", QColor(244, 67, 54)),
            ("AI", "ms", QColor(0, 150, 136)),
        ]
        
        for i, (name, unit, color) in enumerate(metrics_config):
            card = self.create_metric_card(name, unit, color)
            self.metric_cards[name] = card
            
            row = i // 3
            col = i % 3
            layout.addWidget(card, row, col)
    
    def create_metric_card(self, name: str, unit: str, color: QColor) -> QWidget:
        """Create a metric display card"""
        card = QFrame()
        card.setFrameStyle(QFrame.Shape.StyledPanel)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border: 2px solid {color.name()};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        card.setMinimumSize(120, 80)
        
        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Metric name
        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet(f"color: {color.name()}; font-weight: bold; font-size: 12px;")
        layout.addWidget(name_label)
        
        # Metric value
        value_label = QLabel("0")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        layout.addWidget(value_label)
        
        # Metric unit
        unit_label = QLabel(unit)
        unit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        unit_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(unit_label)
        
        # Store references for updating
        card.value_label = value_label
        card.color = color
        
        return card
    
    def update_metric(self, name: str, value: float, status: str = "normal"):
        """Update a metric value"""
        if name in self.metric_cards:
            card = self.metric_cards[name]
            
            # Format value based on metric type
            if name == "Memory":
                formatted_value = f"{value:.0f}"
            elif name in ["Physics", "Render", "AI"]:
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.0f}"
            
            card.value_label.setText(formatted_value)
            
            # Update color based on status
            if status == "warning":
                card.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FF9800;")
            elif status == "error":
                card.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #F44336;")
            else:
                card.value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")


class AIDecisionOverlay(QOpenGLWidget):
    """3D overlay widget for visualizing AI decision-making processes"""
    
    def __init__(self):
        super().__init__()
        self.decisions = []  # List of decision data
        self.paths = []  # List of planned paths
        self.obstacles = []  # List of detected obstacles
        
        self.setMinimumSize(400, 300)
    
    def add_decision(self, vehicle_id: str, decision_type: str, confidence: float, position: Tuple[float, float]):
        """Add an AI decision to visualize"""
        self.decisions.append({
            'vehicle_id': vehicle_id,
            'type': decision_type,
            'confidence': confidence,
            'position': position,
            'timestamp': time.time()
        })
        
        # Keep only recent decisions
        current_time = time.time()
        self.decisions = [d for d in self.decisions if current_time - d['timestamp'] < 5.0]
        
        self.update()
    
    def add_path(self, vehicle_id: str, waypoints: List[Tuple[float, float]]):
        """Add a planned path to visualize"""
        self.paths.append({
            'vehicle_id': vehicle_id,
            'waypoints': waypoints,
            'timestamp': time.time()
        })
        
        # Keep only recent paths
        current_time = time.time()
        self.paths = [p for p in self.paths if current_time - p['timestamp'] < 10.0]
        
        self.update()
    
    def add_obstacle(self, obstacle_id: str, position: Tuple[float, float], size: float):
        """Add a detected obstacle to visualize"""
        self.obstacles.append({
            'id': obstacle_id,
            'position': position,
            'size': size,
            'timestamp': time.time()
        })
        
        # Keep only recent obstacles
        current_time = time.time()
        self.obstacles = [o for o in self.obstacles if current_time - o['timestamp'] < 15.0]
        
        self.update()
    
    def paintGL(self):
        """Paint the 3D overlay"""
        # This would normally use OpenGL calls for 3D rendering
        # For now, we'll use a 2D representation
        pass
    
    def paintEvent(self, event):
        """Paint the AI decision overlay (2D fallback)"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(240, 240, 240, 200))
        
        width = self.width()
        height = self.height()
        
        # Draw title
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        font = QFont()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(10, 20, "AI Decision Visualization")
        
        # Draw coordinate system
        center_x = width // 2
        center_y = height // 2
        scale = min(width, height) // 4
        
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        painter.drawLine(center_x - scale, center_y, center_x + scale, center_y)  # X-axis
        painter.drawLine(center_x, center_y - scale, center_x, center_y + scale)  # Y-axis
        
        # Draw obstacles
        painter.setBrush(QBrush(QColor(255, 100, 100, 150)))
        painter.setPen(QPen(QColor(200, 50, 50), 2))
        for obstacle in self.obstacles:
            x, y = obstacle['position']
            size = obstacle['size']
            
            screen_x = center_x + x * scale / 100
            screen_y = center_y - y * scale / 100
            screen_size = size * scale / 100
            
            painter.drawEllipse(int(screen_x - screen_size/2), int(screen_y - screen_size/2), 
                              int(screen_size), int(screen_size))
        
        # Draw planned paths
        painter.setPen(QPen(QColor(100, 150, 255), 3))
        for path in self.paths:
            waypoints = path['waypoints']
            if len(waypoints) > 1:
                for i in range(len(waypoints) - 1):
                    x1, y1 = waypoints[i]
                    x2, y2 = waypoints[i + 1]
                    
                    screen_x1 = center_x + x1 * scale / 100
                    screen_y1 = center_y - y1 * scale / 100
                    screen_x2 = center_x + x2 * scale / 100
                    screen_y2 = center_y - y2 * scale / 100
                    
                    painter.drawLine(int(screen_x1), int(screen_y1), int(screen_x2), int(screen_y2))
        
        # Draw AI decisions
        for decision in self.decisions:
            x, y = decision['position']
            confidence = decision['confidence']
            decision_type = decision['type']
            
            screen_x = center_x + x * scale / 100
            screen_y = center_y - y * scale / 100
            
            # Color based on decision type
            if decision_type == "brake":
                color = QColor(255, 100, 100)
            elif decision_type == "turn":
                color = QColor(100, 255, 100)
            elif decision_type == "accelerate":
                color = QColor(100, 100, 255)
            else:
                color = QColor(150, 150, 150)
            
            # Size based on confidence
            size = 10 + confidence * 20
            
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(), 2))
            painter.drawEllipse(int(screen_x - size/2), int(screen_y - size/2), int(size), int(size))
            
            # Draw decision type text
            painter.setPen(QPen(QColor(50, 50, 50), 1))
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(int(screen_x + size/2 + 5), int(screen_y), decision_type)


class DataVisualizationWidget(QWidget):
    """Main data visualization widget with comprehensive tools"""
    
    # Signals
    metric_updated = pyqtSignal(str, float)
    graph_data_added = pyqtSignal(str, str, float, float)  # graph_name, series_name, timestamp, value
    
    def __init__(self, simulation_app: SimulationApplication):
        super().__init__()
        self.simulation_app = simulation_app
        
        # Data storage
        self.graphs = {}
        self.update_timer = QTimer()
        
        self.setup_ui()
        self.setup_connections()
        self.start_data_collection()
    
    def setup_ui(self):
        """Setup the main UI with tabs for different visualization types"""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Performance Metrics Tab
        self.setup_performance_tab()
        
        # Real-time Graphs Tab
        self.setup_graphs_tab()
        
        # AI Visualization Tab
        self.setup_ai_tab()
        
        # Custom Dashboard Tab
        self.setup_dashboard_tab()
        
        # Controls
        self.setup_controls(layout)
    
    def setup_performance_tab(self):
        """Setup performance metrics tab"""
        perf_widget = QWidget()
        layout = QVBoxLayout(perf_widget)
        
        # Metrics display
        self.metrics_display = MetricsDisplay()
        layout.addWidget(self.metrics_display)
        
        # Performance graphs
        graphs_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # FPS graph
        self.fps_graph = RealTimeGraph("FPS Over Time", max_points=200)
        self.fps_graph.add_series("FPS", QColor(76, 175, 80))
        self.fps_graph.set_y_range(0, 120)
        graphs_splitter.addWidget(self.fps_graph)
        
        # Memory graph
        self.memory_graph = RealTimeGraph("Memory Usage", max_points=200)
        self.memory_graph.add_series("Memory", QColor(255, 152, 0))
        self.memory_graph.set_y_range(0, 1000)
        graphs_splitter.addWidget(self.memory_graph)
        
        layout.addWidget(graphs_splitter)
        
        self.tab_widget.addTab(perf_widget, "Performance")
    
    def setup_graphs_tab(self):
        """Setup real-time graphs tab"""
        graphs_widget = QWidget()
        layout = QVBoxLayout(graphs_widget)
        
        # Graph controls
        controls_layout = QHBoxLayout()
        
        # Graph selection
        controls_layout.addWidget(QLabel("Graph:"))
        self.graph_selector = QComboBox()
        self.graph_selector.addItems(["Vehicle Telemetry", "Sensor Data", "Physics Data"])
        self.graph_selector.currentTextChanged.connect(self.switch_graph)
        controls_layout.addWidget(self.graph_selector)
        
        # Time range control
        controls_layout.addWidget(QLabel("Time Range:"))
        self.time_range_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_range_slider.setRange(10, 300)  # 10 seconds to 5 minutes
        self.time_range_slider.setValue(60)
        self.time_range_slider.valueChanged.connect(self.update_time_range)
        controls_layout.addWidget(self.time_range_slider)
        
        self.time_range_label = QLabel("60s")
        controls_layout.addWidget(self.time_range_label)
        
        controls_layout.addStretch()
        
        # Auto-scale button
        auto_scale_btn = QPushButton("Auto Scale")
        auto_scale_btn.clicked.connect(self.auto_scale_current_graph)
        controls_layout.addWidget(auto_scale_btn)
        
        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_current_graph)
        controls_layout.addWidget(clear_btn)
        
        layout.addLayout(controls_layout)
        
        # Graph area
        self.graph_area = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_area)
        
        # Create different graph types
        self.create_telemetry_graphs()
        
        layout.addWidget(self.graph_area)
        
        self.tab_widget.addTab(graphs_widget, "Real-time Graphs")
    
    def setup_ai_tab(self):
        """Setup AI visualization tab"""
        ai_widget = QWidget()
        layout = QVBoxLayout(ai_widget)
        
        # AI decision overlay
        self.ai_overlay = AIDecisionOverlay()
        layout.addWidget(self.ai_overlay)
        
        # AI metrics
        ai_metrics_layout = QHBoxLayout()
        
        # Decision confidence graph
        self.confidence_graph = RealTimeGraph("Decision Confidence", max_points=100)
        self.confidence_graph.add_series("Confidence", QColor(156, 39, 176))
        self.confidence_graph.set_y_range(0, 1)
        ai_metrics_layout.addWidget(self.confidence_graph)
        
        # Processing time graph
        self.ai_time_graph = RealTimeGraph("AI Processing Time", max_points=100)
        self.ai_time_graph.add_series("Processing", QColor(244, 67, 54))
        self.ai_time_graph.set_y_range(0, 50)
        ai_metrics_layout.addWidget(self.ai_time_graph)
        
        layout.addLayout(ai_metrics_layout)
        
        self.tab_widget.addTab(ai_widget, "AI Visualization")
    
    def setup_dashboard_tab(self):
        """Setup custom dashboard tab"""
        dashboard_widget = QWidget()
        layout = QVBoxLayout(dashboard_widget)
        
        # Dashboard controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Dashboard Layout:"))
        self.layout_selector = QComboBox()
        self.layout_selector.addItems(["Developer", "Analyst", "Instructor", "Custom"])
        self.layout_selector.currentTextChanged.connect(self.switch_dashboard_layout)
        controls_layout.addWidget(self.layout_selector)
        
        controls_layout.addStretch()
        
        save_layout_btn = QPushButton("Save Layout")
        save_layout_btn.clicked.connect(self.save_dashboard_layout)
        controls_layout.addWidget(save_layout_btn)
        
        layout.addLayout(controls_layout)
        
        # Dashboard area
        self.dashboard_area = QSplitter(Qt.Orientation.Vertical)
        
        # Top section - Key metrics
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Mini performance display
        mini_metrics = MetricsDisplay()
        top_splitter.addWidget(mini_metrics)
        
        # Mini FPS graph
        mini_fps_graph = RealTimeGraph("FPS", max_points=50)
        mini_fps_graph.add_series("FPS", QColor(76, 175, 80))
        mini_fps_graph.set_y_range(0, 120)
        top_splitter.addWidget(mini_fps_graph)
        
        self.dashboard_area.addWidget(top_splitter)
        
        # Bottom section - Detailed view
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        
        # Vehicle telemetry
        vehicle_graph = RealTimeGraph("Vehicle Speed", max_points=100)
        vehicle_graph.add_series("Speed", QColor(33, 150, 243))
        bottom_layout.addWidget(vehicle_graph)
        
        # System log
        log_widget = QTextEdit()
        log_widget.setMaximumHeight(150)
        log_widget.setPlainText("System Log:\n")
        bottom_layout.addWidget(log_widget)
        
        self.dashboard_area.addWidget(bottom_widget)
        
        layout.addWidget(self.dashboard_area)
        
        self.tab_widget.addTab(dashboard_widget, "Dashboard")
    
    def setup_controls(self, parent_layout):
        """Setup global controls"""
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_layout = QHBoxLayout(controls_frame)
        
        # Data collection controls
        self.collection_enabled = QCheckBox("Data Collection")
        self.collection_enabled.setChecked(True)
        self.collection_enabled.toggled.connect(self.toggle_data_collection)
        controls_layout.addWidget(self.collection_enabled)
        
        # Update rate control
        controls_layout.addWidget(QLabel("Update Rate:"))
        self.update_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.update_rate_slider.setRange(1, 10)  # 1-10 Hz
        self.update_rate_slider.setValue(5)  # 5 Hz default
        self.update_rate_slider.valueChanged.connect(self.update_collection_rate)
        controls_layout.addWidget(self.update_rate_slider)
        
        self.update_rate_label = QLabel("5 Hz")
        controls_layout.addWidget(self.update_rate_label)
        
        controls_layout.addStretch()
        
        # Export controls
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_data)
        controls_layout.addWidget(export_btn)
        
        parent_layout.addWidget(controls_frame)
    
    def create_telemetry_graphs(self):
        """Create telemetry graphs"""
        # Clear existing graphs
        for i in reversed(range(self.graph_layout.count())):
            self.graph_layout.itemAt(i).widget().setParent(None)
        
        # Vehicle telemetry graphs
        telemetry_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Speed graph
        speed_graph = RealTimeGraph("Vehicle Speed", max_points=150)
        speed_graph.add_series("Speed", QColor(33, 150, 243))
        speed_graph.set_y_range(0, 100)
        telemetry_splitter.addWidget(speed_graph)
        self.graphs["speed"] = speed_graph
        
        # Acceleration graph
        accel_graph = RealTimeGraph("Acceleration", max_points=150)
        accel_graph.add_series("Accel X", QColor(255, 99, 132))
        accel_graph.add_series("Accel Y", QColor(54, 162, 235))
        accel_graph.set_y_range(-10, 10)
        telemetry_splitter.addWidget(accel_graph)
        self.graphs["acceleration"] = accel_graph
        
        self.graph_layout.addWidget(telemetry_splitter)
        
        # Sensor data graphs
        sensor_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Distance sensors
        distance_graph = RealTimeGraph("Distance Sensors", max_points=150)
        distance_graph.add_series("Front", QColor(255, 205, 86))
        distance_graph.add_series("Left", QColor(75, 192, 192))
        distance_graph.add_series("Right", QColor(153, 102, 255))
        distance_graph.set_y_range(0, 50)
        sensor_splitter.addWidget(distance_graph)
        self.graphs["distance"] = distance_graph
        
        # GPS accuracy
        gps_graph = RealTimeGraph("GPS Accuracy", max_points=150)
        gps_graph.add_series("Accuracy", QColor(255, 159, 64))
        gps_graph.set_y_range(0, 10)
        sensor_splitter.addWidget(gps_graph)
        self.graphs["gps"] = gps_graph
        
        self.graph_layout.addWidget(sensor_splitter)
    
    def setup_connections(self):
        """Setup signal connections"""
        # Connect to simulation app signals if available
        if hasattr(self.simulation_app, 'performance_updated'):
            self.simulation_app.performance_updated.connect(self.update_performance_data)
        
        if hasattr(self.simulation_app, 'vehicle_telemetry_updated'):
            self.simulation_app.vehicle_telemetry_updated.connect(self.update_vehicle_telemetry)
        
        if hasattr(self.simulation_app, 'ai_decision_made'):
            self.simulation_app.ai_decision_made.connect(self.update_ai_visualization)
    
    def start_data_collection(self):
        """Start data collection timer"""
        self.update_timer.timeout.connect(self.collect_data)
        self.update_timer.start(200)  # 5 Hz default
    
    def collect_data(self):
        """Collect data from simulation"""
        if not self.collection_enabled.isChecked():
            return
        
        try:
            # Get performance stats
            stats = self.simulation_app.get_performance_stats()
            current_time = time.time()
            
            # Update metrics display
            self.metrics_display.update_metric("FPS", stats.get('fps', 0))
            self.metrics_display.update_metric("Vehicles", stats.get('vehicle_count', 0))
            self.metrics_display.update_metric("Memory", stats.get('memory_mb', 0))
            self.metrics_display.update_metric("Physics", stats.get('physics_time_ms', 0))
            self.metrics_display.update_metric("Render", stats.get('render_time_ms', 0))
            self.metrics_display.update_metric("AI", stats.get('ai_time_ms', 0))
            
            # Update performance graphs
            self.fps_graph.add_data_point("FPS", current_time, stats.get('fps', 0))
            self.memory_graph.add_data_point("Memory", current_time, stats.get('memory_mb', 0))
            
            # Update telemetry graphs with simulated data
            if "speed" in self.graphs:
                # Simulate vehicle speed data
                speed = 30 + 20 * math.sin(current_time * 0.5)
                self.graphs["speed"].add_data_point("Speed", current_time, speed)
            
            if "acceleration" in self.graphs:
                # Simulate acceleration data
                accel_x = 2 * math.sin(current_time * 0.3)
                accel_y = 1.5 * math.cos(current_time * 0.4)
                self.graphs["acceleration"].add_data_point("Accel X", current_time, accel_x)
                self.graphs["acceleration"].add_data_point("Accel Y", current_time, accel_y)
            
            # Update AI visualization with simulated data
            if hasattr(self, 'confidence_graph'):
                confidence = 0.7 + 0.3 * math.sin(current_time * 0.2)
                self.confidence_graph.add_data_point("Confidence", current_time, confidence)
            
            if hasattr(self, 'ai_time_graph'):
                ai_time = 15 + 10 * math.sin(current_time * 0.1)
                self.ai_time_graph.add_data_point("Processing", current_time, ai_time)
            
        except Exception as e:
            print(f"Error collecting data: {e}")
    
    @pyqtSlot(str)
    def switch_graph(self, graph_type: str):
        """Switch between different graph types"""
        # This would switch the visible graphs based on selection
        pass
    
    @pyqtSlot(int)
    def update_time_range(self, value: int):
        """Update time range for graphs"""
        self.time_range_label.setText(f"{value}s")
        
        # Update all graphs
        for graph in self.graphs.values():
            graph.time_range = value
    
    @pyqtSlot()
    def auto_scale_current_graph(self):
        """Auto-scale the current graph"""
        current_tab = self.tab_widget.currentIndex()
        if current_tab == 1:  # Graphs tab
            for graph in self.graphs.values():
                graph.auto_scale_y()
    
    @pyqtSlot()
    def clear_current_graph(self):
        """Clear the current graph"""
        current_tab = self.tab_widget.currentIndex()
        if current_tab == 1:  # Graphs tab
            for graph in self.graphs.values():
                graph.clear_all()
    
    @pyqtSlot(str)
    def switch_dashboard_layout(self, layout_name: str):
        """Switch dashboard layout"""
        # This would reconfigure the dashboard based on user role
        pass
    
    @pyqtSlot()
    def save_dashboard_layout(self):
        """Save current dashboard layout"""
        # This would save the current layout configuration
        pass
    
    @pyqtSlot(bool)
    def toggle_data_collection(self, enabled: bool):
        """Toggle data collection"""
        if enabled:
            self.update_timer.start()
        else:
            self.update_timer.stop()
    
    @pyqtSlot(int)
    def update_collection_rate(self, rate: int):
        """Update data collection rate"""
        self.update_rate_label.setText(f"{rate} Hz")
        interval = 1000 // rate  # Convert Hz to milliseconds
        self.update_timer.setInterval(interval)
    
    @pyqtSlot()
    def export_data(self):
        """Export collected data"""
        # This would export the collected data to CSV or other formats
        pass
    
    # Slot methods for external data updates
    
    @pyqtSlot(dict)
    def update_performance_data(self, data: Dict[str, Any]):
        """Update performance data from external source"""
        current_time = time.time()
        
        for key, value in data.items():
            if key == "fps" and hasattr(self, 'fps_graph'):
                self.fps_graph.add_data_point("FPS", current_time, value)
            elif key == "memory_mb" and hasattr(self, 'memory_graph'):
                self.memory_graph.add_data_point("Memory", current_time, value)
    
    @pyqtSlot(str, dict)
    def update_vehicle_telemetry(self, vehicle_id: str, telemetry: Dict[str, Any]):
        """Update vehicle telemetry data"""
        current_time = time.time()
        
        if "speed" in self.graphs and "speed" in telemetry:
            self.graphs["speed"].add_data_point("Speed", current_time, telemetry["speed"])
        
        if "acceleration" in self.graphs:
            if "accel_x" in telemetry:
                self.graphs["acceleration"].add_data_point("Accel X", current_time, telemetry["accel_x"])
            if "accel_y" in telemetry:
                self.graphs["acceleration"].add_data_point("Accel Y", current_time, telemetry["accel_y"])
    
    @pyqtSlot(str, str, float, tuple)
    def update_ai_visualization(self, vehicle_id: str, decision_type: str, confidence: float, position: Tuple[float, float]):
        """Update AI visualization with decision data"""
        if hasattr(self, 'ai_overlay'):
            self.ai_overlay.add_decision(vehicle_id, decision_type, confidence, position)
        
        if hasattr(self, 'confidence_graph'):
            current_time = time.time()
            self.confidence_graph.add_data_point("Confidence", current_time, confidence)
    
    # Public interface methods
    
    def add_custom_graph(self, name: str, title: str, series_names: List[str]) -> RealTimeGraph:
        """Add a custom graph to the visualization"""
        graph = RealTimeGraph(title, max_points=150)
        
        for i, series_name in enumerate(series_names):
            color = graph.color_palette[i % len(graph.color_palette)]
            graph.add_series(series_name, color)
        
        self.graphs[name] = graph
        return graph
    
    def update_custom_metric(self, name: str, value: float, status: str = "normal"):
        """Update a custom metric"""
        self.metrics_display.update_metric(name, value, status)
    
    def get_current_tab(self) -> str:
        """Get the name of the current tab"""
        tab_names = ["Performance", "Real-time Graphs", "AI Visualization", "Dashboard"]
        return tab_names[self.tab_widget.currentIndex()]
    
    def set_current_tab(self, tab_name: str):
        """Set the current tab by name"""
        tab_names = ["Performance", "Real-time Graphs", "AI Visualization", "Dashboard"]
        if tab_name in tab_names:
            self.tab_widget.setCurrentIndex(tab_names.index(tab_name))