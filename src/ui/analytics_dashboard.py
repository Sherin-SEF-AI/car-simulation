"""
Comprehensive Analytics Dashboard
Real-time data visualization and analysis for the simulation
"""

import sys
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QFrame, QScrollArea, QGridLayout, QGroupBox, QPushButton,
    QComboBox, QSpinBox, QCheckBox, QSlider, QTextEdit,
    QSplitter, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont

import numpy as np
import time
from collections import deque
from typing import Dict, List, Any, Optional


class RealTimeChart(QWidget):
    """Real-time chart widget for displaying time-series data"""
    
    def __init__(self, title="Chart", max_points=100, y_range=(0, 100)):
        super().__init__()
        self.title = title
        self.max_points = max_points
        self.y_range = y_range
        
        # Data storage
        self.data_series = {}  # name -> deque of values
        self.colors = {}       # name -> color
        self.time_points = deque(maxlen=max_points)
        
        # Chart properties
        self.margin = 40
        self.grid_color = QColor(60, 60, 60)
        self.text_color = QColor(255, 255, 255)
        self.background_color = QColor(30, 30, 30)
        
        self.setMinimumSize(300, 200)
        
        # Auto-update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(100)  # 10 FPS
    
    def add_series(self, name: str, color: QColor):
        """Add a new data series"""
        self.data_series[name] = deque(maxlen=self.max_points)
        self.colors[name] = color
    
    def add_data_point(self, series_name: str, value: float, timestamp: float = None):
        """Add a data point to a series"""
        if timestamp is None:
            timestamp = time.time()
        
        if series_name in self.data_series:
            self.data_series[series_name].append(value)
            
            # Update time points (only once per timestamp)
            if not self.time_points or timestamp > self.time_points[-1]:
                self.time_points.append(timestamp)
    
    def paintEvent(self, event):
        """Paint the chart"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), self.background_color)
        
        # Draw title
        painter.setPen(self.text_color)
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(10, 20, self.title)
        
        # Calculate chart area
        chart_rect = self.rect().adjusted(self.margin, self.margin, -self.margin, -self.margin)
        
        if chart_rect.width() <= 0 or chart_rect.height() <= 0:
            return
        
        # Draw grid
        self.draw_grid(painter, chart_rect)
        
        # Draw data series
        self.draw_data_series(painter, chart_rect)
        
        # Draw legend
        self.draw_legend(painter)
    
    def draw_grid(self, painter, chart_rect):
        """Draw chart grid"""
        painter.setPen(QPen(self.grid_color, 1))
        
        # Vertical grid lines
        for i in range(5):
            x = chart_rect.left() + (chart_rect.width() * i / 4)
            painter.drawLine(int(x), chart_rect.top(), int(x), chart_rect.bottom())
        
        # Horizontal grid lines
        for i in range(5):
            y = chart_rect.top() + (chart_rect.height() * i / 4)
            painter.drawLine(chart_rect.left(), int(y), chart_rect.right(), int(y))
        
        # Draw axes labels
        painter.setPen(self.text_color)
        font = QFont("Arial", 8)
        painter.setFont(font)
        
        # Y-axis labels
        for i in range(5):
            y = chart_rect.top() + (chart_rect.height() * i / 4)
            value = self.y_range[1] - ((self.y_range[1] - self.y_range[0]) * i / 4)
            painter.drawText(5, int(y + 3), f"{value:.1f}")
    
    def draw_data_series(self, painter, chart_rect):
        """Draw all data series"""
        if not self.time_points or len(self.time_points) < 2:
            return
        
        for series_name, data in self.data_series.items():
            if len(data) < 2:
                continue
            
            color = self.colors.get(series_name, QColor(255, 255, 255))
            painter.setPen(QPen(color, 2))
            
            # Convert data to screen coordinates
            points = []
            for i, value in enumerate(data):
                if i >= len(self.time_points):
                    break
                
                x = chart_rect.left() + (chart_rect.width() * i / max(1, len(data) - 1))
                y_norm = (value - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
                y_norm = max(0, min(1, y_norm))  # Clamp to [0, 1]
                y = chart_rect.bottom() - (chart_rect.height() * y_norm)
                
                points.append((x, y))
            
            # Draw lines between points
            for i in range(len(points) - 1):
                painter.drawLine(int(points[i][0]), int(points[i][1]), int(points[i+1][0]), int(points[i+1][1]))
    
    def draw_legend(self, painter):
        """Draw legend"""
        if not self.data_series:
            return
        
        painter.setPen(self.text_color)
        font = QFont("Arial", 9)
        painter.setFont(font)
        
        y_offset = 40
        for series_name, color in self.colors.items():
            # Draw color indicator
            painter.fillRect(self.width() - 120, y_offset - 5, 10, 10, color)
            
            # Draw series name with current value
            current_value = 0
            if series_name in self.data_series and self.data_series[series_name]:
                current_value = self.data_series[series_name][-1]
            
            text = f"{series_name}: {current_value:.1f}"
            painter.drawText(self.width() - 105, y_offset + 3, text)
            y_offset += 20


class MetricsWidget(QWidget):
    """Widget displaying key metrics"""
    
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup metrics display"""
        layout = QGridLayout(self)
        
        # Create metric displays
        self.metric_labels = {}
        self.metric_values = {}
        
        metrics = [
            ("Total Vehicles", "0"),
            ("Active Vehicles", "0"),
            ("Average Speed", "0.0 km/h"),
            ("Total Distance", "0.0 km"),
            ("Collisions", "0"),
            ("Safety Score", "100%"),
            ("Fuel Efficiency", "0.0 L/100km"),
            ("AI Success Rate", "0%")
        ]
        
        for i, (name, default_value) in enumerate(metrics):
            row = i // 2
            col = (i % 2) * 2
            
            label = QLabel(f"{name}:")
            label.setStyleSheet("font-weight: bold; color: #cccccc;")
            layout.addWidget(label, row, col)
            
            value_label = QLabel(default_value)
            value_label.setStyleSheet("color: #4a90e2; font-size: 14px; font-weight: bold;")
            layout.addWidget(value_label, row, col + 1)
            
            self.metric_labels[name] = label
            self.metric_values[name] = value_label
    
    def update_metric(self, name: str, value: str):
        """Update a metric value"""
        if name in self.metric_values:
            self.metric_values[name].setText(value)


class VehicleAnalysisWidget(QWidget):
    """Widget for analyzing individual vehicle performance"""
    
    def __init__(self):
        super().__init__()
        self.selected_vehicle = None
        self.vehicle_data = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup vehicle analysis UI"""
        layout = QVBoxLayout(self)
        
        # Vehicle selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Vehicle:"))
        
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.currentTextChanged.connect(self.on_vehicle_selected)
        selector_layout.addWidget(self.vehicle_combo)
        
        layout.addLayout(selector_layout)
        
        # Vehicle info display
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        
        # Vehicle-specific charts
        self.vehicle_chart = RealTimeChart("Vehicle Performance", max_points=50, y_range=(0, 100))
        self.vehicle_chart.add_series("Speed", QColor(74, 144, 226))
        self.vehicle_chart.add_series("Acceleration", QColor(255, 165, 0))
        self.vehicle_chart.add_series("Steering", QColor(50, 205, 50))
        layout.addWidget(self.vehicle_chart)
    
    def add_vehicle(self, vehicle_id: str):
        """Add a vehicle to the analysis"""
        if vehicle_id not in [self.vehicle_combo.itemText(i) for i in range(self.vehicle_combo.count())]:
            self.vehicle_combo.addItem(vehicle_id)
            self.vehicle_data[vehicle_id] = {
                'speed_history': deque(maxlen=50),
                'position_history': deque(maxlen=50),
                'last_update': time.time()
            }
    
    def remove_vehicle(self, vehicle_id: str):
        """Remove a vehicle from analysis"""
        index = self.vehicle_combo.findText(vehicle_id)
        if index >= 0:
            self.vehicle_combo.removeItem(index)
        if vehicle_id in self.vehicle_data:
            del self.vehicle_data[vehicle_id]
    
    def update_vehicle_data(self, vehicle_id: str, data: Dict[str, Any]):
        """Update vehicle data"""
        if vehicle_id in self.vehicle_data:
            self.vehicle_data[vehicle_id].update(data)
            
            if vehicle_id == self.selected_vehicle:
                self.update_display()
    
    def on_vehicle_selected(self, vehicle_id: str):
        """Handle vehicle selection"""
        self.selected_vehicle = vehicle_id
        self.update_display()
    
    def update_display(self):
        """Update the display for selected vehicle"""
        if not self.selected_vehicle or self.selected_vehicle not in self.vehicle_data:
            return
        
        data = self.vehicle_data[self.selected_vehicle]
        
        # Update info text
        info = f"Vehicle ID: {self.selected_vehicle}\n"
        info += f"Type: {data.get('type', 'Unknown')}\n"
        info += f"Position: ({data.get('x', 0):.1f}, {data.get('y', 0):.1f}, {data.get('z', 0):.1f})\n"
        info += f"Speed: {data.get('speed', 0):.1f} km/h\n"
        info += f"Autonomous: {'Yes' if data.get('autonomous', False) else 'No'}\n"
        info += f"Last Update: {time.strftime('%H:%M:%S', time.localtime(data.get('last_update', 0)))}"
        
        self.info_text.setText(info)
        
        # Update charts
        current_time = time.time()
        self.vehicle_chart.add_data_point("Speed", data.get('speed', 0), current_time)
        self.vehicle_chart.add_data_point("Acceleration", data.get('acceleration', 0), current_time)
        self.vehicle_chart.add_data_point("Steering", data.get('steering_angle', 0), current_time)


class AnalyticsDashboard(QWidget):
    """Main analytics dashboard widget"""
    
    # Signals
    data_exported = pyqtSignal(str)  # filename
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        
        # Data collection
        self.collecting_data = False
        self.data_history = {
            'timestamps': deque(maxlen=1000),
            'vehicle_count': deque(maxlen=1000),
            'average_speed': deque(maxlen=1000),
            'total_distance': deque(maxlen=1000),
            'collision_count': deque(maxlen=1000),
            'fps': deque(maxlen=1000)
        }
        
        self.setup_ui()
        self.setup_timers()
        
        print("Analytics dashboard initialized")
    
    def setup_ui(self):
        """Setup the analytics dashboard UI"""
        layout = QVBoxLayout(self)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Main content area
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)
        
        # Left side - Charts
        charts_widget = self.create_charts_widget()
        main_splitter.addWidget(charts_widget)
        
        # Right side - Metrics and vehicle analysis
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([600, 400])
    
    def create_control_panel(self):
        """Create analytics control panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumHeight(60)
        
        layout = QHBoxLayout(panel)
        
        # Data collection controls
        self.collect_btn = QPushButton("Start Collection")
        self.collect_btn.clicked.connect(self.toggle_data_collection)
        layout.addWidget(self.collect_btn)
        
        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self.export_data)
        layout.addWidget(self.export_btn)
        
        self.clear_btn = QPushButton("Clear Data")
        self.clear_btn.clicked.connect(self.clear_data)
        layout.addWidget(self.clear_btn)
        
        layout.addStretch()
        
        # Time range selector
        layout.addWidget(QLabel("Time Range:"))
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["1 minute", "5 minutes", "15 minutes", "1 hour", "All"])
        layout.addWidget(self.time_range_combo)
        
        # Update frequency
        layout.addWidget(QLabel("Update Rate:"))
        self.update_rate_combo = QComboBox()
        self.update_rate_combo.addItems(["10 Hz", "5 Hz", "1 Hz", "0.5 Hz"])
        self.update_rate_combo.currentTextChanged.connect(self.change_update_rate)
        layout.addWidget(self.update_rate_combo)
        
        return panel
    
    def create_charts_widget(self):
        """Create charts widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create tab widget for different chart categories
        tab_widget = QTabWidget()
        
        # Performance tab
        perf_widget = QWidget()
        perf_layout = QVBoxLayout(perf_widget)
        
        # FPS and performance chart
        self.fps_chart = RealTimeChart("Performance Metrics", max_points=100, y_range=(0, 120))
        self.fps_chart.add_series("FPS", QColor(74, 144, 226))
        self.fps_chart.add_series("Physics Time (ms)", QColor(255, 165, 0))
        self.fps_chart.add_series("Render Time (ms)", QColor(50, 205, 50))
        perf_layout.addWidget(self.fps_chart)
        
        tab_widget.addTab(perf_widget, "Performance")
        
        # Vehicle metrics tab
        vehicle_widget = QWidget()
        vehicle_layout = QVBoxLayout(vehicle_widget)
        
        # Vehicle count and speed chart
        self.vehicle_chart = RealTimeChart("Vehicle Metrics", max_points=100, y_range=(0, 50))
        self.vehicle_chart.add_series("Vehicle Count", QColor(255, 99, 132))
        self.vehicle_chart.add_series("Average Speed", QColor(54, 162, 235))
        vehicle_layout.addWidget(self.vehicle_chart)
        
        # Safety metrics chart
        self.safety_chart = RealTimeChart("Safety Metrics", max_points=100, y_range=(0, 10))
        self.safety_chart.add_series("Collisions", QColor(255, 0, 0))
        self.safety_chart.add_series("Near Misses", QColor(255, 165, 0))
        vehicle_layout.addWidget(self.safety_chart)
        
        tab_widget.addTab(vehicle_widget, "Vehicles")
        
        # Environment tab
        env_widget = QWidget()
        env_layout = QVBoxLayout(env_widget)
        
        # Environment metrics chart
        self.env_chart = RealTimeChart("Environment", max_points=100, y_range=(-10, 40))
        self.env_chart.add_series("Temperature (°C)", QColor(255, 99, 71))
        self.env_chart.add_series("Visibility (km)", QColor(135, 206, 235))
        env_layout.addWidget(self.env_chart)
        
        tab_widget.addTab(env_widget, "Environment")
        
        layout.addWidget(tab_widget)
        return widget
    
    def create_right_panel(self):
        """Create right panel with metrics and vehicle analysis"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Key metrics
        metrics_group = QGroupBox("Key Metrics")
        self.metrics_widget = MetricsWidget()
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.addWidget(self.metrics_widget)
        layout.addWidget(metrics_group)
        
        # Vehicle analysis
        vehicle_group = QGroupBox("Vehicle Analysis")
        self.vehicle_analysis = VehicleAnalysisWidget()
        vehicle_layout = QVBoxLayout(vehicle_group)
        vehicle_layout.addWidget(self.vehicle_analysis)
        layout.addWidget(vehicle_group)
        
        return widget
    
    def setup_timers(self):
        """Setup update timers"""
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.collect_data)
        
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(100)  # 10 FPS UI updates
    
    def start_data_collection(self):
        """Start collecting data"""
        self.collecting_data = True
        self.data_timer.start(100)  # 10 Hz by default
        self.collect_btn.setText("Stop Collection")
        print("Started data collection")
    
    def stop_data_collection(self):
        """Stop collecting data"""
        self.collecting_data = False
        self.data_timer.stop()
        self.collect_btn.setText("Start Collection")
        print("Stopped data collection")
    
    def toggle_data_collection(self):
        """Toggle data collection"""
        if self.collecting_data:
            self.stop_data_collection()
        else:
            self.start_data_collection()
    
    def change_update_rate(self, rate_text):
        """Change data collection update rate"""
        rates = {"10 Hz": 100, "5 Hz": 200, "1 Hz": 1000, "0.5 Hz": 2000}
        interval = rates.get(rate_text, 100)
        
        if self.collecting_data:
            self.data_timer.setInterval(interval)
    
    def collect_data(self):
        """Collect data from simulation"""
        if not self.collecting_data:
            return
        
        try:
            current_time = time.time()
            self.data_history['timestamps'].append(current_time)
            
            # Collect performance data
            if hasattr(self.simulation_app, 'get_performance_stats'):
                stats = self.simulation_app.get_performance_stats()
                
                self.data_history['fps'].append(stats.get('fps', 0))
                
                # Add to charts
                self.fps_chart.add_data_point("FPS", stats.get('fps', 0), current_time)
                self.fps_chart.add_data_point("Physics Time (ms)", stats.get('physics_time_ms', 0), current_time)
                self.fps_chart.add_data_point("Render Time (ms)", stats.get('render_time_ms', 0), current_time)
            
            # Collect vehicle data
            vehicle_count = 0
            total_speed = 0
            active_vehicles = 0
            
            if hasattr(self.simulation_app, 'vehicle_manager'):
                vehicles = self.simulation_app.vehicle_manager.vehicles
                vehicle_count = len(vehicles)
                
                for vehicle_id, vehicle in vehicles.items():
                    if hasattr(vehicle, 'physics'):
                        speed = getattr(vehicle.physics, 'speed', 0)
                        total_speed += speed
                        active_vehicles += 1
                        
                        # Update individual vehicle analysis
                        self.vehicle_analysis.update_vehicle_data(vehicle_id, {
                            'speed': speed * 3.6,  # Convert to km/h
                            'x': getattr(vehicle.physics.position, 'x', 0),
                            'y': getattr(vehicle.physics.position, 'y', 0),
                            'z': getattr(vehicle.physics.position, 'z', 0),
                            'type': getattr(vehicle, 'vehicle_type', 'unknown'),
                            'autonomous': getattr(vehicle, 'is_autonomous', False),
                            'last_update': current_time
                        })
            
            avg_speed = (total_speed / active_vehicles * 3.6) if active_vehicles > 0 else 0  # km/h
            
            self.data_history['vehicle_count'].append(vehicle_count)
            self.data_history['average_speed'].append(avg_speed)
            
            # Add to charts
            self.vehicle_chart.add_data_point("Vehicle Count", vehicle_count, current_time)
            self.vehicle_chart.add_data_point("Average Speed", avg_speed, current_time)
            
            # Collect environment data
            if hasattr(self.simulation_app, 'environment'):
                env = self.simulation_app.environment
                temp = getattr(env, 'temperature', 20)
                visibility = getattr(env, 'visibility', 10)
                
                self.env_chart.add_data_point("Temperature (°C)", temp, current_time)
                self.env_chart.add_data_point("Visibility (km)", visibility, current_time)
            
        except Exception as e:
            print(f"Error collecting data: {e}")
    
    def update_ui(self):
        """Update UI elements"""
        try:
            # Update metrics
            if self.data_history['vehicle_count']:
                current_vehicles = self.data_history['vehicle_count'][-1]
                self.metrics_widget.update_metric("Active Vehicles", str(current_vehicles))
            
            if self.data_history['average_speed']:
                avg_speed = self.data_history['average_speed'][-1]
                self.metrics_widget.update_metric("Average Speed", f"{avg_speed:.1f} km/h")
            
            if self.data_history['fps']:
                current_fps = self.data_history['fps'][-1]
                # Update FPS metric if it exists
                pass
            
        except Exception as e:
            pass  # Ignore UI update errors
    
    def clear_data(self):
        """Clear all collected data"""
        for key in self.data_history:
            self.data_history[key].clear()
        
        # Clear charts
        for chart in [self.fps_chart, self.vehicle_chart, self.safety_chart, self.env_chart]:
            for series_name in chart.data_series:
                chart.data_series[series_name].clear()
            chart.time_points.clear()
        
        print("Data cleared")
    
    def export_data(self):
        """Export collected data"""
        try:
            import json
            from PyQt6.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Analytics Data", "", 
                "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)"
            )
            
            if filename:
                # Convert deques to lists for JSON serialization
                export_data = {}
                for key, deque_data in self.data_history.items():
                    export_data[key] = list(deque_data)
                
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2)
                elif filename.endswith('.csv'):
                    import csv
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        
                        # Write header
                        headers = list(export_data.keys())
                        writer.writerow(headers)
                        
                        # Write data rows
                        max_len = max(len(data) for data in export_data.values()) if export_data else 0
                        for i in range(max_len):
                            row = []
                            for header in headers:
                                data = export_data[header]
                                row.append(data[i] if i < len(data) else '')
                            writer.writerow(row)
                
                self.data_exported.emit(filename)
                print(f"Data exported to {filename}")
                
        except Exception as e:
            print(f"Error exporting data: {e}")
    
    @pyqtSlot(str)
    def on_vehicle_event(self, vehicle_id):
        """Handle vehicle events"""
        self.vehicle_analysis.add_vehicle(vehicle_id)