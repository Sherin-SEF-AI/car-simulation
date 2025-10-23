"""
Sensor Dashboard for visualizing vehicle sensor data
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QFrame, QScrollArea, QGridLayout, QGroupBox, QPushButton,
    QComboBox, QProgressBar, QTextEdit, QSlider, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap

import math
import numpy as np
from collections import deque


class LidarVisualization(QWidget):
    """LIDAR data visualization widget"""
    
    def __init__(self):
        super().__init__()
        self.lidar_data = []
        self.range_max = 100  # meters
        self.setMinimumSize(300, 300)
        
    def update_lidar_data(self, data):
        """Update LIDAR data"""
        self.lidar_data = data
        self.update()
    
    def paintEvent(self, event):
        """Paint LIDAR visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = min(center_x, center_y) - 20
        
        # Draw range circles
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for i in range(1, 5):
            circle_radius = radius * i / 4
            painter.drawEllipse(int(center_x - circle_radius), int(center_y - circle_radius),
                              int(circle_radius * 2), int(circle_radius * 2))
        
        # Draw angle lines
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            end_x = center_x + radius * math.cos(rad)
            end_y = center_y + radius * math.sin(rad)
            painter.drawLine(int(center_x), int(center_y), int(end_x), int(end_y))
        
        # Draw LIDAR points
        if self.lidar_data:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            for angle, distance in self.lidar_data:
                if distance > 0 and distance < self.range_max:
                    rad = math.radians(angle)
                    point_radius = radius * distance / self.range_max
                    point_x = center_x + point_radius * math.cos(rad)
                    point_y = center_y + point_radius * math.sin(rad)
                    painter.drawEllipse(int(point_x - 2), int(point_y - 2), 4, 4)
        
        # Draw labels
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(10, 20, f"LIDAR - Range: {self.range_max}m")
        painter.drawText(center_x - 5, 15, "0°")
        painter.drawText(self.width() - 20, center_y, "90°")
        painter.drawText(center_x - 10, self.height() - 5, "180°")
        painter.drawText(5, center_y, "270°")


class CameraView(QWidget):
    """Camera view widget"""
    
    def __init__(self):
        super().__init__()
        self.camera_image = None
        self.detected_objects = []
        self.setMinimumSize(320, 240)
        
    def update_camera_data(self, image, objects=None):
        """Update camera data"""
        self.camera_image = image
        self.detected_objects = objects or []
        self.update()
    
    def paintEvent(self, event):
        """Paint camera view"""
        painter = QPainter(self)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        if self.camera_image:
            # Draw camera image (placeholder)
            painter.fillRect(10, 10, self.width() - 20, self.height() - 20, QColor(100, 100, 100))
        
        # Draw detected objects
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        for obj in self.detected_objects:
            x, y, w, h = obj.get('bbox', (0, 0, 0, 0))
            painter.drawRect(x, y, w, h)
            
            # Draw label
            label = obj.get('label', 'Unknown')
            confidence = obj.get('confidence', 0)
            painter.drawText(x, y - 5, f"{label} ({confidence:.1%})")
        
        # Draw overlay info
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(15, 25, "Camera View")
        painter.drawText(15, self.height() - 15, f"Objects: {len(self.detected_objects)}")


class SensorMetrics(QWidget):
    """Sensor metrics display"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup sensor metrics UI"""
        layout = QGridLayout(self)
        
        # Sensor status indicators
        self.sensors = {
            'Camera': {'status': True, 'quality': 95},
            'LIDAR': {'status': True, 'quality': 98},
            'GPS': {'status': True, 'quality': 87},
            'IMU': {'status': True, 'quality': 92},
            'Ultrasonic': {'status': True, 'quality': 89},
            'Radar': {'status': False, 'quality': 0}
        }
        
        self.status_labels = {}
        self.quality_bars = {}
        
        for i, (sensor_name, data) in enumerate(self.sensors.items()):
            # Sensor name
            name_label = QLabel(sensor_name)
            name_label.setStyleSheet("font-weight: bold;")
            layout.addWidget(name_label, i, 0)
            
            # Status indicator
            status_label = QLabel("●")
            status_label.setStyleSheet(f"color: {'green' if data['status'] else 'red'}; font-size: 16px;")
            layout.addWidget(status_label, i, 1)
            self.status_labels[sensor_name] = status_label
            
            # Quality bar
            quality_bar = QProgressBar()
            quality_bar.setRange(0, 100)
            quality_bar.setValue(data['quality'])
            quality_bar.setMaximumHeight(20)
            layout.addWidget(quality_bar, i, 2)
            self.quality_bars[sensor_name] = quality_bar
            
            # Quality percentage
            quality_label = QLabel(f"{data['quality']}%")
            layout.addWidget(quality_label, i, 3)
    
    def update_sensor_status(self, sensor_name, status, quality):
        """Update sensor status"""
        if sensor_name in self.status_labels:
            self.status_labels[sensor_name].setStyleSheet(
                f"color: {'green' if status else 'red'}; font-size: 16px;"
            )
            self.quality_bars[sensor_name].setValue(quality)


class GPSVisualization(QWidget):
    """GPS visualization widget"""
    
    def __init__(self):
        super().__init__()
        self.gps_data = {'lat': 0, 'lon': 0, 'accuracy': 0}
        self.path_history = deque(maxlen=100)
        self.setMinimumSize(300, 200)
    
    def update_gps_data(self, lat, lon, accuracy):
        """Update GPS data"""
        self.gps_data = {'lat': lat, 'lon': lon, 'accuracy': accuracy}
        self.path_history.append((lat, lon))
        self.update()
    
    def paintEvent(self, event):
        """Paint GPS visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(25, 25, 25))
        
        # Draw grid
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        grid_size = 20
        for x in range(0, self.width(), grid_size):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), grid_size):
            painter.drawLine(0, y, self.width(), y)
        
        # Draw path history
        if len(self.path_history) > 1:
            painter.setPen(QPen(QColor(0, 150, 255), 2))
            points = list(self.path_history)
            
            # Convert GPS coordinates to screen coordinates (simplified)
            center_x, center_y = self.width() // 2, self.height() // 2
            
            for i in range(len(points) - 1):
                # Simplified coordinate conversion
                x1 = center_x + (points[i][1] - self.gps_data['lon']) * 1000
                y1 = center_y - (points[i][0] - self.gps_data['lat']) * 1000
                x2 = center_x + (points[i+1][1] - self.gps_data['lon']) * 1000
                y2 = center_y - (points[i+1][0] - self.gps_data['lat']) * 1000
                
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw current position
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        center_x, center_y = self.width() // 2, self.height() // 2
        painter.drawEllipse(int(center_x - 5), int(center_y - 5), 10, 10)
        
        # Draw accuracy circle
        if self.gps_data['accuracy'] > 0:
            painter.setPen(QPen(QColor(255, 255, 0, 100), 2))
            painter.setBrush(QBrush())
            accuracy_radius = min(50, self.gps_data['accuracy'] * 10)
            painter.drawEllipse(int(center_x - accuracy_radius), int(center_y - accuracy_radius),
                              int(accuracy_radius * 2), int(accuracy_radius * 2))
        
        # Draw info
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(10, 20, f"GPS: {self.gps_data['lat']:.6f}, {self.gps_data['lon']:.6f}")
        painter.drawText(10, 35, f"Accuracy: ±{self.gps_data['accuracy']:.1f}m")


class SensorDashboard(QWidget):
    """Main sensor dashboard widget"""
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        self.selected_vehicle = None
        self.setup_ui()
        self.setup_timers()
        
        print("Sensor dashboard initialized")
    
    def setup_ui(self):
        """Setup sensor dashboard UI"""
        layout = QVBoxLayout(self)
        
        # Vehicle selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Vehicle:"))
        
        self.vehicle_combo = QComboBox()
        self.vehicle_combo.currentTextChanged.connect(self.on_vehicle_selected)
        selector_layout.addWidget(self.vehicle_combo)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_vehicles)
        selector_layout.addWidget(refresh_btn)
        
        layout.addLayout(selector_layout)
        
        # Main content
        tab_widget = QTabWidget()
        
        # Camera tab
        camera_widget = QWidget()
        camera_layout = QVBoxLayout(camera_widget)
        
        self.camera_view = CameraView()
        camera_layout.addWidget(self.camera_view)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        camera_controls.addWidget(QLabel("Exposure:"))
        
        exposure_slider = QSlider(Qt.Orientation.Horizontal)
        exposure_slider.setRange(-100, 100)
        exposure_slider.setValue(0)
        camera_controls.addWidget(exposure_slider)
        
        camera_controls.addWidget(QLabel("Gain:"))
        
        gain_slider = QSlider(Qt.Orientation.Horizontal)
        gain_slider.setRange(0, 100)
        gain_slider.setValue(50)
        camera_controls.addWidget(gain_slider)
        
        camera_layout.addLayout(camera_controls)
        
        tab_widget.addTab(camera_widget, "Camera")
        
        # LIDAR tab
        lidar_widget = QWidget()
        lidar_layout = QVBoxLayout(lidar_widget)
        
        self.lidar_viz = LidarVisualization()
        lidar_layout.addWidget(self.lidar_viz)
        
        # LIDAR controls
        lidar_controls = QHBoxLayout()
        lidar_controls.addWidget(QLabel("Range:"))
        
        self.range_combo = QComboBox()
        self.range_combo.addItems(["50m", "100m", "150m", "200m"])
        self.range_combo.setCurrentText("100m")
        self.range_combo.currentTextChanged.connect(self.change_lidar_range)
        lidar_controls.addWidget(self.range_combo)
        
        lidar_controls.addWidget(QLabel("Resolution:"))
        
        resolution_slider = QSlider(Qt.Orientation.Horizontal)
        resolution_slider.setRange(32, 128)
        resolution_slider.setValue(64)
        lidar_controls.addWidget(resolution_slider)
        
        lidar_layout.addLayout(lidar_controls)
        
        tab_widget.addTab(lidar_widget, "LIDAR")
        
        # GPS tab
        gps_widget = QWidget()
        gps_layout = QVBoxLayout(gps_widget)
        
        self.gps_viz = GPSVisualization()
        gps_layout.addWidget(self.gps_viz)
        
        # GPS info
        gps_info = QTextEdit()
        gps_info.setMaximumHeight(100)
        gps_info.setReadOnly(True)
        gps_layout.addWidget(gps_info)
        self.gps_info = gps_info
        
        tab_widget.addTab(gps_widget, "GPS")
        
        # Sensors overview tab
        overview_widget = QWidget()
        overview_layout = QVBoxLayout(overview_widget)
        
        self.sensor_metrics = SensorMetrics()
        overview_layout.addWidget(self.sensor_metrics)
        
        # Sensor data text
        sensor_data_text = QTextEdit()
        sensor_data_text.setReadOnly(True)
        sensor_data_text.setMaximumHeight(150)
        sensor_data_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 9px;
            }
        """)
        overview_layout.addWidget(sensor_data_text)
        self.sensor_data_text = sensor_data_text
        
        tab_widget.addTab(overview_widget, "Overview")
        
        layout.addWidget(tab_widget)
    
    def setup_timers(self):
        """Setup update timers"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_sensor_data)
        self.update_timer.start(100)  # 10 FPS
        
        self.vehicle_refresh_timer = QTimer()
        self.vehicle_refresh_timer.timeout.connect(self.refresh_vehicles)
        self.vehicle_refresh_timer.start(5000)  # Refresh every 5 seconds
    
    def refresh_vehicles(self):
        """Refresh vehicle list"""
        try:
            current_vehicles = set()
            if hasattr(self.simulation_app, 'vehicle_manager'):
                current_vehicles = set(self.simulation_app.vehicle_manager.vehicles.keys())
            
            # Get current items in combo
            combo_vehicles = set()
            for i in range(self.vehicle_combo.count()):
                combo_vehicles.add(self.vehicle_combo.itemText(i))
            
            # Add new vehicles
            for vehicle_id in current_vehicles - combo_vehicles:
                self.vehicle_combo.addItem(vehicle_id)
            
            # Remove destroyed vehicles
            for vehicle_id in combo_vehicles - current_vehicles:
                index = self.vehicle_combo.findText(vehicle_id)
                if index >= 0:
                    self.vehicle_combo.removeItem(index)
            
        except Exception as e:
            pass  # Ignore refresh errors
    
    def on_vehicle_selected(self, vehicle_id):
        """Handle vehicle selection"""
        self.selected_vehicle = vehicle_id
    
    def change_lidar_range(self, range_text):
        """Change LIDAR range"""
        range_value = int(range_text.replace('m', ''))
        self.lidar_viz.range_max = range_value
    
    def update_sensor_data(self):
        """Update sensor data display"""
        if not self.selected_vehicle:
            return
        
        try:
            # Get vehicle from simulation
            if hasattr(self.simulation_app, 'vehicle_manager'):
                vehicles = self.simulation_app.vehicle_manager.vehicles
                if self.selected_vehicle in vehicles:
                    vehicle = vehicles[self.selected_vehicle]
                    
                    # Update sensor displays with simulated data
                    self.update_simulated_sensor_data(vehicle)
                    
        except Exception as e:
            pass  # Ignore update errors
    
    def update_simulated_sensor_data(self, vehicle):
        """Update with simulated sensor data"""
        import random
        import time
        
        # Simulate LIDAR data
        lidar_data = []
        for angle in range(0, 360, 5):
            distance = random.uniform(5, 95) if random.random() > 0.1 else 0
            lidar_data.append((angle, distance))
        self.lidar_viz.update_lidar_data(lidar_data)
        
        # Simulate camera data
        detected_objects = []
        if random.random() > 0.7:  # 30% chance of detecting objects
            for _ in range(random.randint(1, 3)):
                detected_objects.append({
                    'bbox': (random.randint(50, 200), random.randint(50, 150), 
                            random.randint(30, 80), random.randint(20, 60)),
                    'label': random.choice(['Car', 'Pedestrian', 'Cyclist', 'Sign']),
                    'confidence': random.uniform(0.6, 0.95)
                })
        self.camera_view.update_camera_data(None, detected_objects)
        
        # Simulate GPS data
        if hasattr(vehicle, 'physics') and hasattr(vehicle.physics, 'position'):
            pos = vehicle.physics.position
            # Convert position to GPS coordinates (simplified)
            lat = 40.7128 + pos.x * 0.00001  # Approximate NYC coordinates
            lon = -74.0060 + pos.y * 0.00001
            accuracy = random.uniform(1, 5)
            self.gps_viz.update_gps_data(lat, lon, accuracy)
            
            # Update GPS info
            gps_info = f"""GPS Data for {self.selected_vehicle}:
Latitude: {lat:.6f}°
Longitude: {lon:.6f}°
Altitude: {pos.z:.1f} m
Accuracy: ±{accuracy:.1f} m
Satellites: {random.randint(8, 12)}
HDOP: {random.uniform(0.8, 2.0):.1f}
Speed: {getattr(vehicle.physics, 'speed', 0) * 3.6:.1f} km/h
Heading: {random.uniform(0, 360):.1f}°
"""
            self.gps_info.setText(gps_info)
        
        # Update sensor metrics
        for sensor_name in ['Camera', 'LIDAR', 'GPS', 'IMU', 'Ultrasonic']:
            status = random.random() > 0.05  # 95% uptime
            quality = random.randint(85, 100) if status else 0
            self.sensor_metrics.update_sensor_status(sensor_name, status, quality)
        
        # Update sensor data text
        sensor_text = f"""[{time.strftime('%H:%M:%S')}] Sensor Data Update
Vehicle: {self.selected_vehicle}
LIDAR: {len([d for d in lidar_data if d[1] > 0])} points detected
Camera: {len(detected_objects)} objects detected
GPS: Signal strength {random.randint(70, 100)}%
IMU: Accel({random.uniform(-2, 2):.2f}, {random.uniform(-2, 2):.2f}, {random.uniform(8, 12):.2f}) m/s²
      Gyro({random.uniform(-0.5, 0.5):.3f}, {random.uniform(-0.5, 0.5):.3f}, {random.uniform(-0.5, 0.5):.3f}) rad/s
Ultrasonic: FL:{random.uniform(0.5, 5):.1f}m FR:{random.uniform(0.5, 5):.1f}m RL:{random.uniform(0.5, 5):.1f}m RR:{random.uniform(0.5, 5):.1f}m
"""
        self.sensor_data_text.setText(sensor_text)