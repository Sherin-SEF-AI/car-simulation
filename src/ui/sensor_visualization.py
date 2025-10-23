"""
Real-time sensor data visualization components
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QProgressBar, QTextEdit, QTabWidget, QFrame,
                            QGridLayout, QScrollArea, QGroupBox)
from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap
import pyqtgraph as pg
import numpy as np
from typing import Dict, Any, List, Optional
import math

from ..core.sensor_simulation import (SensorReading, CameraReading, LidarReading, 
                                     UltrasonicReading, GPSReading, IMUReading,
                                     Vector3, SensorManager)

class SensorVisualizationWidget(QWidget):
    """Main widget for sensor data visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sensor_managers: Dict[str, SensorManager] = {}
        self.current_vehicle_id: Optional[str] = None
        
        self.setup_ui()
        self.setup_update_timer()
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        
        # Vehicle selector (placeholder for now)
        self.vehicle_label = QLabel("Vehicle: None Selected")
        self.vehicle_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.vehicle_label)
        
        # Tab widget for different sensor types
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Camera tab
        self.camera_widget = CameraVisualizationWidget()
        self.tab_widget.addTab(self.camera_widget, "Camera")
        
        # LIDAR tab
        self.lidar_widget = LidarVisualizationWidget()
        self.tab_widget.addTab(self.lidar_widget, "LIDAR")
        
        # Ultrasonic tab
        self.ultrasonic_widget = UltrasonicVisualizationWidget()
        self.tab_widget.addTab(self.ultrasonic_widget, "Ultrasonic")
        
        # GPS tab
        self.gps_widget = GPSVisualizationWidget()
        self.tab_widget.addTab(self.gps_widget, "GPS")
        
        # IMU tab
        self.imu_widget = IMUVisualizationWidget()
        self.tab_widget.addTab(self.imu_widget, "IMU")
        
        # Sensor status tab
        self.status_widget = SensorStatusWidget()
        self.tab_widget.addTab(self.status_widget, "Status")
    
    def setup_update_timer(self):
        """Setup timer for updating visualizations"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualizations)
        self.update_timer.start(50)  # 20 FPS update rate
    
    def add_sensor_manager(self, vehicle_id: str, sensor_manager: SensorManager):
        """Add a sensor manager for a vehicle"""
        self.sensor_managers[vehicle_id] = sensor_manager
        if self.current_vehicle_id is None:
            self.set_current_vehicle(vehicle_id)
    
    def remove_sensor_manager(self, vehicle_id: str):
        """Remove sensor manager for a vehicle"""
        if vehicle_id in self.sensor_managers:
            del self.sensor_managers[vehicle_id]
            if self.current_vehicle_id == vehicle_id:
                # Switch to another vehicle or clear
                if self.sensor_managers:
                    self.set_current_vehicle(next(iter(self.sensor_managers.keys())))
                else:
                    self.set_current_vehicle(None)
    
    def set_current_vehicle(self, vehicle_id: Optional[str]):
        """Set the currently displayed vehicle"""
        self.current_vehicle_id = vehicle_id
        if vehicle_id:
            self.vehicle_label.setText(f"Vehicle: {vehicle_id}")
        else:
            self.vehicle_label.setText("Vehicle: None Selected")
    
    def update_visualizations(self):
        """Update all sensor visualizations"""
        if not self.current_vehicle_id or self.current_vehicle_id not in self.sensor_managers:
            return
        
        sensor_manager = self.sensor_managers[self.current_vehicle_id]
        readings = sensor_manager.get_all_latest_readings()
        
        # Update each visualization widget
        self.camera_widget.update_data(readings)
        self.lidar_widget.update_data(readings)
        self.ultrasonic_widget.update_data(readings)
        self.gps_widget.update_data(readings)
        self.imu_widget.update_data(readings)
        self.status_widget.update_data(sensor_manager.get_sensor_status())

class CameraVisualizationWidget(QWidget):
    """Widget for visualizing camera sensor data"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.detected_objects = []
        self.lane_lines = []
        self.traffic_signs = []
        self.traffic_lights = []
    
    def setup_ui(self):
        """Setup camera visualization UI"""
        layout = QVBoxLayout(self)
        
        # Image display area (simulated)
        self.image_frame = QFrame()
        self.image_frame.setFixedSize(640, 480)
        self.image_frame.setStyleSheet("border: 2px solid black; background-color: #2b2b2b;")
        layout.addWidget(self.image_frame)
        
        # Detection info
        info_layout = QHBoxLayout()
        
        # Objects detected
        objects_group = QGroupBox("Detected Objects")
        objects_layout = QVBoxLayout(objects_group)
        self.objects_text = QTextEdit()
        self.objects_text.setMaximumHeight(150)
        self.objects_text.setReadOnly(True)
        objects_layout.addWidget(self.objects_text)
        info_layout.addWidget(objects_group)
        
        # Lane detection
        lanes_group = QGroupBox("Lane Detection")
        lanes_layout = QVBoxLayout(lanes_group)
        self.lanes_text = QTextEdit()
        self.lanes_text.setMaximumHeight(150)
        self.lanes_text.setReadOnly(True)
        lanes_layout.addWidget(self.lanes_text)
        info_layout.addWidget(lanes_group)
        
        layout.addLayout(info_layout)
        
        # Image properties
        props_layout = QHBoxLayout()
        
        self.brightness_label = QLabel("Brightness: --")
        self.brightness_bar = QProgressBar()
        self.brightness_bar.setRange(0, 100)
        props_layout.addWidget(QLabel("Brightness:"))
        props_layout.addWidget(self.brightness_bar)
        
        self.contrast_label = QLabel("Contrast: --")
        self.contrast_bar = QProgressBar()
        self.contrast_bar.setRange(0, 100)
        props_layout.addWidget(QLabel("Contrast:"))
        props_layout.addWidget(self.contrast_bar)
        
        layout.addLayout(props_layout)
    
    def update_data(self, readings: Dict[str, SensorReading]):
        """Update camera visualization with new data"""
        camera_readings = {k: v for k, v in readings.items() 
                          if isinstance(v, CameraReading)}
        
        if not camera_readings:
            return
        
        # Use first camera reading
        reading = next(iter(camera_readings.values()))
        
        # Update detected objects
        objects_text = ""
        for obj in reading.detected_objects:
            objects_text += f"Type: {obj['type']}\n"
            objects_text += f"Distance: {obj['distance']:.1f}m\n"
            objects_text += f"Confidence: {obj['confidence']:.2f}\n\n"
        
        self.objects_text.setPlainText(objects_text)
        
        # Update lane detection
        lanes_text = ""
        for lane in reading.lane_lines:
            lanes_text += f"Side: {lane['side']}\n"
            lanes_text += f"Type: {lane['type']}\n"
            lanes_text += f"Confidence: {lane['confidence']:.2f}\n\n"
        
        self.lanes_text.setPlainText(lanes_text)
        
        # Update image properties
        self.brightness_bar.setValue(int(reading.brightness * 100))
        self.contrast_bar.setValue(int(reading.contrast * 100))
        
        # Store data for custom painting
        self.detected_objects = reading.detected_objects
        self.lane_lines = reading.lane_lines
        self.traffic_signs = reading.traffic_signs
        self.traffic_lights = reading.traffic_lights
        
        # Trigger repaint of image frame
        self.image_frame.update()
    
    def paintEvent(self, event):
        """Custom paint event for drawing detection overlays"""
        super().paintEvent(event)
        
        painter = QPainter(self.image_frame)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw detected objects as bounding boxes
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        for obj in self.detected_objects:
            bbox = obj.get('bounding_box', {})
            if bbox:
                x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('width', 0), bbox.get('height', 0)
                # Scale to widget size
                scale_x = self.image_frame.width() / 1920
                scale_y = self.image_frame.height() / 1080
                
                painter.drawRect(int(x * scale_x), int(y * scale_y), 
                               int(w * scale_x), int(h * scale_y))
                
                # Draw label
                painter.drawText(int(x * scale_x), int(y * scale_y) - 5, 
                               f"{obj['type']} ({obj['confidence']:.2f})")
        
        # Draw lane lines
        painter.setPen(QPen(QColor(255, 255, 0), 3))
        for lane in self.lane_lines:
            points = lane.get('points', [])
            if len(points) > 1:
                scale_x = self.image_frame.width() / 1920
                scale_y = self.image_frame.height() / 1080
                
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    painter.drawLine(int(x1 * scale_x), int(y1 * scale_y),
                                   int(x2 * scale_x), int(y2 * scale_y))

class LidarVisualizationWidget(QWidget):
    """Widget for visualizing LIDAR sensor data"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.point_cloud = []
        self.distances = []
        self.angles = []
    
    def setup_ui(self):
        """Setup LIDAR visualization UI"""
        layout = QVBoxLayout(self)
        
        # Point cloud plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Y Position (m)')
        self.plot_widget.setLabel('bottom', 'X Position (m)')
        self.plot_widget.setTitle('LIDAR Point Cloud')
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(True, True)
        
        # Set plot range
        self.plot_widget.setXRange(-50, 50)
        self.plot_widget.setYRange(-50, 50)
        
        layout.addWidget(self.plot_widget)
        
        # Distance histogram
        self.distance_plot = pg.PlotWidget()
        self.distance_plot.setLabel('left', 'Count')
        self.distance_plot.setLabel('bottom', 'Distance (m)')
        self.distance_plot.setTitle('Distance Distribution')
        self.distance_plot.setMaximumHeight(200)
        
        layout.addWidget(self.distance_plot)
        
        # Statistics
        stats_layout = QHBoxLayout()
        
        self.stats_label = QLabel("Points: 0 | Min: --m | Max: --m | Avg: --m")
        stats_layout.addWidget(self.stats_label)
        
        layout.addLayout(stats_layout)
    
    def update_data(self, readings: Dict[str, SensorReading]):
        """Update LIDAR visualization with new data"""
        lidar_readings = {k: v for k, v in readings.items() 
                         if isinstance(v, LidarReading)}
        
        if not lidar_readings:
            return
        
        # Use first LIDAR reading
        reading = next(iter(lidar_readings.values()))
        
        # Update point cloud
        if reading.point_cloud:
            x_data = [p.x for p in reading.point_cloud]
            y_data = [p.y for p in reading.point_cloud]
            
            self.plot_widget.clear()
            scatter = pg.ScatterPlotItem(x=x_data, y=y_data, 
                                       pen=pg.mkPen(None), 
                                       brush=pg.mkBrush(255, 0, 0, 120),
                                       size=3)
            self.plot_widget.addItem(scatter)
        
        # Update distance histogram
        if reading.distances:
            hist, bins = np.histogram(reading.distances, bins=50)
            self.distance_plot.clear()
            self.distance_plot.plot(bins[:-1], hist, stepMode=True, 
                                  fillLevel=0, brush=(0, 0, 255, 100))
        
        # Update statistics
        if reading.distances:
            min_dist = min(reading.distances)
            max_dist = max(reading.distances)
            avg_dist = sum(reading.distances) / len(reading.distances)
            
            self.stats_label.setText(
                f"Points: {len(reading.distances)} | "
                f"Min: {min_dist:.1f}m | "
                f"Max: {max_dist:.1f}m | "
                f"Avg: {avg_dist:.1f}m"
            )

class UltrasonicVisualizationWidget(QWidget):
    """Widget for visualizing ultrasonic sensor data"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.sensor_data = {}
    
    def setup_ui(self):
        """Setup ultrasonic visualization UI"""
        layout = QVBoxLayout(self)
        
        # Sensor layout visualization
        self.sensor_frame = QFrame()
        self.sensor_frame.setFixedSize(400, 300)
        self.sensor_frame.setStyleSheet("border: 2px solid black; background-color: white;")
        layout.addWidget(self.sensor_frame)
        
        # Sensor readings table
        self.readings_scroll = QScrollArea()
        self.readings_widget = QWidget()
        self.readings_layout = QVBoxLayout(self.readings_widget)
        self.readings_scroll.setWidget(self.readings_widget)
        self.readings_scroll.setMaximumHeight(200)
        layout.addWidget(self.readings_scroll)
    
    def update_data(self, readings: Dict[str, SensorReading]):
        """Update ultrasonic visualization with new data"""
        ultrasonic_readings = {k: v for k, v in readings.items() 
                             if isinstance(v, UltrasonicReading)}
        
        self.sensor_data = ultrasonic_readings
        
        # Clear previous readings display
        for i in reversed(range(self.readings_layout.count())):
            self.readings_layout.itemAt(i).widget().setParent(None)
        
        # Add sensor readings
        for sensor_id, reading in ultrasonic_readings.items():
            sensor_widget = QWidget()
            sensor_layout = QHBoxLayout(sensor_widget)
            
            # Sensor info
            info_label = QLabel(f"{sensor_id}:")
            info_label.setMinimumWidth(150)
            sensor_layout.addWidget(info_label)
            
            # Distance bar
            distance_bar = QProgressBar()
            distance_bar.setRange(0, int(reading.max_range * 100))
            distance_bar.setValue(int(reading.distance * 100))
            distance_bar.setFormat(f"{reading.distance:.2f}m")
            sensor_layout.addWidget(distance_bar)
            
            # Confidence
            conf_label = QLabel(f"Conf: {reading.confidence:.2f}")
            conf_label.setMinimumWidth(80)
            sensor_layout.addWidget(conf_label)
            
            self.readings_layout.addWidget(sensor_widget)
        
        # Trigger repaint for sensor visualization
        self.sensor_frame.update()
    
    def paintEvent(self, event):
        """Custom paint event for drawing sensor layout"""
        super().paintEvent(event)
        
        painter = QPainter(self.sensor_frame)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw vehicle outline
        vehicle_width = 60
        vehicle_height = 120
        center_x = self.sensor_frame.width() // 2
        center_y = self.sensor_frame.height() // 2
        
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.setBrush(QBrush(QColor(200, 200, 200)))
        painter.drawRect(center_x - vehicle_width//2, center_y - vehicle_height//2,
                        vehicle_width, vehicle_height)
        
        # Draw sensors and their readings
        for sensor_id, reading in self.sensor_data.items():
            # Determine sensor position (simplified)
            if "front" in sensor_id:
                sensor_x = center_x
                sensor_y = center_y - vehicle_height//2 - 10
                angle = -90  # Pointing forward
            else:
                continue  # Skip other sensors for now
            
            # Draw sensor
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QBrush(QColor(255, 0, 0)))
            painter.drawEllipse(sensor_x - 3, sensor_y - 3, 6, 6)
            
            # Draw detection beam
            if reading.distance < reading.max_range:
                beam_length = min(reading.distance * 20, 100)  # Scale for display
                
                painter.setPen(QPen(QColor(0, 255, 0, 100), 1))
                painter.setBrush(QBrush(QColor(0, 255, 0, 50)))
                
                # Draw beam cone
                beam_angle = math.radians(reading.beam_angle)
                points = [
                    (sensor_x, sensor_y),
                    (sensor_x - beam_length * math.sin(beam_angle/2), 
                     sensor_y - beam_length * math.cos(beam_angle/2)),
                    (sensor_x + beam_length * math.sin(beam_angle/2), 
                     sensor_y - beam_length * math.cos(beam_angle/2))
                ]
                
                painter.drawPolygon([pg.QtCore.QPoint(int(x), int(y)) for x, y in points])

class GPSVisualizationWidget(QWidget):
    """Widget for visualizing GPS sensor data"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.position_history = []
        self.max_history = 100
    
    def setup_ui(self):
        """Setup GPS visualization UI"""
        layout = QVBoxLayout(self)
        
        # Position plot
        self.position_plot = pg.PlotWidget()
        self.position_plot.setLabel('left', 'Latitude')
        self.position_plot.setLabel('bottom', 'Longitude')
        self.position_plot.setTitle('GPS Position Track')
        self.position_plot.showGrid(True, True)
        
        layout.addWidget(self.position_plot)
        
        # GPS info
        info_layout = QGridLayout()
        
        self.lat_label = QLabel("Latitude: --")
        self.lon_label = QLabel("Longitude: --")
        self.alt_label = QLabel("Altitude: --")
        self.acc_label = QLabel("Accuracy: --")
        self.sat_label = QLabel("Satellites: --")
        self.hdop_label = QLabel("HDOP: --")
        
        info_layout.addWidget(QLabel("Position:"), 0, 0)
        info_layout.addWidget(self.lat_label, 0, 1)
        info_layout.addWidget(self.lon_label, 0, 2)
        info_layout.addWidget(self.alt_label, 0, 3)
        
        info_layout.addWidget(QLabel("Quality:"), 1, 0)
        info_layout.addWidget(self.acc_label, 1, 1)
        info_layout.addWidget(self.sat_label, 1, 2)
        info_layout.addWidget(self.hdop_label, 1, 3)
        
        layout.addLayout(info_layout)
        
        # Accuracy visualization
        self.accuracy_bar = QProgressBar()
        self.accuracy_bar.setRange(0, 100)
        layout.addWidget(QLabel("Signal Quality:"))
        layout.addWidget(self.accuracy_bar)
    
    def update_data(self, readings: Dict[str, SensorReading]):
        """Update GPS visualization with new data"""
        gps_readings = {k: v for k, v in readings.items() 
                       if isinstance(v, GPSReading)}
        
        if not gps_readings:
            return
        
        # Use first GPS reading
        reading = next(iter(gps_readings.values()))
        
        # Update position history
        self.position_history.append((reading.longitude, reading.latitude))
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        # Update position plot
        if len(self.position_history) > 1:
            x_data = [pos[0] for pos in self.position_history]
            y_data = [pos[1] for pos in self.position_history]
            
            self.position_plot.clear()
            self.position_plot.plot(x_data, y_data, pen=pg.mkPen('b', width=2))
            
            # Mark current position
            current_pos = pg.ScatterPlotItem([x_data[-1]], [y_data[-1]], 
                                           pen=pg.mkPen(None), 
                                           brush=pg.mkBrush(255, 0, 0),
                                           size=10)
            self.position_plot.addItem(current_pos)
        
        # Update info labels
        self.lat_label.setText(f"Latitude: {reading.latitude:.6f}°")
        self.lon_label.setText(f"Longitude: {reading.longitude:.6f}°")
        self.alt_label.setText(f"Altitude: {reading.altitude:.1f}m")
        self.acc_label.setText(f"Accuracy: ±{reading.accuracy:.1f}m")
        self.sat_label.setText(f"Satellites: {reading.satellites}")
        self.hdop_label.setText(f"HDOP: {reading.hdop:.2f}")
        
        # Update accuracy bar (inverse of accuracy - lower is better)
        quality = max(0, min(100, 100 - (reading.accuracy * 10)))
        self.accuracy_bar.setValue(int(quality))

class IMUVisualizationWidget(QWidget):
    """Widget for visualizing IMU sensor data"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.data_history = {'acc': [], 'gyro': [], 'orient': []}
        self.max_history = 200
    
    def setup_ui(self):
        """Setup IMU visualization UI"""
        layout = QVBoxLayout(self)
        
        # Acceleration plot
        self.acc_plot = pg.PlotWidget()
        self.acc_plot.setLabel('left', 'Acceleration (m/s²)')
        self.acc_plot.setLabel('bottom', 'Time')
        self.acc_plot.setTitle('Acceleration')
        self.acc_plot.addLegend()
        layout.addWidget(self.acc_plot)
        
        # Angular velocity plot
        self.gyro_plot = pg.PlotWidget()
        self.gyro_plot.setLabel('left', 'Angular Velocity (rad/s)')
        self.gyro_plot.setLabel('bottom', 'Time')
        self.gyro_plot.setTitle('Angular Velocity')
        self.gyro_plot.addLegend()
        layout.addWidget(self.gyro_plot)
        
        # Current values
        values_layout = QGridLayout()
        
        self.acc_x_label = QLabel("Acc X: --")
        self.acc_y_label = QLabel("Acc Y: --")
        self.acc_z_label = QLabel("Acc Z: --")
        
        self.gyro_x_label = QLabel("Gyro X: --")
        self.gyro_y_label = QLabel("Gyro Y: --")
        self.gyro_z_label = QLabel("Gyro Z: --")
        
        self.roll_label = QLabel("Roll: --")
        self.pitch_label = QLabel("Pitch: --")
        self.yaw_label = QLabel("Yaw: --")
        
        values_layout.addWidget(QLabel("Acceleration:"), 0, 0)
        values_layout.addWidget(self.acc_x_label, 0, 1)
        values_layout.addWidget(self.acc_y_label, 0, 2)
        values_layout.addWidget(self.acc_z_label, 0, 3)
        
        values_layout.addWidget(QLabel("Angular Velocity:"), 1, 0)
        values_layout.addWidget(self.gyro_x_label, 1, 1)
        values_layout.addWidget(self.gyro_y_label, 1, 2)
        values_layout.addWidget(self.gyro_z_label, 1, 3)
        
        values_layout.addWidget(QLabel("Orientation:"), 2, 0)
        values_layout.addWidget(self.roll_label, 2, 1)
        values_layout.addWidget(self.pitch_label, 2, 2)
        values_layout.addWidget(self.yaw_label, 2, 3)
        
        layout.addLayout(values_layout)
    
    def update_data(self, readings: Dict[str, SensorReading]):
        """Update IMU visualization with new data"""
        imu_readings = {k: v for k, v in readings.items() 
                       if isinstance(v, IMUReading)}
        
        if not imu_readings:
            return
        
        # Use first IMU reading
        reading = next(iter(imu_readings.values()))
        
        # Update data history
        self.data_history['acc'].append((reading.acceleration.x, reading.acceleration.y, reading.acceleration.z))
        self.data_history['gyro'].append((reading.angular_velocity.x, reading.angular_velocity.y, reading.angular_velocity.z))
        self.data_history['orient'].append((reading.orientation.x, reading.orientation.y, reading.orientation.z))
        
        # Trim history
        for key in self.data_history:
            if len(self.data_history[key]) > self.max_history:
                self.data_history[key].pop(0)
        
        # Update acceleration plot
        if self.data_history['acc']:
            time_data = list(range(len(self.data_history['acc'])))
            acc_x = [data[0] for data in self.data_history['acc']]
            acc_y = [data[1] for data in self.data_history['acc']]
            acc_z = [data[2] for data in self.data_history['acc']]
            
            self.acc_plot.clear()
            self.acc_plot.plot(time_data, acc_x, pen='r', name='X')
            self.acc_plot.plot(time_data, acc_y, pen='g', name='Y')
            self.acc_plot.plot(time_data, acc_z, pen='b', name='Z')
        
        # Update gyroscope plot
        if self.data_history['gyro']:
            time_data = list(range(len(self.data_history['gyro'])))
            gyro_x = [data[0] for data in self.data_history['gyro']]
            gyro_y = [data[1] for data in self.data_history['gyro']]
            gyro_z = [data[2] for data in self.data_history['gyro']]
            
            self.gyro_plot.clear()
            self.gyro_plot.plot(time_data, gyro_x, pen='r', name='X')
            self.gyro_plot.plot(time_data, gyro_y, pen='g', name='Y')
            self.gyro_plot.plot(time_data, gyro_z, pen='b', name='Z')
        
        # Update current value labels
        self.acc_x_label.setText(f"Acc X: {reading.acceleration.x:.2f}")
        self.acc_y_label.setText(f"Acc Y: {reading.acceleration.y:.2f}")
        self.acc_z_label.setText(f"Acc Z: {reading.acceleration.z:.2f}")
        
        self.gyro_x_label.setText(f"Gyro X: {reading.angular_velocity.x:.3f}")
        self.gyro_y_label.setText(f"Gyro Y: {reading.angular_velocity.y:.3f}")
        self.gyro_z_label.setText(f"Gyro Z: {reading.angular_velocity.z:.3f}")
        
        self.roll_label.setText(f"Roll: {math.degrees(reading.orientation.x):.1f}°")
        self.pitch_label.setText(f"Pitch: {math.degrees(reading.orientation.y):.1f}°")
        self.yaw_label.setText(f"Yaw: {math.degrees(reading.orientation.z):.1f}°")

class SensorStatusWidget(QWidget):
    """Widget for displaying sensor status information"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup sensor status UI"""
        layout = QVBoxLayout(self)
        
        # Status scroll area
        self.status_scroll = QScrollArea()
        self.status_widget = QWidget()
        self.status_layout = QVBoxLayout(self.status_widget)
        self.status_scroll.setWidget(self.status_widget)
        layout.addWidget(self.status_scroll)
    
    def update_data(self, sensor_status: Dict[str, Dict[str, Any]]):
        """Update sensor status display"""
        # Clear previous status display
        for i in reversed(range(self.status_layout.count())):
            self.status_layout.itemAt(i).widget().setParent(None)
        
        # Add sensor status widgets
        for sensor_id, status in sensor_status.items():
            sensor_widget = QGroupBox(sensor_id)
            sensor_layout = QGridLayout(sensor_widget)
            
            # Status indicators
            type_label = QLabel(f"Type: {status['type']}")
            enabled_label = QLabel(f"Enabled: {'Yes' if status['enabled'] else 'No'}")
            active_label = QLabel(f"Active: {'Yes' if status['active'] else 'No'}")
            failed_label = QLabel(f"Failed: {'Yes' if status['failed'] else 'No'}")
            
            # Color code based on status
            if status['failed']:
                failed_label.setStyleSheet("color: red; font-weight: bold;")
            elif not status['enabled'] or not status['active']:
                enabled_label.setStyleSheet("color: orange;")
                active_label.setStyleSheet("color: orange;")
            else:
                enabled_label.setStyleSheet("color: green;")
                active_label.setStyleSheet("color: green;")
            
            rate_label = QLabel(f"Update Rate: {status['update_rate']:.1f} Hz")
            noise_label = QLabel(f"Noise Level: {status['noise_level']:.2f}")
            
            sensor_layout.addWidget(type_label, 0, 0)
            sensor_layout.addWidget(enabled_label, 0, 1)
            sensor_layout.addWidget(active_label, 1, 0)
            sensor_layout.addWidget(failed_label, 1, 1)
            sensor_layout.addWidget(rate_label, 2, 0)
            sensor_layout.addWidget(noise_label, 2, 1)
            
            self.status_layout.addWidget(sensor_widget)