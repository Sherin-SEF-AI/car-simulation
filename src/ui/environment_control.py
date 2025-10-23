"""
Environment Control Panel
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QComboBox, QSlider, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal


class EnvironmentControlPanel(QWidget):
    """Environment control panel"""
    
    environment_changed = pyqtSignal(str, dict)
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        self.setup_ui()
        
    def setup_ui(self):
        """Setup environment control UI"""
        layout = QVBoxLayout(self)
        
        # Weather control
        weather_group = QGroupBox("Weather")
        weather_layout = QVBoxLayout(weather_group)
        
        weather_select_layout = QHBoxLayout()
        weather_select_layout.addWidget(QLabel("Condition:"))
        self.weather_combo = QComboBox()
        self.weather_combo.addItems(["Clear", "Cloudy", "Rain", "Snow", "Fog"])
        weather_select_layout.addWidget(self.weather_combo)
        weather_layout.addLayout(weather_select_layout)
        
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensity:"))
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(50)
        intensity_layout.addWidget(self.intensity_slider)
        weather_layout.addLayout(intensity_layout)
        
        layout.addWidget(weather_group)
        
        # Time control
        time_group = QGroupBox("Time of Day")
        time_layout = QVBoxLayout(time_group)
        
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 1440)  # 24 hours in minutes
        self.time_slider.setValue(720)  # 12:00 PM
        time_layout.addWidget(self.time_slider)
        
        time_labels = QHBoxLayout()
        time_labels.addWidget(QLabel("00:00"))
        self.time_label = QLabel("12:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_labels.addWidget(self.time_label)
        time_labels.addWidget(QLabel("24:00"))
        time_layout.addLayout(time_labels)
        
        layout.addWidget(time_group)
        
        # Temperature
        temp_group = QGroupBox("Temperature")
        temp_layout = QVBoxLayout(temp_group)
        
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(-20, 50)
        self.temp_slider.setValue(20)
        temp_layout.addWidget(self.temp_slider)
        
        temp_labels = QHBoxLayout()
        temp_labels.addWidget(QLabel("-20°C"))
        self.temp_label = QLabel("20°C")
        self.temp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        temp_labels.addWidget(self.temp_label)
        temp_labels.addWidget(QLabel("50°C"))
        temp_layout.addLayout(temp_labels)
        
        layout.addWidget(temp_group)
        
        # Connect signals
        self.time_slider.valueChanged.connect(self.update_time_display)
        self.temp_slider.valueChanged.connect(self.update_temp_display)
        
    def update_time_display(self, minutes):
        """Update time display"""
        hours = minutes // 60
        mins = minutes % 60
        self.time_label.setText(f"{hours:02d}:{mins:02d}")
        
    def update_temp_display(self, temp):
        """Update temperature display"""
        self.temp_label.setText(f"{temp}°C")