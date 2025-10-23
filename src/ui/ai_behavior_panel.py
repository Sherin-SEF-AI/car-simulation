"""
AI Behavior Panel for controlling vehicle AI
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QComboBox, QSlider, QCheckBox, QTextEdit,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal


class AIBehaviorPanel(QWidget):
    """AI behavior control panel"""
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        self.setup_ui()
        
    def setup_ui(self):
        """Setup AI behavior panel UI"""
        layout = QVBoxLayout(self)
        
        # Vehicle selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Vehicle:"))
        self.vehicle_combo = QComboBox()
        selector_layout.addWidget(self.vehicle_combo)
        layout.addLayout(selector_layout)
        
        # AI controls
        ai_group = QGroupBox("AI Control")
        ai_layout = QVBoxLayout(ai_group)
        
        # Enable/disable AI
        self.ai_enabled = QCheckBox("Enable Autonomous Mode")
        self.ai_enabled.setChecked(True)
        ai_layout.addWidget(self.ai_enabled)
        
        # Behavior selection
        behavior_layout = QHBoxLayout()
        behavior_layout.addWidget(QLabel("Behavior:"))
        self.behavior_combo = QComboBox()
        self.behavior_combo.addItems(["Default", "Aggressive", "Cautious", "Learning"])
        behavior_layout.addWidget(self.behavior_combo)
        ai_layout.addLayout(behavior_layout)
        
        # AI parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Speed preference
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed Preference:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(0, 100)
        self.speed_slider.setValue(50)
        speed_layout.addWidget(self.speed_slider)
        params_layout.addLayout(speed_layout)
        
        # Risk tolerance
        risk_layout = QHBoxLayout()
        risk_layout.addWidget(QLabel("Risk Tolerance:"))
        self.risk_slider = QSlider(Qt.Orientation.Horizontal)
        self.risk_slider.setRange(0, 100)
        self.risk_slider.setValue(30)
        risk_layout.addWidget(self.risk_slider)
        params_layout.addLayout(risk_layout)
        
        ai_layout.addWidget(params_group)
        layout.addWidget(ai_group)
        
        # AI status
        status_group = QGroupBox("AI Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)
        
        layout.addWidget(status_group)
        
        # Update status
        self.status_text.setText("AI system ready. Select a vehicle to configure behavior.")