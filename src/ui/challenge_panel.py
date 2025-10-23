"""
Challenge Panel for simulation challenges
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QListWidget, QTextEdit, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer


class ChallengePanel(QWidget):
    """Challenge management panel"""
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        self.setup_ui()
        
    def setup_ui(self):
        """Setup challenge panel UI"""
        layout = QVBoxLayout(self)
        
        # Available challenges
        challenges_group = QGroupBox("Available Challenges")
        challenges_layout = QVBoxLayout(challenges_group)
        
        self.challenges_list = QListWidget()
        challenges = [
            "Parallel Parking",
            "Highway Merging", 
            "Emergency Braking",
            "Obstacle Avoidance",
            "Traffic Light Navigation",
            "Roundabout Navigation"
        ]
        self.challenges_list.addItems(challenges)
        challenges_layout.addWidget(self.challenges_list)
        
        # Challenge controls
        controls_layout = QHBoxLayout()
        start_btn = QPushButton("Start Challenge")
        controls_layout.addWidget(start_btn)
        
        stop_btn = QPushButton("Stop Challenge")
        controls_layout.addWidget(stop_btn)
        
        challenges_layout.addLayout(controls_layout)
        layout.addWidget(challenges_group)
        
        # Challenge progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_text = QTextEdit()
        self.progress_text.setMaximumHeight(100)
        self.progress_text.setReadOnly(True)
        progress_layout.addWidget(self.progress_text)
        
        layout.addWidget(progress_group)
        
        # Initialize
        self.progress_text.setText("Select a challenge to begin.")