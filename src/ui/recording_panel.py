"""
Recording Panel for simulation recording and replay
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QPushButton, QSlider, QTextEdit, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer


class RecordingPanel(QWidget):
    """Recording and replay panel"""
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        self.recording = False
        self.setup_ui()
        
    def setup_ui(self):
        """Setup recording panel UI"""
        layout = QVBoxLayout(self)
        
        # Recording controls
        record_group = QGroupBox("Recording")
        record_layout = QVBoxLayout(record_group)
        
        # Record buttons
        record_controls = QHBoxLayout()
        self.record_btn = QPushButton("● Record")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setStyleSheet("QPushButton { background-color: #d32f2f; }")
        record_controls.addWidget(self.record_btn)
        
        save_btn = QPushButton("Save Recording")
        save_btn.clicked.connect(self.save_recording)
        record_controls.addWidget(save_btn)
        
        record_layout.addLayout(record_controls)
        
        # Recording status
        self.record_status = QLabel("Ready to record")
        record_layout.addWidget(self.record_status)
        
        layout.addWidget(record_group)
        
        # Playback controls
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout(playback_group)
        
        # Load recording
        load_layout = QHBoxLayout()
        load_btn = QPushButton("Load Recording")
        load_btn.clicked.connect(self.load_recording)
        load_layout.addWidget(load_btn)
        playback_layout.addLayout(load_layout)
        
        # Playback controls
        playback_controls = QHBoxLayout()
        play_btn = QPushButton("▶ Play")
        playback_controls.addWidget(play_btn)
        
        pause_btn = QPushButton("⏸ Pause")
        playback_controls.addWidget(pause_btn)
        
        stop_btn = QPushButton("⏹ Stop")
        playback_controls.addWidget(stop_btn)
        
        playback_layout.addLayout(playback_controls)
        
        # Playback progress
        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.setRange(0, 100)
        playback_layout.addWidget(self.playback_slider)
        
        layout.addWidget(playback_group)
        
        # Recording info
        info_group = QGroupBox("Recording Info")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        
        # Initialize
        self.info_text.setText("No recording loaded.")
        
    def toggle_recording(self):
        """Toggle recording state"""
        if self.recording:
            self.recording = False
            self.record_btn.setText("● Record")
            self.record_btn.setStyleSheet("QPushButton { background-color: #d32f2f; }")
            self.record_status.setText("Recording stopped")
        else:
            self.recording = True
            self.record_btn.setText("⏹ Stop")
            self.record_btn.setStyleSheet("QPushButton { background-color: #4caf50; }")
            self.record_status.setText("Recording...")
            
    def save_recording(self):
        """Save current recording"""
        filename, _ = QFileDialog.getSaveFileName(self, "Save Recording", "", 
                                                "Recording Files (*.rec);;All Files (*)")
        if filename:
            self.info_text.setText(f"Recording saved to: {filename}")
            
    def load_recording(self):
        """Load a recording file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Recording", "", 
                                                "Recording Files (*.rec);;All Files (*)")
        if filename:
            self.info_text.setText(f"Recording loaded from: {filename}")