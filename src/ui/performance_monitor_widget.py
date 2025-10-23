"""
Performance Monitor Widget
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont


class PerformanceMonitorWidget(QWidget):
    """Performance monitoring widget"""
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        """Setup performance monitor UI"""
        layout = QVBoxLayout(self)
        
        # Performance metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_bar = QProgressBar()
        self.fps_bar.setRange(0, 120)
        self.fps_bar.setValue(60)
        fps_layout.addWidget(self.fps_bar)
        self.fps_label = QLabel("60")
        fps_layout.addWidget(self.fps_label)
        metrics_layout.addLayout(fps_layout)
        
        # Memory
        mem_layout = QHBoxLayout()
        mem_layout.addWidget(QLabel("Memory:"))
        self.mem_bar = QProgressBar()
        self.mem_bar.setRange(0, 2048)  # 2GB max
        self.mem_bar.setValue(512)
        mem_layout.addWidget(self.mem_bar)
        self.mem_label = QLabel("512 MB")
        mem_layout.addWidget(self.mem_label)
        metrics_layout.addLayout(mem_layout)
        
        # CPU
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(QLabel("CPU:"))
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setValue(25)
        cpu_layout.addWidget(self.cpu_bar)
        self.cpu_label = QLabel("25%")
        cpu_layout.addWidget(self.cpu_label)
        metrics_layout.addLayout(cpu_layout)
        
        layout.addWidget(metrics_group)
        
        # Performance log
        log_group = QGroupBox("Performance Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 9px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
    def setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_performance)
        self.timer.start(1000)  # Update every second
        
    def update_performance(self):
        """Update performance metrics"""
        import psutil
        import time
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_mb = memory.used // (1024 * 1024)
            
            # Update bars
            self.cpu_bar.setValue(int(cpu_percent))
            self.cpu_label.setText(f"{cpu_percent:.1f}%")
            
            self.mem_bar.setValue(memory_mb)
            self.mem_label.setText(f"{memory_mb} MB")
            
            # Simulate FPS (would come from actual renderer)
            import random
            fps = random.randint(55, 65)
            self.fps_bar.setValue(fps)
            self.fps_label.setText(str(fps))
            
            # Add log entry
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] FPS: {fps}, CPU: {cpu_percent:.1f}%, MEM: {memory_mb}MB"
            self.log_text.append(log_entry)
            
            # Keep only last 20 lines
            text = self.log_text.toPlainText()
            lines = text.split('\n')
            if len(lines) > 20:
                self.log_text.setPlainText('\n'.join(lines[-20:]))
                
        except Exception as e:
            pass  # Ignore performance monitoring errors