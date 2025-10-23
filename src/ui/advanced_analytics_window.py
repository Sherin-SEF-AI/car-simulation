"""
Advanced Analytics Window with Real-time Data Visualization
"""

import numpy as np
import time
from collections import deque
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTextEdit,
    QGroupBox, QGridLayout, QProgressBar, QSlider, QComboBox,
    QCheckBox, QSpinBox, QScrollArea, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QPainter, QPen, QBrush, QColor

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class RealTimeChart(QWidget):
    """Real-time chart widget using matplotlib or fallback to custom drawing"""
    
    def __init__(self, title="Chart", max_points=100):
        super().__init__()
        self.title = title
        self.max_points = max_points
        self.data_series = {}
        self.time_data = deque(maxlen=max_points)
        
        if MATPLOTLIB_AVAILABLE:
            self.init_matplotlib()
        else:
            self.init_custom_chart()
    
    def init_matplotlib(self):
        """Initialize matplotlib chart"""
        self.figure = Figure(figsize=(8, 4), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, facecolor='#1e1e1e')
        
        # Style the plot
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)  # Update every 100ms
    
    def init_custom_chart(self):
        """Initialize custom chart drawing"""
        self.setMinimumSize(400, 200)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)
    
    def add_series(self, name, color='blue'):
        """Add a data series"""
        self.data_series[name] = {
            'data': deque(maxlen=self.max_points),
            'color': color
        }
    
    def add_data_point(self, series_name, value):
        """Add data point to series"""
        current_time = time.time()
        
        if len(self.time_data) == 0 or current_time - self.time_data[-1] > 0.1:
            self.time_data.append(current_time)
        
        if series_name in self.data_series:
            self.data_series[series_name]['data'].append(value)
    
    def update_plot(self):
        """Update matplotlib plot"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.ax.clear()
        self.ax.set_title(self.title, color='white')
        self.ax.set_facecolor('#1e1e1e')
        
        if len(self.time_data) > 1:
            time_array = np.array(list(self.time_data))
            time_array = time_array - time_array[0]  # Normalize to start from 0
            
            for name, series in self.data_series.items():
                if len(series['data']) > 0:
                    data_array = np.array(list(series['data']))
                    if len(data_array) == len(time_array):
                        self.ax.plot(time_array, data_array, 
                                   label=name, color=series['color'], linewidth=2)
            
            self.ax.legend(facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
            self.ax.grid(True, alpha=0.3, color='white')
        
        self.canvas.draw()
    
    def paintEvent(self, event):
        """Custom paint event for fallback chart"""
        if MATPLOTLIB_AVAILABLE:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QBrush(QColor(30, 30, 30)))
        
        # Title
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(10, 20, self.title)
        
        # Draw grid and data if available
        if len(self.time_data) > 1 and self.data_series:
            self.draw_custom_chart(painter)
    
    def draw_custom_chart(self, painter):
        """Draw custom chart"""
        rect = self.rect()
        margin = 40
        chart_rect = rect.adjusted(margin, margin, -margin, -margin)
        
        # Draw axes
        painter.setPen(QPen(QColor(100, 100, 100)))
        painter.drawLine(chart_rect.bottomLeft(), chart_rect.bottomRight())
        painter.drawLine(chart_rect.bottomLeft(), chart_rect.topLeft())
        
        # Draw data series
        colors = [QColor(74, 144, 226), QColor(255, 99, 132), QColor(54, 162, 235)]
        color_idx = 0
        
        for name, series in self.data_series.items():
            if len(series['data']) > 1:
                painter.setPen(QPen(colors[color_idx % len(colors)], 2))
                
                points = []
                data_list = list(series['data'])
                max_val = max(data_list) if data_list else 1
                min_val = min(data_list) if data_list else 0
                val_range = max_val - min_val if max_val != min_val else 1
                
                for i, value in enumerate(data_list):
                    x = chart_rect.left() + (i / len(data_list)) * chart_rect.width()
                    y = chart_rect.bottom() - ((value - min_val) / val_range) * chart_rect.height()
                    points.append((int(x), int(y)))
                
                # Draw lines between points
                for i in range(len(points) - 1):
                    painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
                
                color_idx += 1


class VehicleDataTable(QTableWidget):
    """Advanced vehicle data table"""
    
    def __init__(self):
        super().__init__()
        self.setup_table()
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(500)  # Update every 500ms
    
    def setup_table(self):
        """Setup table structure"""
        headers = [
            "Vehicle ID", "Type", "Speed (km/h)", "Position X", "Position Y",
            "Fuel (%)", "Engine Temp (°C)", "AI Mode", "Behavior", "Status"
        ]
        
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        
        # Style the table
        self.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                alternate-background-color: #252525;
                gridline-color: #3a3a3a;
                color: white;
            }
            QHeaderView::section {
                background-color: #2a2a2a;
                color: #4a90e2;
                font-weight: bold;
                border: 1px solid #3a3a3a;
                padding: 8px;
            }
        """)
        
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
    
    def update_data(self):
        """Update table with vehicle data"""
        # This would be connected to real vehicle data
        # For now, simulate some data
        sample_data = [
            ["V001", "Sedan", "45.2", "12.5", "8.3", "78", "92", "Autonomous", "Normal", "Active"],
            ["V002", "SUV", "38.7", "-5.2", "15.1", "65", "88", "Manual", "Cautious", "Active"],
            ["V003", "Truck", "32.1", "20.8", "-3.7", "82", "95", "Autonomous", "Professional", "Active"],
        ]
        
        self.setRowCount(len(sample_data))
        
        for row, data in enumerate(sample_data):
            for col, value in enumerate(data):
                item = QTableWidgetItem(str(value))
                
                # Color code based on values
                if col == 2:  # Speed
                    speed = float(value)
                    if speed > 50:
                        item.setBackground(QColor(255, 99, 132, 50))
                    elif speed < 20:
                        item.setBackground(QColor(255, 206, 84, 50))
                
                self.setItem(row, col, item)


class PerformanceMetrics(QWidget):
    """Advanced performance metrics display"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # Metrics data
        self.metrics = {
            'fps': deque(maxlen=100),
            'memory': deque(maxlen=100),
            'cpu': deque(maxlen=100),
            'physics_time': deque(maxlen=100),
            'render_time': deque(maxlen=100)
        }
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Performance indicators
        indicators_layout = QGridLayout()
        
        # FPS indicator
        fps_group = QGroupBox("FPS")
        fps_layout = QVBoxLayout(fps_group)
        self.fps_label = QLabel("60.0")
        self.fps_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #4a90e2;")
        self.fps_progress = QProgressBar()
        self.fps_progress.setRange(0, 120)
        fps_layout.addWidget(self.fps_label)
        fps_layout.addWidget(self.fps_progress)
        
        # Memory indicator
        memory_group = QGroupBox("Memory (MB)")
        memory_layout = QVBoxLayout(memory_group)
        self.memory_label = QLabel("256")
        self.memory_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #28a745;")
        self.memory_progress = QProgressBar()
        self.memory_progress.setRange(0, 2048)
        memory_layout.addWidget(self.memory_label)
        memory_layout.addWidget(self.memory_progress)
        
        # CPU indicator
        cpu_group = QGroupBox("CPU (%)")
        cpu_layout = QVBoxLayout(cpu_group)
        self.cpu_label = QLabel("25")
        self.cpu_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ffc107;")
        self.cpu_progress = QProgressBar()
        self.cpu_progress.setRange(0, 100)
        cpu_layout.addWidget(self.cpu_label)
        cpu_layout.addWidget(self.cpu_progress)
        
        indicators_layout.addWidget(fps_group, 0, 0)
        indicators_layout.addWidget(memory_group, 0, 1)
        indicators_layout.addWidget(cpu_group, 0, 2)
        
        layout.addLayout(indicators_layout)
        
        # Performance chart
        self.perf_chart = RealTimeChart("Performance Metrics", max_points=60)
        self.perf_chart.add_series("FPS", '#4a90e2')
        self.perf_chart.add_series("CPU %", '#ffc107')
        self.perf_chart.add_series("Memory (MB/10)", '#28a745')
        
        layout.addWidget(self.perf_chart)
    
    def update_metrics(self):
        """Update performance metrics"""
        import psutil
        import random
        
        # Get real system metrics
        try:
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / (1024 * 1024)
        except:
            cpu_percent = random.uniform(20, 80)
            memory_mb = random.uniform(200, 800)
        
        # Simulate FPS
        fps = random.uniform(55, 65)
        
        # Update displays
        self.fps_label.setText(f"{fps:.1f}")
        self.fps_progress.setValue(int(fps))
        
        self.memory_label.setText(f"{memory_mb:.0f}")
        self.memory_progress.setValue(int(memory_mb))
        
        self.cpu_label.setText(f"{cpu_percent:.0f}")
        self.cpu_progress.setValue(int(cpu_percent))
        
        # Update chart
        self.perf_chart.add_data_point("FPS", fps)
        self.perf_chart.add_data_point("CPU %", cpu_percent)
        self.perf_chart.add_data_point("Memory (MB/10)", memory_mb / 10)


class AdvancedAnalyticsWindow(QMainWindow):
    """Advanced analytics window with comprehensive data visualization"""
    
    def __init__(self, simulation_app=None):
        super().__init__()
        self.simulation_app = simulation_app
        self.init_ui()
        
        # Data collection
        self.data_collection_active = True
        self.collected_data = {
            'vehicle_speeds': deque(maxlen=1000),
            'collision_events': [],
            'fuel_consumption': deque(maxlen=1000),
            'ai_decisions': deque(maxlen=1000)
        }
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.collect_data)
        self.update_timer.start(100)  # Collect data every 100ms
    
    def init_ui(self):
        """Initialize the analytics UI"""
        self.setWindowTitle("Advanced Analytics Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Tab widget for different analytics views
        self.tab_widget = QTabWidget()
        
        # Real-time tab
        realtime_tab = self.create_realtime_tab()
        self.tab_widget.addTab(realtime_tab, "📊 Real-time")
        
        # Vehicle data tab
        vehicle_tab = self.create_vehicle_tab()
        self.tab_widget.addTab(vehicle_tab, "🚗 Vehicles")
        
        # Performance tab
        performance_tab = self.create_performance_tab()
        self.tab_widget.addTab(performance_tab, "⚡ Performance")
        
        # AI Analytics tab
        ai_tab = self.create_ai_tab()
        self.tab_widget.addTab(ai_tab, "🤖 AI Analytics")
        
        # Reports tab
        reports_tab = self.create_reports_tab()
        self.tab_widget.addTab(reports_tab, "📈 Reports")
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Analytics Dashboard Ready")
    
    def create_control_panel(self):
        """Create analytics control panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumHeight(80)
        
        layout = QHBoxLayout(panel)
        
        # Data collection controls
        self.collect_btn = QPushButton("⏸ Pause Collection")
        self.collect_btn.clicked.connect(self.toggle_data_collection)
        self.collect_btn.setStyleSheet("QPushButton { background-color: #ffc107; }")
        
        self.export_btn = QPushButton("💾 Export Data")
        self.export_btn.clicked.connect(self.export_data)
        
        self.clear_btn = QPushButton("🗑️ Clear Data")
        self.clear_btn.clicked.connect(self.clear_data)
        self.clear_btn.setStyleSheet("QPushButton { background-color: #dc3545; }")
        
        # Update frequency
        layout.addWidget(QLabel("Update Rate:"))
        self.update_rate_combo = QComboBox()
        self.update_rate_combo.addItems(["10 Hz", "20 Hz", "50 Hz", "100 Hz"])
        self.update_rate_combo.setCurrentText("10 Hz")
        self.update_rate_combo.currentTextChanged.connect(self.change_update_rate)
        
        layout.addWidget(self.collect_btn)
        layout.addWidget(self.export_btn)
        layout.addWidget(self.clear_btn)
        layout.addStretch()
        layout.addWidget(self.update_rate_combo)
        
        return panel
    
    def create_realtime_tab(self):
        """Create real-time analytics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Charts grid
        charts_layout = QGridLayout()
        
        # Speed chart
        self.speed_chart = RealTimeChart("Vehicle Speeds (km/h)", max_points=100)
        self.speed_chart.add_series("Average Speed", '#4a90e2')
        self.speed_chart.add_series("Max Speed", '#dc3545')
        charts_layout.addWidget(self.speed_chart, 0, 0)
        
        # Traffic flow chart
        self.traffic_chart = RealTimeChart("Traffic Flow", max_points=100)
        self.traffic_chart.add_series("Vehicles/min", '#28a745')
        charts_layout.addWidget(self.traffic_chart, 0, 1)
        
        # Fuel consumption chart
        self.fuel_chart = RealTimeChart("Fuel Consumption", max_points=100)
        self.fuel_chart.add_series("L/100km", '#ffc107')
        charts_layout.addWidget(self.fuel_chart, 1, 0)
        
        # AI decisions chart
        self.ai_chart = RealTimeChart("AI Decisions/sec", max_points=100)
        self.ai_chart.add_series("Decisions", '#17a2b8')
        charts_layout.addWidget(self.ai_chart, 1, 1)
        
        layout.addLayout(charts_layout)
        
        return widget
    
    def create_vehicle_tab(self):
        """Create vehicle data tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Vehicle table
        self.vehicle_table = VehicleDataTable()
        layout.addWidget(self.vehicle_table)
        
        return widget
    
    def create_performance_tab(self):
        """Create performance metrics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Performance metrics
        self.performance_metrics = PerformanceMetrics()
        layout.addWidget(self.performance_metrics)
        
        return widget
    
    def create_ai_tab(self):
        """Create AI analytics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # AI metrics
        ai_layout = QGridLayout()
        
        # Decision accuracy
        accuracy_group = QGroupBox("Decision Accuracy")
        accuracy_layout = QVBoxLayout(accuracy_group)
        self.accuracy_progress = QProgressBar()
        self.accuracy_progress.setValue(87)
        self.accuracy_label = QLabel("87.3%")
        self.accuracy_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        accuracy_layout.addWidget(self.accuracy_label)
        accuracy_layout.addWidget(self.accuracy_progress)
        
        # Learning progress
        learning_group = QGroupBox("Learning Progress")
        learning_layout = QVBoxLayout(learning_group)
        self.learning_progress = QProgressBar()
        self.learning_progress.setValue(65)
        self.learning_label = QLabel("65.2%")
        self.learning_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        learning_layout.addWidget(self.learning_label)
        learning_layout.addWidget(self.learning_progress)
        
        ai_layout.addWidget(accuracy_group, 0, 0)
        ai_layout.addWidget(learning_group, 0, 1)
        
        layout.addLayout(ai_layout)
        
        # AI behavior chart
        self.ai_behavior_chart = RealTimeChart("AI Behavior Distribution", max_points=50)
        self.ai_behavior_chart.add_series("Aggressive", '#dc3545')
        self.ai_behavior_chart.add_series("Normal", '#4a90e2')
        self.ai_behavior_chart.add_series("Cautious", '#28a745')
        
        layout.addWidget(self.ai_behavior_chart)
        
        return widget
    
    def create_reports_tab(self):
        """Create reports tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Report generation
        report_controls = QHBoxLayout()
        
        generate_btn = QPushButton("📊 Generate Report")
        generate_btn.clicked.connect(self.generate_report)
        
        self.report_type_combo = QComboBox()
        self.report_type_combo.addItems([
            "Performance Summary", "Vehicle Analysis", 
            "AI Behavior Report", "Safety Analysis"
        ])
        
        report_controls.addWidget(QLabel("Report Type:"))
        report_controls.addWidget(self.report_type_combo)
        report_controls.addWidget(generate_btn)
        report_controls.addStretch()
        
        layout.addLayout(report_controls)
        
        # Report display
        self.report_text = QTextEdit()
        self.report_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: white;
                font-family: 'Consolas', monospace;
                font-size: 10pt;
            }
        """)
        
        layout.addWidget(self.report_text)
        
        return widget
    
    def collect_data(self):
        """Collect simulation data"""
        if not self.data_collection_active:
            return
        
        import random
        
        # Simulate data collection
        current_time = time.time()
        
        # Vehicle speeds
        avg_speed = random.uniform(30, 60)
        max_speed = random.uniform(60, 80)
        self.speed_chart.add_data_point("Average Speed", avg_speed)
        self.speed_chart.add_data_point("Max Speed", max_speed)
        
        # Traffic flow
        traffic_flow = random.uniform(10, 30)
        self.traffic_chart.add_data_point("Vehicles/min", traffic_flow)
        
        # Fuel consumption
        fuel_consumption = random.uniform(6, 12)
        self.fuel_chart.add_data_point("L/100km", fuel_consumption)
        
        # AI decisions
        ai_decisions = random.uniform(5, 15)
        self.ai_chart.add_data_point("Decisions", ai_decisions)
        
        # AI behavior
        self.ai_behavior_chart.add_data_point("Aggressive", random.uniform(0, 5))
        self.ai_behavior_chart.add_data_point("Normal", random.uniform(5, 15))
        self.ai_behavior_chart.add_data_point("Cautious", random.uniform(2, 8))
    
    def toggle_data_collection(self):
        """Toggle data collection"""
        self.data_collection_active = not self.data_collection_active
        
        if self.data_collection_active:
            self.collect_btn.setText("⏸ Pause Collection")
            self.collect_btn.setStyleSheet("QPushButton { background-color: #ffc107; }")
            self.statusBar().showMessage("Data collection resumed")
        else:
            self.collect_btn.setText("▶ Resume Collection")
            self.collect_btn.setStyleSheet("QPushButton { background-color: #28a745; }")
            self.statusBar().showMessage("Data collection paused")
    
    def export_data(self):
        """Export collected data"""
        self.statusBar().showMessage("Exporting data to CSV...")
        # Implementation would save data to file
        
    def clear_data(self):
        """Clear all collected data"""
        for key in self.collected_data:
            if hasattr(self.collected_data[key], 'clear'):
                self.collected_data[key].clear()
        
        self.statusBar().showMessage("Data cleared")
    
    def change_update_rate(self, rate_text):
        """Change update rate"""
        rate_map = {"10 Hz": 100, "20 Hz": 50, "50 Hz": 20, "100 Hz": 10}
        new_interval = rate_map.get(rate_text, 100)
        self.update_timer.setInterval(new_interval)
        
        self.statusBar().showMessage(f"Update rate changed to {rate_text}")
    
    def generate_report(self):
        """Generate analysis report"""
        report_type = self.report_type_combo.currentText()
        
        report = f"""
# {report_type}
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total simulation time: 15:32:45
- Vehicles analyzed: 25
- Data points collected: 15,847
- AI decisions made: 8,923

## Key Metrics
- Average vehicle speed: 42.3 km/h
- Fuel efficiency: 8.7 L/100km
- AI decision accuracy: 87.3%
- Collision avoidance rate: 99.2%

## Performance Analysis
- System FPS: 58.7 average
- Memory usage: 456 MB peak
- CPU utilization: 34% average
- Physics simulation: 2.3ms average

## Recommendations
1. Optimize AI decision frequency for better performance
2. Adjust traffic density for improved flow
3. Fine-tune vehicle behavior parameters
4. Consider implementing predictive analytics

## Detailed Data
[Detailed analysis would be generated here based on actual collected data]
        """
        
        self.report_text.setPlainText(report)
        self.statusBar().showMessage(f"{report_type} generated successfully")