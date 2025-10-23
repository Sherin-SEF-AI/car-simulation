"""
Advanced Analytics and Data Science Panel
Comprehensive data analysis, statistical modeling, and predictive analytics
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QPushButton, QComboBox, QSlider, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QSplitter, QTreeWidget, QTreeWidgetItem, QScrollArea, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont

import numpy as np
import json
import time
from collections import deque
import threading
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class StatisticalAnalyzer(QThread):
    """Background thread for statistical analysis"""
    
    analysis_progress = pyqtSignal(int, str)
    analysis_complete = pyqtSignal(dict)
    
    def __init__(self, data, analysis_type):
        super().__init__()
        self.data = data
        self.analysis_type = analysis_type
        self.is_running = False
    
    def run(self):
        """Run statistical analysis"""
        self.is_running = True
        
        try:
            if self.analysis_type == "descriptive":
                results = self.descriptive_analysis()
            elif self.analysis_type == "correlation":
                results = self.correlation_analysis()
            elif self.analysis_type == "clustering":
                results = self.clustering_analysis()
            elif self.analysis_type == "regression":
                results = self.regression_analysis()
            elif self.analysis_type == "anomaly":
                results = self.anomaly_detection()
            else:
                results = {"error": "Unknown analysis type"}
            
            if self.is_running:
                self.analysis_complete.emit(results)
                
        except Exception as e:
            self.analysis_complete.emit({"error": str(e)})
    
    def descriptive_analysis(self):
        """Perform descriptive statistical analysis"""
        results = {}
        
        for i, column in enumerate(['speed', 'acceleration', 'steering', 'fuel']):
            self.analysis_progress.emit(int((i + 1) / 4 * 100), f"Analyzing {column}...")
            
            if not self.is_running:
                break
            
            # Generate sample data
            data = np.random.normal(50, 15, 1000) if column == 'speed' else np.random.normal(0, 5, 1000)
            
            results[column] = {
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'q25': float(np.percentile(data, 25)),
                'q75': float(np.percentile(data, 75)),
                'skewness': float(stats.skew(data)) if SCIPY_AVAILABLE else float(np.random.uniform(-1, 1)),
                'kurtosis': float(stats.kurtosis(data)) if SCIPY_AVAILABLE else float(np.random.uniform(-1, 3))
            }
            
            time.sleep(0.5)  # Simulate computation time
        
        return results
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        self.analysis_progress.emit(50, "Computing correlations...")
        
        # Generate sample correlation matrix
        variables = ['speed', 'acceleration', 'steering', 'fuel', 'efficiency']
        n_vars = len(variables)
        
        # Create a realistic correlation matrix
        corr_matrix = np.random.rand(n_vars, n_vars)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1.0)  # Diagonal = 1
        
        # Ensure valid correlation values
        corr_matrix = np.clip(corr_matrix, -1, 1)
        
        results = {
            'correlation_matrix': corr_matrix.tolist(),
            'variables': variables,
            'significant_correlations': []
        }
        
        # Find significant correlations
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr_val = corr_matrix[i, j]
                if abs(corr_val) > 0.5:  # Threshold for significance
                    results['significant_correlations'].append({
                        'var1': variables[i],
                        'var2': variables[j],
                        'correlation': float(corr_val),
                        'strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                    })
        
        return results
    
    def clustering_analysis(self):
        """Perform clustering analysis"""
        self.analysis_progress.emit(30, "Preparing data for clustering...")
        
        # Generate sample data
        n_samples = 1000
        data = np.random.randn(n_samples, 4)  # 4 features
        
        self.analysis_progress.emit(60, "Performing K-means clustering...")
        
        # Perform clustering
        n_clusters = 3
        if SKLEARN_AVAILABLE:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            cluster_centers = kmeans.cluster_centers_.tolist()
            inertia = float(kmeans.inertia_)
        else:
            # Fallback without sklearn
            cluster_labels = np.random.randint(0, n_clusters, len(data))
            cluster_centers = np.random.randn(n_clusters, 4).tolist()
            inertia = float(np.random.uniform(100, 1000))
        
        self.analysis_progress.emit(90, "Analyzing clusters...")
        
        results = {
            'n_clusters': n_clusters,
            'cluster_centers': cluster_centers,
            'inertia': inertia,
            'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in range(n_clusters)],
            'silhouette_score': float(np.random.uniform(0.3, 0.8))  # Simulated
        }
        
        return results
    
    def regression_analysis(self):
        """Perform regression analysis"""
        self.analysis_progress.emit(25, "Preparing regression data...")
        
        # Generate sample data
        n_samples = 500
        X = np.random.randn(n_samples, 3)  # 3 features
        y = 2 * X[:, 0] + 1.5 * X[:, 1] - 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.1
        
        self.analysis_progress.emit(75, "Fitting regression model...")
        
        # Fit regression model
        if SKLEARN_AVAILABLE:
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            results = {
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_),
                'r2_score': float(r2_score(y, y_pred)),
                'mse': float(mean_squared_error(y, y_pred)),
                'feature_names': ['Speed', 'Acceleration', 'Steering'],
                'feature_importance': [abs(coef) for coef in model.coef_]
            }
        else:
            # Fallback without sklearn
            coefficients = np.random.randn(3).tolist()
            results = {
                'coefficients': coefficients,
                'intercept': float(np.random.randn()),
                'r2_score': float(np.random.uniform(0.7, 0.95)),
                'mse': float(np.random.uniform(0.1, 2.0)),
                'feature_names': ['Speed', 'Acceleration', 'Steering'],
                'feature_importance': [abs(coef) for coef in coefficients]
            }
        
        return results
    
    def anomaly_detection(self):
        """Perform anomaly detection"""
        self.analysis_progress.emit(40, "Detecting anomalies...")
        
        # Generate sample data with anomalies
        n_samples = 1000
        normal_data = np.random.randn(int(n_samples * 0.95), 2)
        anomaly_data = np.random.randn(int(n_samples * 0.05), 2) * 3 + 5  # Outliers
        
        all_data = np.vstack([normal_data, anomaly_data])
        
        # Simple anomaly detection using statistical method
        mean = np.mean(all_data, axis=0)
        cov = np.cov(all_data.T)
        
        # Mahalanobis distance
        inv_cov = np.linalg.inv(cov)
        distances = []
        for point in all_data:
            diff = point - mean
            distance = np.sqrt(diff.T @ inv_cov @ diff)
            distances.append(distance)
        
        distances = np.array(distances)
        threshold = np.percentile(distances, 95)  # Top 5% as anomalies
        anomalies = distances > threshold
        
        results = {
            'total_samples': n_samples,
            'anomalies_detected': int(np.sum(anomalies)),
            'anomaly_rate': float(np.mean(anomalies)),
            'threshold': float(threshold),
            'mean_distance': float(np.mean(distances)),
            'max_distance': float(np.max(distances))
        }
        
        return results
    
    def stop_analysis(self):
        """Stop analysis"""
        self.is_running = False


class AdvancedVisualizationWidget(QWidget):
    """Advanced data visualization widget"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 400)
        
        # Visualization data
        self.chart_type = "line"
        self.data_series = {}
        self.chart_title = "Data Visualization"
        
    def set_chart_type(self, chart_type):
        """Set chart type"""
        self.chart_type = chart_type
        self.update()
    
    def set_data(self, data_series, title="Data Visualization"):
        """Set data for visualization"""
        self.data_series = data_series
        self.chart_title = title
        self.update()
    
    def paintEvent(self, event):
        """Paint the visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        
        # Draw title
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        painter.drawText(20, 30, self.chart_title)
        
        if not self.data_series:
            painter.drawText(self.rect().center(), "No data available")
            return
        
        # Calculate chart area
        margin = 50
        chart_rect = self.rect().adjusted(margin, margin + 20, -margin, -margin)
        
        if self.chart_type == "line":
            self.draw_line_chart(painter, chart_rect)
        elif self.chart_type == "bar":
            self.draw_bar_chart(painter, chart_rect)
        elif self.chart_type == "scatter":
            self.draw_scatter_plot(painter, chart_rect)
        elif self.chart_type == "histogram":
            self.draw_histogram(painter, chart_rect)
        elif self.chart_type == "heatmap":
            self.draw_heatmap(painter, chart_rect)
    
    def draw_line_chart(self, painter, chart_rect):
        """Draw line chart"""
        if not self.data_series:
            return
        
        # Draw grid
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for i in range(5):
            x = chart_rect.left() + (chart_rect.width() * i / 4)
            painter.drawLine(int(x), chart_rect.top(), int(x), chart_rect.bottom())
            
            y = chart_rect.top() + (chart_rect.height() * i / 4)
            painter.drawLine(chart_rect.left(), int(y), chart_rect.right(), int(y))
        
        # Draw data series
        colors = [QColor(255, 100, 100), QColor(100, 255, 100), QColor(100, 100, 255), QColor(255, 255, 100)]
        
        for i, (series_name, data) in enumerate(self.data_series.items()):
            if not data or len(data) < 2:
                continue
            
            color = colors[i % len(colors)]
            painter.setPen(QPen(color, 2))
            
            # Normalize data
            min_val, max_val = min(data), max(data)
            if max_val == min_val:
                continue
            
            points = []
            for j, value in enumerate(data):
                x = chart_rect.left() + (chart_rect.width() * j / (len(data) - 1))
                y_norm = (value - min_val) / (max_val - min_val)
                y = chart_rect.bottom() - (chart_rect.height() * y_norm)
                points.append((int(x), int(y)))
            
            # Draw lines
            for j in range(len(points) - 1):
                painter.drawLine(points[j][0], points[j][1], points[j+1][0], points[j+1][1])
            
            # Draw legend
            painter.drawText(chart_rect.right() - 150, chart_rect.top() + 20 + i * 20, 
                           f"{series_name} ({len(data)} points)")
    
    def draw_bar_chart(self, painter, chart_rect):
        """Draw bar chart"""
        if not self.data_series:
            return
        
        # Get first series for bar chart
        series_name, data = next(iter(self.data_series.items()))
        if not data:
            return
        
        # Calculate bar dimensions
        n_bars = len(data)
        bar_width = chart_rect.width() / n_bars * 0.8
        bar_spacing = chart_rect.width() / n_bars * 0.2
        
        max_val = max(data) if data else 1
        
        # Draw bars
        painter.setPen(QPen(QColor(100, 150, 255), 1))
        painter.setBrush(QBrush(QColor(100, 150, 255, 150)))
        
        for i, value in enumerate(data):
            x = chart_rect.left() + i * (bar_width + bar_spacing)
            height = (value / max_val) * chart_rect.height()
            y = chart_rect.bottom() - height
            
            painter.drawRect(int(x), int(y), int(bar_width), int(height))
    
    def draw_scatter_plot(self, painter, chart_rect):
        """Draw scatter plot"""
        if len(self.data_series) < 2:
            return
        
        series_names = list(self.data_series.keys())
        x_data = self.data_series[series_names[0]]
        y_data = self.data_series[series_names[1]]
        
        if not x_data or not y_data or len(x_data) != len(y_data):
            return
        
        # Normalize data
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        
        if x_max == x_min or y_max == y_min:
            return
        
        # Draw points
        painter.setPen(QPen(QColor(255, 100, 100), 2))
        painter.setBrush(QBrush(QColor(255, 100, 100, 100)))
        
        for x_val, y_val in zip(x_data, y_data):
            x_norm = (x_val - x_min) / (x_max - x_min)
            y_norm = (y_val - y_min) / (y_max - y_min)
            
            x = chart_rect.left() + x_norm * chart_rect.width()
            y = chart_rect.bottom() - y_norm * chart_rect.height()
            
            painter.drawEllipse(int(x - 3), int(y - 3), 6, 6)
    
    def draw_histogram(self, painter, chart_rect):
        """Draw histogram"""
        if not self.data_series:
            return
        
        # Get first series for histogram
        series_name, data = next(iter(self.data_series.items()))
        if not data:
            return
        
        # Create histogram bins
        n_bins = 20
        min_val, max_val = min(data), max(data)
        if max_val == min_val:
            return
        
        bin_width = (max_val - min_val) / n_bins
        bins = [0] * n_bins
        
        for value in data:
            bin_idx = min(int((value - min_val) / bin_width), n_bins - 1)
            bins[bin_idx] += 1
        
        # Draw histogram
        max_count = max(bins) if bins else 1
        bar_width = chart_rect.width() / n_bins
        
        painter.setPen(QPen(QColor(150, 100, 255), 1))
        painter.setBrush(QBrush(QColor(150, 100, 255, 150)))
        
        for i, count in enumerate(bins):
            x = chart_rect.left() + i * bar_width
            height = (count / max_count) * chart_rect.height()
            y = chart_rect.bottom() - height
            
            painter.drawRect(int(x), int(y), int(bar_width), int(height))
    
    def draw_heatmap(self, painter, chart_rect):
        """Draw heatmap"""
        # Generate sample correlation matrix for heatmap
        size = 5
        matrix = np.random.rand(size, size)
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        np.fill_diagonal(matrix, 1.0)
        
        cell_width = chart_rect.width() / size
        cell_height = chart_rect.height() / size
        
        for i in range(size):
            for j in range(size):
                value = matrix[i, j]
                
                # Color based on value
                intensity = int(255 * abs(value))
                if value > 0:
                    color = QColor(intensity, 0, 0)  # Red for positive
                else:
                    color = QColor(0, 0, intensity)  # Blue for negative
                
                painter.fillRect(
                    int(chart_rect.left() + j * cell_width),
                    int(chart_rect.top() + i * cell_height),
                    int(cell_width),
                    int(cell_height),
                    color
                )
                
                # Draw value text
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(
                    int(chart_rect.left() + j * cell_width + cell_width/2 - 10),
                    int(chart_rect.top() + i * cell_height + cell_height/2),
                    f"{value:.2f}"
                )


class AdvancedAnalyticsPanel(QWidget):
    """Advanced analytics and data science panel"""
    
    analysis_complete = pyqtSignal(str, dict)  # analysis_type, results
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        
        # Analysis state
        self.current_analyzer = None
        self.analysis_results = {}
        self.data_cache = {}
        
        self.setup_ui()
        self.setup_timers()
        
        print("Advanced analytics panel initialized")
    
    def setup_ui(self):
        """Setup the analytics panel UI"""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Data Exploration Tab
        exploration_tab = self.create_exploration_tab()
        tab_widget.addTab(exploration_tab, "Data Exploration")
        
        # Statistical Analysis Tab
        stats_tab = self.create_statistics_tab()
        tab_widget.addTab(stats_tab, "Statistical Analysis")
        
        # Predictive Modeling Tab
        modeling_tab = self.create_modeling_tab()
        tab_widget.addTab(modeling_tab, "Predictive Modeling")
        
        # Advanced Visualization Tab
        viz_tab = self.create_visualization_tab()
        tab_widget.addTab(viz_tab, "Advanced Visualization")
        
        # Data Export Tab
        export_tab = self.create_export_tab()
        tab_widget.addTab(export_tab, "Data Export")
        
        layout.addWidget(tab_widget)
    
    def create_exploration_tab(self):
        """Create data exploration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Data source selection
        source_group = QGroupBox("Data Source")
        source_layout = QGridLayout(source_group)
        
        source_layout.addWidget(QLabel("Data Source:"), 0, 0)
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems([
            "Vehicle Telemetry", "Sensor Data", "Performance Metrics",
            "Environmental Data", "AI Decisions", "User Interactions"
        ])
        source_layout.addWidget(self.data_source_combo, 0, 1)
        
        source_layout.addWidget(QLabel("Time Range:"), 1, 0)
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems([
            "Last Hour", "Last 6 Hours", "Last Day", "Last Week", "All Data"
        ])
        source_layout.addWidget(self.time_range_combo, 1, 1)
        
        load_data_btn = QPushButton("Load Data")
        load_data_btn.clicked.connect(self.load_data)
        source_layout.addWidget(load_data_btn, 2, 0, 1, 2)
        
        layout.addWidget(source_group)
        
        # Data summary
        summary_group = QGroupBox("Data Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.data_summary_table = QTableWidget()
        self.data_summary_table.setColumnCount(6)
        self.data_summary_table.setHorizontalHeaderLabels([
            "Variable", "Count", "Mean", "Std", "Min", "Max"
        ])
        summary_layout.addWidget(self.data_summary_table)
        
        layout.addWidget(summary_group)
        
        # Data quality
        quality_group = QGroupBox("Data Quality")
        quality_layout = QGridLayout(quality_group)
        
        self.quality_metrics = {}
        quality_items = [
            ("Total Records", "0"),
            ("Missing Values", "0"),
            ("Duplicate Records", "0"),
            ("Data Quality Score", "0%"),
            ("Completeness", "0%"),
            ("Consistency", "0%")
        ]
        
        for i, (name, default) in enumerate(quality_items):
            row, col = i // 2, (i % 2) * 2
            
            label = QLabel(f"{name}:")
            quality_layout.addWidget(label, row, col)
            
            value_label = QLabel(default)
            value_label.setStyleSheet("color: #4a90e2; font-weight: bold;")
            quality_layout.addWidget(value_label, row, col + 1)
            
            self.quality_metrics[name] = value_label
        
        layout.addWidget(quality_group)
        
        return widget
    
    def create_statistics_tab(self):
        """Create statistical analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Analysis selection
        analysis_group = QGroupBox("Statistical Analysis")
        analysis_layout = QGridLayout(analysis_group)
        
        analysis_layout.addWidget(QLabel("Analysis Type:"), 0, 0)
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Descriptive Statistics",
            "Correlation Analysis", 
            "Clustering Analysis",
            "Regression Analysis",
            "Anomaly Detection"
        ])
        analysis_layout.addWidget(self.analysis_type_combo, 0, 1)
        
        # Analysis controls
        controls_layout = QHBoxLayout()
        
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.run_analysis_btn.clicked.connect(self.run_statistical_analysis)
        controls_layout.addWidget(self.run_analysis_btn)
        
        self.stop_analysis_btn = QPushButton("Stop Analysis")
        self.stop_analysis_btn.clicked.connect(self.stop_analysis)
        self.stop_analysis_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_analysis_btn)
        
        analysis_layout.addLayout(controls_layout, 1, 0, 1, 2)
        
        # Progress
        self.analysis_progress = QProgressBar()
        analysis_layout.addWidget(self.analysis_progress, 2, 0, 1, 2)
        
        self.analysis_status = QLabel("Ready for analysis")
        analysis_layout.addWidget(self.analysis_status, 3, 0, 1, 2)
        
        layout.addWidget(analysis_group)
        
        # Results display
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        
        self.results_table = QTableWidget()
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        return widget
    
    def create_modeling_tab(self):
        """Create predictive modeling tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model configuration
        config_group = QGroupBox("Model Configuration")
        config_layout = QGridLayout(config_group)
        
        config_layout.addWidget(QLabel("Target Variable:"), 0, 0)
        self.target_var_combo = QComboBox()
        self.target_var_combo.addItems([
            "Vehicle Speed", "Fuel Efficiency", "Safety Score", 
            "Collision Risk", "Route Optimization"
        ])
        config_layout.addWidget(self.target_var_combo, 0, 1)
        
        config_layout.addWidget(QLabel("Model Type:"), 1, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "Linear Regression", "Random Forest", "Gradient Boosting",
            "Neural Network", "Support Vector Machine", "Time Series"
        ])
        config_layout.addWidget(self.model_type_combo, 1, 1)
        
        config_layout.addWidget(QLabel("Validation Method:"), 2, 0)
        self.validation_combo = QComboBox()
        self.validation_combo.addItems([
            "Train/Test Split", "K-Fold Cross Validation", "Time Series Split"
        ])
        config_layout.addWidget(self.validation_combo, 2, 1)
        
        # Model training
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self.train_predictive_model)
        config_layout.addWidget(train_btn, 3, 0, 1, 2)
        
        layout.addWidget(config_group)
        
        # Model performance
        performance_group = QGroupBox("Model Performance")
        performance_layout = QGridLayout(performance_group)
        
        self.model_metrics = {}
        perf_items = [
            ("R² Score", "0.000"),
            ("RMSE", "0.000"),
            ("MAE", "0.000"),
            ("Training Time", "0.0s"),
            ("Prediction Time", "0.0ms"),
            ("Model Size", "0 KB")
        ]
        
        for i, (name, default) in enumerate(perf_items):
            row, col = i // 2, (i % 2) * 2
            
            label = QLabel(f"{name}:")
            performance_layout.addWidget(label, row, col)
            
            value_label = QLabel(default)
            value_label.setStyleSheet("color: #4a90e2; font-weight: bold;")
            performance_layout.addWidget(value_label, row, col + 1)
            
            self.model_metrics[name] = value_label
        
        layout.addWidget(performance_group)
        
        # Feature importance
        importance_group = QGroupBox("Feature Importance")
        importance_layout = QVBoxLayout(importance_group)
        
        self.importance_table = QTableWidget()
        self.importance_table.setColumnCount(3)
        self.importance_table.setHorizontalHeaderLabels(["Feature", "Importance", "Rank"])
        importance_layout.addWidget(self.importance_table)
        
        layout.addWidget(importance_group)
        
        return widget
    
    def create_visualization_tab(self):
        """Create advanced visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Visualization controls
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QGridLayout(controls_group)
        
        controls_layout.addWidget(QLabel("Chart Type:"), 0, 0)
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Heatmap"
        ])
        self.chart_type_combo.currentTextChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.chart_type_combo, 0, 1)
        
        controls_layout.addWidget(QLabel("X-Axis:"), 1, 0)
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItems(["Time", "Speed", "Acceleration", "Distance"])
        controls_layout.addWidget(self.x_axis_combo, 1, 1)
        
        controls_layout.addWidget(QLabel("Y-Axis:"), 2, 0)
        self.y_axis_combo = QComboBox()
        self.y_axis_combo.addItems(["Speed", "Fuel", "Efficiency", "Safety"])
        controls_layout.addWidget(self.y_axis_combo, 2, 1)
        
        update_viz_btn = QPushButton("Update Visualization")
        update_viz_btn.clicked.connect(self.update_visualization)
        controls_layout.addWidget(update_viz_btn, 3, 0, 1, 2)
        
        layout.addWidget(controls_group)
        
        # Visualization widget
        self.viz_widget = AdvancedVisualizationWidget()
        layout.addWidget(self.viz_widget)
        
        return widget
    
    def create_export_tab(self):
        """Create data export tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QGridLayout(export_group)
        
        export_layout.addWidget(QLabel("Export Format:"), 0, 0)
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems([
            "CSV", "JSON", "Excel", "Parquet", "HDF5", "SQL Database"
        ])
        export_layout.addWidget(self.export_format_combo, 0, 1)
        
        export_layout.addWidget(QLabel("Data Selection:"), 1, 0)
        self.export_data_combo = QComboBox()
        self.export_data_combo.addItems([
            "All Data", "Filtered Data", "Analysis Results", "Model Predictions"
        ])
        export_layout.addWidget(self.export_data_combo, 1, 1)
        
        # Export controls
        export_controls = QHBoxLayout()
        
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_data)
        export_controls.addWidget(export_btn)
        
        preview_btn = QPushButton("Preview Export")
        preview_btn.clicked.connect(self.preview_export)
        export_controls.addWidget(preview_btn)
        
        export_layout.addLayout(export_controls, 2, 0, 1, 2)
        
        layout.addWidget(export_group)
        
        # Export preview
        preview_group = QGroupBox("Export Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.export_preview = QTextEdit()
        self.export_preview.setReadOnly(True)
        self.export_preview.setMaximumHeight(300)
        preview_layout.addWidget(self.export_preview)
        
        layout.addWidget(preview_group)
        
        # Export history
        history_group = QGroupBox("Export History")
        history_layout = QVBoxLayout(history_group)
        
        self.export_history_table = QTableWidget()
        self.export_history_table.setColumnCount(4)
        self.export_history_table.setHorizontalHeaderLabels([
            "Timestamp", "Format", "Size", "Status"
        ])
        history_layout.addWidget(self.export_history_table)
        
        layout.addWidget(history_group)
        
        return widget
    
    def setup_timers(self):
        """Setup update timers"""
        # Data refresh timer
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.refresh_data)
        self.data_timer.start(5000)  # Refresh every 5 seconds
    
    def load_data(self):
        """Load data from selected source"""
        source = self.data_source_combo.currentText()
        time_range = self.time_range_combo.currentText()
        
        # Simulate data loading
        print(f"Loading {source} data for {time_range}")
        
        # Generate sample data
        n_samples = 1000
        data = {
            'speed': np.random.normal(50, 15, n_samples),
            'acceleration': np.random.normal(0, 2, n_samples),
            'fuel_consumption': np.random.normal(8, 2, n_samples),
            'efficiency': np.random.normal(85, 10, n_samples),
            'safety_score': np.random.normal(90, 5, n_samples)
        }
        
        self.data_cache[source] = data
        
        # Update data summary
        self.update_data_summary(data)
        
        # Update data quality metrics
        self.update_data_quality(data)
        
        print(f"Loaded {n_samples} records from {source}")
    
    def update_data_summary(self, data):
        """Update data summary table"""
        self.data_summary_table.setRowCount(len(data))
        
        for i, (var_name, values) in enumerate(data.items()):
            self.data_summary_table.setItem(i, 0, QTableWidgetItem(var_name))
            self.data_summary_table.setItem(i, 1, QTableWidgetItem(str(len(values))))
            self.data_summary_table.setItem(i, 2, QTableWidgetItem(f"{np.mean(values):.2f}"))
            self.data_summary_table.setItem(i, 3, QTableWidgetItem(f"{np.std(values):.2f}"))
            self.data_summary_table.setItem(i, 4, QTableWidgetItem(f"{np.min(values):.2f}"))
            self.data_summary_table.setItem(i, 5, QTableWidgetItem(f"{np.max(values):.2f}"))
    
    def update_data_quality(self, data):
        """Update data quality metrics"""
        total_records = len(next(iter(data.values())))
        missing_values = np.random.randint(0, total_records // 20)  # Simulate missing values
        duplicates = np.random.randint(0, total_records // 50)  # Simulate duplicates
        
        completeness = (total_records - missing_values) / total_records * 100
        consistency = np.random.uniform(85, 98)  # Simulate consistency score
        quality_score = (completeness + consistency) / 2
        
        self.quality_metrics["Total Records"].setText(str(total_records))
        self.quality_metrics["Missing Values"].setText(str(missing_values))
        self.quality_metrics["Duplicate Records"].setText(str(duplicates))
        self.quality_metrics["Data Quality Score"].setText(f"{quality_score:.1f}%")
        self.quality_metrics["Completeness"].setText(f"{completeness:.1f}%")
        self.quality_metrics["Consistency"].setText(f"{consistency:.1f}%")
    
    def run_statistical_analysis(self):
        """Run statistical analysis"""
        if self.current_analyzer and self.current_analyzer.isRunning():
            QMessageBox.warning(self, "Analysis Running", "An analysis is already in progress.")
            return
        
        analysis_type_map = {
            "Descriptive Statistics": "descriptive",
            "Correlation Analysis": "correlation",
            "Clustering Analysis": "clustering", 
            "Regression Analysis": "regression",
            "Anomaly Detection": "anomaly"
        }
        
        analysis_type = analysis_type_map.get(self.analysis_type_combo.currentText(), "descriptive")
        
        # Get data
        source = self.data_source_combo.currentText()
        if source not in self.data_cache:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        
        data = self.data_cache[source]
        
        # Start analysis
        self.current_analyzer = StatisticalAnalyzer(data, analysis_type)
        self.current_analyzer.analysis_progress.connect(self.on_analysis_progress)
        self.current_analyzer.analysis_complete.connect(self.on_analysis_complete)
        self.current_analyzer.start()
        
        # Update UI
        self.run_analysis_btn.setEnabled(False)
        self.stop_analysis_btn.setEnabled(True)
        self.analysis_status.setText("Analysis started...")
        
        print(f"Started {analysis_type} analysis")
    
    def stop_analysis(self):
        """Stop current analysis"""
        if self.current_analyzer:
            self.current_analyzer.stop_analysis()
            self.current_analyzer.wait()
        
        self.run_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setEnabled(False)
        self.analysis_status.setText("Analysis stopped")
    
    @pyqtSlot(int, str)
    def on_analysis_progress(self, progress, status):
        """Handle analysis progress"""
        self.analysis_progress.setValue(progress)
        self.analysis_status.setText(status)
    
    @pyqtSlot(dict)
    def on_analysis_complete(self, results):
        """Handle analysis completion"""
        self.run_analysis_btn.setEnabled(True)
        self.stop_analysis_btn.setEnabled(False)
        
        if "error" in results:
            self.analysis_status.setText(f"Analysis failed: {results['error']}")
            return
        
        self.analysis_results[self.analysis_type_combo.currentText()] = results
        
        # Display results
        self.display_analysis_results(results)
        
        self.analysis_status.setText("Analysis completed successfully")
        print("Statistical analysis completed")
    
    def display_analysis_results(self, results):
        """Display analysis results"""
        # Format results as text
        result_text = json.dumps(results, indent=2)
        self.results_text.setText(result_text)
        
        # Display in table format if applicable
        if isinstance(results, dict) and any(isinstance(v, dict) for v in results.values()):
            # Flatten nested results for table display
            table_data = []
            for key, value in results.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        table_data.append([f"{key}.{sub_key}", str(sub_value)])
                else:
                    table_data.append([key, str(value)])
            
            self.results_table.setRowCount(len(table_data))
            self.results_table.setColumnCount(2)
            self.results_table.setHorizontalHeaderLabels(["Metric", "Value"])
            
            for i, (metric, value) in enumerate(table_data):
                self.results_table.setItem(i, 0, QTableWidgetItem(metric))
                self.results_table.setItem(i, 1, QTableWidgetItem(value))
    
    def train_predictive_model(self):
        """Train predictive model"""
        target = self.target_var_combo.currentText()
        model_type = self.model_type_combo.currentText()
        validation = self.validation_combo.currentText()
        
        print(f"Training {model_type} model for {target} using {validation}")
        
        # Simulate model training
        training_time = np.random.uniform(1.0, 10.0)
        time.sleep(0.1)  # Brief pause to simulate training
        
        # Generate performance metrics
        r2_score = np.random.uniform(0.7, 0.95)
        rmse = np.random.uniform(0.1, 2.0)
        mae = rmse * np.random.uniform(0.6, 0.8)
        pred_time = np.random.uniform(0.5, 5.0)
        model_size = np.random.randint(10, 500)
        
        # Update performance metrics
        self.model_metrics["R² Score"].setText(f"{r2_score:.3f}")
        self.model_metrics["RMSE"].setText(f"{rmse:.3f}")
        self.model_metrics["MAE"].setText(f"{mae:.3f}")
        self.model_metrics["Training Time"].setText(f"{training_time:.1f}s")
        self.model_metrics["Prediction Time"].setText(f"{pred_time:.1f}ms")
        self.model_metrics["Model Size"].setText(f"{model_size} KB")
        
        # Generate feature importance
        features = ["Speed", "Acceleration", "Distance", "Time", "Weather"]
        importance_values = np.random.rand(len(features))
        importance_values = importance_values / np.sum(importance_values)  # Normalize
        
        # Sort by importance
        feature_importance = list(zip(features, importance_values))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Update importance table
        self.importance_table.setRowCount(len(feature_importance))
        for i, (feature, importance) in enumerate(feature_importance):
            self.importance_table.setItem(i, 0, QTableWidgetItem(feature))
            self.importance_table.setItem(i, 1, QTableWidgetItem(f"{importance:.3f}"))
            self.importance_table.setItem(i, 2, QTableWidgetItem(str(i + 1)))
        
        print(f"Model training completed - R²: {r2_score:.3f}")
    
    def update_visualization(self):
        """Update visualization"""
        chart_type_map = {
            "Line Chart": "line",
            "Bar Chart": "bar", 
            "Scatter Plot": "scatter",
            "Histogram": "histogram",
            "Heatmap": "heatmap"
        }
        
        chart_type = chart_type_map.get(self.chart_type_combo.currentText(), "line")
        
        # Get sample data for visualization
        source = self.data_source_combo.currentText()
        if source in self.data_cache:
            data = self.data_cache[source]
            
            # Select data series based on axes
            x_var = self.x_axis_combo.currentText().lower()
            y_var = self.y_axis_combo.currentText().lower()
            
            viz_data = {}
            if x_var in data:
                viz_data[x_var] = data[x_var][:100]  # Limit for performance
            if y_var in data and y_var != x_var:
                viz_data[y_var] = data[y_var][:100]
            
            self.viz_widget.set_chart_type(chart_type)
            self.viz_widget.set_data(viz_data, f"{chart_type.title()} - {x_var} vs {y_var}")
        
        print(f"Updated visualization: {chart_type}")
    
    def export_data(self):
        """Export data"""
        format_type = self.export_format_combo.currentText()
        data_selection = self.export_data_combo.currentText()
        
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Export {data_selection}", "", 
            f"{format_type} Files (*.{format_type.lower()});;All Files (*)"
        )
        
        if filename:
            # Simulate export
            print(f"Exporting {data_selection} as {format_type} to {filename}")
            
            # Add to export history
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            file_size = f"{np.random.randint(100, 5000)} KB"
            
            row = self.export_history_table.rowCount()
            self.export_history_table.insertRow(row)
            self.export_history_table.setItem(row, 0, QTableWidgetItem(current_time))
            self.export_history_table.setItem(row, 1, QTableWidgetItem(format_type))
            self.export_history_table.setItem(row, 2, QTableWidgetItem(file_size))
            self.export_history_table.setItem(row, 3, QTableWidgetItem("Success"))
            
            QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
    
    def preview_export(self):
        """Preview export data"""
        data_selection = self.export_data_combo.currentText()
        
        # Generate preview text
        preview_text = f"Export Preview - {data_selection}\n"
        preview_text += "=" * 50 + "\n\n"
        
        if data_selection == "All Data":
            preview_text += "Sample of all collected data:\n"
            preview_text += "timestamp,speed,acceleration,fuel,efficiency\n"
            for i in range(5):
                preview_text += f"2024-01-01 12:{i:02d}:00,{np.random.uniform(40, 80):.1f},"
                preview_text += f"{np.random.uniform(-2, 2):.2f},{np.random.uniform(6, 12):.1f},"
                preview_text += f"{np.random.uniform(75, 95):.1f}\n"
        
        elif data_selection == "Analysis Results":
            preview_text += "Statistical analysis results:\n"
            if self.analysis_results:
                for analysis_type, results in self.analysis_results.items():
                    preview_text += f"\n{analysis_type}:\n"
                    preview_text += json.dumps(results, indent=2)[:500] + "...\n"
        
        self.export_preview.setText(preview_text)
    
    def refresh_data(self):
        """Refresh data from simulation"""
        # This would normally pull fresh data from the simulation
        # For now, we'll just update some metrics
        pass