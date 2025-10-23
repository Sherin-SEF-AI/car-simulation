"""
Advanced Machine Learning Panel
Real-time ML model training, evaluation, and deployment for autonomous vehicles
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QPushButton, QComboBox, QSlider, QCheckBox, QTextEdit,
    QProgressBar, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QSplitter, QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont

import numpy as np
import json
import time
from collections import deque
import threading


class MLModelTrainer(QThread):
    """Background thread for ML model training"""
    
    training_progress = pyqtSignal(int, str)  # progress, status
    training_complete = pyqtSignal(dict)      # results
    
    def __init__(self, model_config, training_data):
        super().__init__()
        self.model_config = model_config
        self.training_data = training_data
        self.is_training = False
    
    def run(self):
        """Run training process"""
        self.is_training = True
        
        try:
            # Simulate training process
            epochs = self.model_config.get('epochs', 100)
            
            for epoch in range(epochs):
                if not self.is_training:
                    break
                
                # Simulate training step
                time.sleep(0.1)  # Simulate computation time
                
                # Calculate progress
                progress = int((epoch + 1) / epochs * 100)
                status = f"Epoch {epoch + 1}/{epochs} - Loss: {np.random.uniform(0.1, 1.0):.4f}"
                
                self.training_progress.emit(progress, status)
            
            # Training complete
            if self.is_training:
                results = {
                    'final_loss': np.random.uniform(0.05, 0.2),
                    'accuracy': np.random.uniform(0.85, 0.98),
                    'training_time': epochs * 0.1,
                    'model_size': np.random.randint(1, 50)  # MB
                }
                self.training_complete.emit(results)
                
        except Exception as e:
            self.training_progress.emit(0, f"Training failed: {str(e)}")
    
    def stop_training(self):
        """Stop training process"""
        self.is_training = False


class ModelPerformanceChart(QWidget):
    """Chart for displaying model performance metrics"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 200)
        
        # Performance data
        self.loss_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        self.epochs = deque(maxlen=100)
        
    def add_data_point(self, epoch, loss, accuracy):
        """Add a new data point"""
        self.epochs.append(epoch)
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        self.update()
    
    def paintEvent(self, event):
        """Paint the performance chart"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(25, 25, 25))
        
        if len(self.loss_history) < 2:
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(self.rect().center(), "No data available")
            return
        
        # Calculate chart area
        margin = 40
        chart_rect = self.rect().adjusted(margin, margin, -margin, -margin)
        
        # Draw grid
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for i in range(5):
            x = chart_rect.left() + (chart_rect.width() * i / 4)
            painter.drawLine(int(x), chart_rect.top(), int(x), chart_rect.bottom())
            
            y = chart_rect.top() + (chart_rect.height() * i / 4)
            painter.drawLine(chart_rect.left(), int(y), chart_rect.right(), int(y))
        
        # Draw loss curve (red)
        if self.loss_history:
            painter.setPen(QPen(QColor(255, 100, 100), 2))
            max_loss = max(self.loss_history) if self.loss_history else 1
            
            points = []
            for i, loss in enumerate(self.loss_history):
                x = chart_rect.left() + (chart_rect.width() * i / max(1, len(self.loss_history) - 1))
                y = chart_rect.bottom() - (chart_rect.height() * loss / max_loss)
                points.append((int(x), int(y)))
            
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
        
        # Draw accuracy curve (green)
        if self.accuracy_history:
            painter.setPen(QPen(QColor(100, 255, 100), 2))
            
            points = []
            for i, acc in enumerate(self.accuracy_history):
                x = chart_rect.left() + (chart_rect.width() * i / max(1, len(self.accuracy_history) - 1))
                y = chart_rect.bottom() - (chart_rect.height() * acc)
                points.append((int(x), int(y)))
            
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
        
        # Draw legend
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(10, 25, "Training Progress")
        
        # Loss legend
        painter.setPen(QPen(QColor(255, 100, 100), 2))
        painter.drawLine(10, 35, 30, 35)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(35, 38, "Loss")
        
        # Accuracy legend
        painter.setPen(QPen(QColor(100, 255, 100), 2))
        painter.drawLine(10, 50, 30, 50)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(35, 53, "Accuracy")


class AdvancedMLPanel(QWidget):
    """Advanced Machine Learning control panel"""
    
    model_deployed = pyqtSignal(str, dict)  # model_name, config
    
    def __init__(self, simulation_app):
        super().__init__()
        self.simulation_app = simulation_app
        
        # ML state
        self.current_trainer = None
        self.trained_models = {}
        self.active_models = {}
        
        self.setup_ui()
        self.setup_timers()
        
        print("Advanced ML panel initialized")
    
    def setup_ui(self):
        """Setup the ML panel UI"""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Model Training Tab
        training_tab = self.create_training_tab()
        tab_widget.addTab(training_tab, "Model Training")
        
        # Model Evaluation Tab
        evaluation_tab = self.create_evaluation_tab()
        tab_widget.addTab(evaluation_tab, "Evaluation")
        
        # Deployment Tab
        deployment_tab = self.create_deployment_tab()
        tab_widget.addTab(deployment_tab, "Deployment")
        
        # Real-time Inference Tab
        inference_tab = self.create_inference_tab()
        tab_widget.addTab(inference_tab, "Real-time Inference")
        
        layout.addWidget(tab_widget)
    
    def create_training_tab(self):
        """Create model training tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model configuration
        config_group = QGroupBox("Model Configuration")
        config_layout = QGridLayout(config_group)
        
        # Model type
        config_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "Deep Q-Network (DQN)",
            "Policy Gradient",
            "Actor-Critic",
            "Convolutional Neural Network",
            "Recurrent Neural Network",
            "Transformer",
            "Custom Architecture"
        ])
        config_layout.addWidget(self.model_type_combo, 0, 1)
        
        # Training parameters
        config_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 1.0)
        self.learning_rate_spin.setValue(0.001)
        self.learning_rate_spin.setDecimals(4)
        config_layout.addWidget(self.learning_rate_spin, 1, 1)
        
        config_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        self.batch_size_spin.setValue(32)
        config_layout.addWidget(self.batch_size_spin, 2, 1)
        
        config_layout.addWidget(QLabel("Epochs:"), 3, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        config_layout.addWidget(self.epochs_spin, 3, 1)
        
        # Advanced options
        self.use_gpu_check = QCheckBox("Use GPU Acceleration")
        self.use_gpu_check.setChecked(True)
        config_layout.addWidget(self.use_gpu_check, 4, 0, 1, 2)
        
        self.data_augmentation_check = QCheckBox("Enable Data Augmentation")
        self.data_augmentation_check.setChecked(True)
        config_layout.addWidget(self.data_augmentation_check, 5, 0, 1, 2)
        
        layout.addWidget(config_group)
        
        # Training controls
        controls_group = QGroupBox("Training Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self.start_training)
        controls_layout.addWidget(self.start_training_btn)
        
        self.stop_training_btn = QPushButton("Stop Training")
        self.stop_training_btn.clicked.connect(self.stop_training)
        self.stop_training_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_training_btn)
        
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        controls_layout.addWidget(self.save_model_btn)
        
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        controls_layout.addWidget(self.load_model_btn)
        
        layout.addWidget(controls_group)
        
        # Training progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.training_progress = QProgressBar()
        progress_layout.addWidget(self.training_progress)
        
        self.training_status = QLabel("Ready to train")
        progress_layout.addWidget(self.training_status)
        
        # Performance chart
        self.performance_chart = ModelPerformanceChart()
        progress_layout.addWidget(self.performance_chart)
        
        layout.addWidget(progress_group)
        
        return widget
    
    def create_evaluation_tab(self):
        """Create model evaluation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model selection
        selection_group = QGroupBox("Model Selection")
        selection_layout = QHBoxLayout(selection_group)
        
        selection_layout.addWidget(QLabel("Model:"))
        self.eval_model_combo = QComboBox()
        selection_layout.addWidget(self.eval_model_combo)
        
        evaluate_btn = QPushButton("Evaluate")
        evaluate_btn.clicked.connect(self.evaluate_model)
        selection_layout.addWidget(evaluate_btn)
        
        layout.addWidget(selection_group)
        
        # Evaluation metrics
        metrics_group = QGroupBox("Evaluation Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        # Create metric displays
        self.metric_labels = {}
        metrics = [
            ("Accuracy", "0.00%"),
            ("Precision", "0.00%"),
            ("Recall", "0.00%"),
            ("F1-Score", "0.00%"),
            ("Mean Squared Error", "0.000"),
            ("Mean Absolute Error", "0.000"),
            ("R² Score", "0.000"),
            ("Inference Time", "0.0 ms")
        ]
        
        for i, (name, default) in enumerate(metrics):
            row, col = i // 2, (i % 2) * 2
            
            label = QLabel(f"{name}:")
            metrics_layout.addWidget(label, row, col)
            
            value_label = QLabel(default)
            value_label.setStyleSheet("color: #4a90e2; font-weight: bold;")
            metrics_layout.addWidget(value_label, row, col + 1)
            
            self.metric_labels[name] = value_label
        
        layout.addWidget(metrics_group)
        
        # Confusion matrix / results
        results_group = QGroupBox("Detailed Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Value", "Threshold", "Status"])
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        return widget
    
    def create_deployment_tab(self):
        """Create model deployment tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Available models
        models_group = QGroupBox("Available Models")
        models_layout = QVBoxLayout(models_group)
        
        self.models_tree = QTreeWidget()
        self.models_tree.setHeaderLabels(["Model", "Type", "Accuracy", "Size", "Status"])
        models_layout.addWidget(self.models_tree)
        
        # Model actions
        actions_layout = QHBoxLayout()
        
        deploy_btn = QPushButton("Deploy Model")
        deploy_btn.clicked.connect(self.deploy_model)
        actions_layout.addWidget(deploy_btn)
        
        undeploy_btn = QPushButton("Undeploy Model")
        undeploy_btn.clicked.connect(self.undeploy_model)
        actions_layout.addWidget(undeploy_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_models)
        actions_layout.addWidget(refresh_btn)
        
        models_layout.addLayout(actions_layout)
        layout.addWidget(models_group)
        
        # Deployment configuration
        deploy_config_group = QGroupBox("Deployment Configuration")
        deploy_config_layout = QGridLayout(deploy_config_group)
        
        deploy_config_layout.addWidget(QLabel("Target Vehicle:"), 0, 0)
        self.target_vehicle_combo = QComboBox()
        self.target_vehicle_combo.addItems(["All Vehicles", "Selected Vehicle", "Vehicle Type"])
        deploy_config_layout.addWidget(self.target_vehicle_combo, 0, 1)
        
        deploy_config_layout.addWidget(QLabel("Inference Mode:"), 1, 0)
        self.inference_mode_combo = QComboBox()
        self.inference_mode_combo.addItems(["Real-time", "Batch", "On-demand"])
        deploy_config_layout.addWidget(self.inference_mode_combo, 1, 1)
        
        deploy_config_layout.addWidget(QLabel("Update Frequency:"), 2, 0)
        self.update_freq_spin = QSpinBox()
        self.update_freq_spin.setRange(1, 1000)
        self.update_freq_spin.setValue(10)
        self.update_freq_spin.setSuffix(" Hz")
        deploy_config_layout.addWidget(self.update_freq_spin, 2, 1)
        
        layout.addWidget(deploy_config_group)
        
        # Active deployments
        active_group = QGroupBox("Active Deployments")
        active_layout = QVBoxLayout(active_group)
        
        self.active_deployments_table = QTableWidget()
        self.active_deployments_table.setColumnCount(5)
        self.active_deployments_table.setHorizontalHeaderLabels([
            "Model", "Vehicle", "Mode", "Frequency", "Performance"
        ])
        active_layout.addWidget(self.active_deployments_table)
        
        layout.addWidget(active_group)
        
        return widget
    
    def create_inference_tab(self):
        """Create real-time inference tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Real-time monitoring
        monitoring_group = QGroupBox("Real-time Monitoring")
        monitoring_layout = QVBoxLayout(monitoring_group)
        
        # Inference statistics
        stats_layout = QGridLayout()
        
        self.inference_stats = {}
        stats = [
            ("Inferences/sec", "0"),
            ("Avg Latency", "0.0 ms"),
            ("Max Latency", "0.0 ms"),
            ("Error Rate", "0.0%"),
            ("GPU Utilization", "0%"),
            ("Memory Usage", "0 MB")
        ]
        
        for i, (name, default) in enumerate(stats):
            row, col = i // 2, (i % 2) * 2
            
            label = QLabel(f"{name}:")
            stats_layout.addWidget(label, row, col)
            
            value_label = QLabel(default)
            value_label.setStyleSheet("color: #4a90e2; font-weight: bold;")
            stats_layout.addWidget(value_label, row, col + 1)
            
            self.inference_stats[name] = value_label
        
        monitoring_layout.addLayout(stats_layout)
        layout.addWidget(monitoring_group)
        
        # Live predictions
        predictions_group = QGroupBox("Live Predictions")
        predictions_layout = QVBoxLayout(predictions_group)
        
        self.predictions_text = QTextEdit()
        self.predictions_text.setMaximumHeight(200)
        self.predictions_text.setReadOnly(True)
        self.predictions_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New', monospace;
                font-size: 9px;
            }
        """)
        predictions_layout.addWidget(self.predictions_text)
        
        layout.addWidget(predictions_group)
        
        # Model comparison
        comparison_group = QGroupBox("Model Comparison")
        comparison_layout = QVBoxLayout(comparison_group)
        
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(6)
        self.comparison_table.setHorizontalHeaderLabels([
            "Model", "Accuracy", "Latency", "Throughput", "Memory", "Score"
        ])
        comparison_layout.addWidget(self.comparison_table)
        
        layout.addWidget(comparison_group)
        
        return widget
    
    def setup_timers(self):
        """Setup update timers"""
        # Inference monitoring timer
        self.inference_timer = QTimer()
        self.inference_timer.timeout.connect(self.update_inference_stats)
        self.inference_timer.start(1000)  # Update every second
        
        # Model refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_models)
        self.refresh_timer.start(10000)  # Refresh every 10 seconds
    
    def start_training(self):
        """Start model training"""
        if self.current_trainer and self.current_trainer.isRunning():
            QMessageBox.warning(self, "Training in Progress", "A training session is already running.")
            return
        
        # Get training configuration
        config = {
            'model_type': self.model_type_combo.currentText(),
            'learning_rate': self.learning_rate_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'epochs': self.epochs_spin.value(),
            'use_gpu': self.use_gpu_check.isChecked(),
            'data_augmentation': self.data_augmentation_check.isChecked()
        }
        
        # Generate dummy training data
        training_data = np.random.randn(1000, 10)  # Dummy data
        
        # Start training thread
        self.current_trainer = MLModelTrainer(config, training_data)
        self.current_trainer.training_progress.connect(self.on_training_progress)
        self.current_trainer.training_complete.connect(self.on_training_complete)
        self.current_trainer.start()
        
        # Update UI
        self.start_training_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(True)
        self.training_status.setText("Training started...")
        
        print(f"Started training {config['model_type']} model")
    
    def stop_training(self):
        """Stop model training"""
        if self.current_trainer:
            self.current_trainer.stop_training()
            self.current_trainer.wait()
        
        # Update UI
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.training_status.setText("Training stopped")
    
    @pyqtSlot(int, str)
    def on_training_progress(self, progress, status):
        """Handle training progress updates"""
        self.training_progress.setValue(progress)
        self.training_status.setText(status)
        
        # Extract metrics from status (simplified)
        if "Loss:" in status:
            try:
                loss = float(status.split("Loss: ")[1])
                accuracy = np.random.uniform(0.7, 0.95)  # Simulated accuracy
                epoch = int(status.split("Epoch ")[1].split("/")[0])
                
                self.performance_chart.add_data_point(epoch, loss, accuracy)
            except:
                pass
    
    @pyqtSlot(dict)
    def on_training_complete(self, results):
        """Handle training completion"""
        self.start_training_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        
        # Display results
        status = f"Training complete! Loss: {results['final_loss']:.4f}, "
        status += f"Accuracy: {results['accuracy']:.2%}, "
        status += f"Time: {results['training_time']:.1f}s"
        
        self.training_status.setText(status)
        
        # Save model to trained models
        model_name = f"{self.model_type_combo.currentText()}_{int(time.time())}"
        self.trained_models[model_name] = {
            'config': {
                'model_type': self.model_type_combo.currentText(),
                'learning_rate': self.learning_rate_spin.value(),
                'batch_size': self.batch_size_spin.value(),
                'epochs': self.epochs_spin.value()
            },
            'results': results,
            'timestamp': time.time()
        }
        
        # Update model lists
        self.refresh_models()
        
        print(f"Training completed: {model_name}")
    
    def evaluate_model(self):
        """Evaluate selected model"""
        model_name = self.eval_model_combo.currentText()
        if not model_name or model_name not in self.trained_models:
            QMessageBox.warning(self, "No Model", "Please select a model to evaluate.")
            return
        
        # Simulate evaluation
        model = self.trained_models[model_name]
        
        # Generate evaluation metrics
        metrics = {
            'Accuracy': np.random.uniform(0.85, 0.98),
            'Precision': np.random.uniform(0.80, 0.95),
            'Recall': np.random.uniform(0.75, 0.92),
            'F1-Score': np.random.uniform(0.78, 0.93),
            'Mean Squared Error': np.random.uniform(0.01, 0.1),
            'Mean Absolute Error': np.random.uniform(0.005, 0.05),
            'R² Score': np.random.uniform(0.85, 0.98),
            'Inference Time': np.random.uniform(1.0, 10.0)
        }
        
        # Update metric displays
        for name, value in metrics.items():
            if name in self.metric_labels:
                if name in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                    self.metric_labels[name].setText(f"{value:.2%}")
                elif name == 'Inference Time':
                    self.metric_labels[name].setText(f"{value:.1f} ms")
                else:
                    self.metric_labels[name].setText(f"{value:.3f}")
        
        # Update results table
        self.results_table.setRowCount(len(metrics))
        for i, (name, value) in enumerate(metrics.items()):
            self.results_table.setItem(i, 0, QTableWidgetItem(name))
            
            if name in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{value:.2%}"))
                threshold = "80%"
                status = "✓ Pass" if value > 0.8 else "✗ Fail"
            elif name == 'Inference Time':
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{value:.1f} ms"))
                threshold = "< 5ms"
                status = "✓ Pass" if value < 5.0 else "✗ Fail"
            else:
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{value:.3f}"))
                threshold = "< 0.05"
                status = "✓ Pass" if value < 0.05 else "✗ Fail"
            
            self.results_table.setItem(i, 2, QTableWidgetItem(threshold))
            self.results_table.setItem(i, 3, QTableWidgetItem(status))
        
        print(f"Evaluated model: {model_name}")
    
    def deploy_model(self):
        """Deploy selected model"""
        current_item = self.models_tree.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Model", "Please select a model to deploy.")
            return
        
        model_name = current_item.text(0)
        if model_name not in self.trained_models:
            return
        
        # Get deployment configuration
        target = self.target_vehicle_combo.currentText()
        mode = self.inference_mode_combo.currentText()
        frequency = self.update_freq_spin.value()
        
        # Deploy model
        deployment_id = f"{model_name}_{int(time.time())}"
        self.active_models[deployment_id] = {
            'model_name': model_name,
            'target': target,
            'mode': mode,
            'frequency': frequency,
            'deployed_at': time.time(),
            'performance': {
                'inferences': 0,
                'avg_latency': 0,
                'error_rate': 0
            }
        }
        
        # Update active deployments table
        self.update_active_deployments()
        
        # Emit deployment signal
        self.model_deployed.emit(model_name, self.trained_models[model_name])
        
        QMessageBox.information(self, "Deployment", f"Model {model_name} deployed successfully!")
        print(f"Deployed model: {model_name} to {target}")
    
    def undeploy_model(self):
        """Undeploy selected model"""
        current_row = self.active_deployments_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Deployment", "Please select a deployment to remove.")
            return
        
        # Get deployment ID from table
        model_name = self.active_deployments_table.item(current_row, 0).text()
        
        # Find and remove deployment
        to_remove = None
        for deployment_id, deployment in self.active_models.items():
            if deployment['model_name'] == model_name:
                to_remove = deployment_id
                break
        
        if to_remove:
            del self.active_models[to_remove]
            self.update_active_deployments()
            QMessageBox.information(self, "Undeployment", f"Model {model_name} undeployed successfully!")
            print(f"Undeployed model: {model_name}")
    
    def refresh_models(self):
        """Refresh model lists"""
        # Update evaluation combo
        self.eval_model_combo.clear()
        self.eval_model_combo.addItems(list(self.trained_models.keys()))
        
        # Update models tree
        self.models_tree.clear()
        for model_name, model_data in self.trained_models.items():
            item = QTreeWidgetItem([
                model_name,
                model_data['config']['model_type'],
                f"{model_data['results']['accuracy']:.1%}",
                f"{model_data['results']['model_size']} MB",
                "Ready"
            ])
            self.models_tree.addTopLevelItem(item)
    
    def update_active_deployments(self):
        """Update active deployments table"""
        self.active_deployments_table.setRowCount(len(self.active_models))
        
        for i, (deployment_id, deployment) in enumerate(self.active_models.items()):
            self.active_deployments_table.setItem(i, 0, QTableWidgetItem(deployment['model_name']))
            self.active_deployments_table.setItem(i, 1, QTableWidgetItem(deployment['target']))
            self.active_deployments_table.setItem(i, 2, QTableWidgetItem(deployment['mode']))
            self.active_deployments_table.setItem(i, 3, QTableWidgetItem(f"{deployment['frequency']} Hz"))
            
            # Performance indicator
            perf = deployment['performance']
            perf_text = f"Inf: {perf['inferences']}, Lat: {perf['avg_latency']:.1f}ms"
            self.active_deployments_table.setItem(i, 4, QTableWidgetItem(perf_text))
    
    def update_inference_stats(self):
        """Update real-time inference statistics"""
        if not self.active_models:
            return
        
        # Simulate inference statistics
        total_inferences = sum(np.random.randint(50, 200) for _ in self.active_models)
        avg_latency = np.random.uniform(1.0, 5.0)
        max_latency = avg_latency * np.random.uniform(1.5, 3.0)
        error_rate = np.random.uniform(0.0, 2.0)
        gpu_util = np.random.uniform(30, 90)
        memory_usage = np.random.uniform(100, 2000)
        
        # Update displays
        self.inference_stats["Inferences/sec"].setText(str(total_inferences))
        self.inference_stats["Avg Latency"].setText(f"{avg_latency:.1f} ms")
        self.inference_stats["Max Latency"].setText(f"{max_latency:.1f} ms")
        self.inference_stats["Error Rate"].setText(f"{error_rate:.1f}%")
        self.inference_stats["GPU Utilization"].setText(f"{gpu_util:.0f}%")
        self.inference_stats["Memory Usage"].setText(f"{memory_usage:.0f} MB")
        
        # Add live prediction
        if np.random.random() > 0.7:  # 30% chance
            timestamp = time.strftime("%H:%M:%S")
            vehicle_id = f"vehicle_{np.random.randint(1, 10)}"
            action = np.random.choice(["accelerate", "brake", "turn_left", "turn_right", "maintain"])
            confidence = np.random.uniform(0.7, 0.99)
            
            prediction = f"[{timestamp}] {vehicle_id}: {action} (confidence: {confidence:.2%})"
            self.predictions_text.append(prediction)
            
            # Keep only last 50 lines
            text = self.predictions_text.toPlainText()
            lines = text.split('\n')
            if len(lines) > 50:
                self.predictions_text.setPlainText('\n'.join(lines[-50:]))
    
    def save_model(self):
        """Save trained model to file"""
        if not self.trained_models:
            QMessageBox.warning(self, "No Models", "No trained models available to save.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "Model Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.trained_models, f, indent=2)
                QMessageBox.information(self, "Save Complete", f"Models saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save models: {str(e)}")
    
    def load_model(self):
        """Load trained model from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Model Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    loaded_models = json.load(f)
                
                self.trained_models.update(loaded_models)
                self.refresh_models()
                
                QMessageBox.information(self, "Load Complete", 
                                      f"Loaded {len(loaded_models)} models from {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load models: {str(e)}")