"""
Machine Learning integration framework for autonomous vehicle simulation
"""

import numpy as np
import json
import pickle
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of ML models supported"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    NEURAL_NETWORK = "neural_network"
    DECISION_TREE = "decision_tree"
    ENSEMBLE = "ensemble"

class ModelStatus(Enum):
    """Status of ML models"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    TRAINING = "training"
    ERROR = "error"

@dataclass
class ModelMetrics:
    """Performance metrics for ML models"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0  # Mean Squared Error for regression
    mae: float = 0.0  # Mean Absolute Error for regression
    inference_time: float = 0.0  # Average inference time in seconds
    training_time: float = 0.0  # Training time in seconds
    model_size: int = 0  # Model size in bytes
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mse': self.mse,
            'mae': self.mae,
            'inference_time': self.inference_time,
            'training_time': self.training_time,
            'model_size': self.model_size
        }

@dataclass
class TrainingData:
    """Training data container"""
    features: np.ndarray
    labels: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate training data after initialization"""
        if len(self.features) != len(self.labels):
            raise ValueError("Features and labels must have the same length")
        
        if len(self.features) == 0:
            raise ValueError("Training data cannot be empty")

@dataclass
class PredictionResult:
    """Result of model prediction"""
    prediction: Union[float, int, np.ndarray, List]
    confidence: float = 0.0
    probabilities: Optional[np.ndarray] = None
    inference_time: float = 0.0
    model_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction result to dictionary"""
        result = {
            'prediction': self.prediction.tolist() if isinstance(self.prediction, np.ndarray) else self.prediction,
            'confidence': self.confidence,
            'inference_time': self.inference_time,
            'model_id': self.model_id,
            'timestamp': self.timestamp
        }
        
        if self.probabilities is not None:
            result['probabilities'] = self.probabilities.tolist()
        
        return result

class MLModel(ABC):
    """Abstract base class for ML models"""
    
    def __init__(self, model_id: str, model_type: ModelType):
        self.model_id = model_id
        self.model_type = model_type
        self.status = ModelStatus.UNLOADED
        self.metrics = ModelMetrics()
        self.model = None
        self.feature_names: List[str] = []
        self.label_names: List[str] = []
        self.training_history: List[Dict[str, Any]] = []
        self.created_at = time.time()
        self.last_updated = time.time()
    
    @abstractmethod
    def train(self, training_data: TrainingData) -> bool:
        """Train the model with provided data"""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> PredictionResult:
        """Make predictions on input features"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """Save model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """Load model from file"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_id': self.model_id,
            'model_type': self.model_type.value,
            'status': self.status.value,
            'metrics': self.metrics.to_dict(),
            'feature_names': self.feature_names,
            'label_names': self.label_names,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'training_history_length': len(self.training_history)
        }
    
    def update_metrics(self, new_metrics: ModelMetrics):
        """Update model metrics"""
        self.metrics = new_metrics
        self.last_updated = time.time()

class MockMLModel(MLModel):
    """Mock ML model for testing and demonstration"""
    
    def __init__(self, model_id: str, model_type: ModelType = ModelType.CLASSIFICATION):
        super().__init__(model_id, model_type)
        self.weights = None
        self.bias = None
        
    def train(self, training_data: TrainingData) -> bool:
        """Mock training implementation"""
        try:
            self.status = ModelStatus.TRAINING
            logger.info(f"Training mock model {self.model_id}")
            
            # Simulate training time
            training_start = time.time()
            time.sleep(0.1)  # Simulate training
            training_end = time.time()
            
            # Create mock weights and bias
            n_features = training_data.features.shape[1]
            self.weights = np.random.randn(n_features)
            self.bias = np.random.randn()
            
            # Update metrics
            self.metrics.training_time = training_end - training_start
            self.metrics.accuracy = 0.85 + np.random.random() * 0.1  # Mock accuracy
            self.metrics.precision = 0.80 + np.random.random() * 0.15
            self.metrics.recall = 0.75 + np.random.random() * 0.2
            self.metrics.f1_score = 2 * (self.metrics.precision * self.metrics.recall) / (self.metrics.precision + self.metrics.recall)
            
            # Add to training history
            self.training_history.append({
                'timestamp': time.time(),
                'data_size': len(training_data.features),
                'training_time': self.metrics.training_time,
                'accuracy': self.metrics.accuracy
            })
            
            self.status = ModelStatus.READY
            logger.info(f"Mock model {self.model_id} training completed")
            return True
            
        except Exception as e:
            logger.error(f"Error training mock model {self.model_id}: {e}")
            self.status = ModelStatus.ERROR
            return False
    
    def predict(self, features: np.ndarray) -> PredictionResult:
        """Mock prediction implementation"""
        if self.status != ModelStatus.READY or self.weights is None:
            raise RuntimeError(f"Model {self.model_id} is not ready for prediction")
        
        inference_start = time.time()
        
        # Mock prediction calculation
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Simple linear combination for mock prediction
        raw_predictions = np.dot(features, self.weights) + self.bias
        
        if self.model_type == ModelType.CLASSIFICATION:
            # Convert to probabilities using sigmoid
            probabilities = 1 / (1 + np.exp(-raw_predictions))
            predictions = (probabilities > 0.5).astype(int)
            confidence = np.max(probabilities) if len(probabilities) == 1 else np.mean(np.max(probabilities, axis=1))
        else:  # Regression
            predictions = raw_predictions
            probabilities = None
            confidence = 1.0 - np.abs(predictions - np.mean(predictions)) / (np.std(predictions) + 1e-8)
            if len(predictions) == 1:
                confidence = confidence[0]
        
        inference_end = time.time()
        inference_time = inference_end - inference_start
        
        # Update inference time metric
        self.metrics.inference_time = inference_time
        
        return PredictionResult(
            prediction=predictions[0] if len(predictions) == 1 else predictions,
            confidence=float(confidence),
            probabilities=probabilities,
            inference_time=inference_time,
            model_id=self.model_id
        )
    
    def save_model(self, filepath: str) -> bool:
        """Save mock model to file"""
        try:
            model_data = {
                'model_id': self.model_id,
                'model_type': self.model_type.value,
                'weights': self.weights.tolist() if self.weights is not None else None,
                'bias': float(self.bias) if self.bias is not None else None,
                'metrics': self.metrics.to_dict(),
                'feature_names': self.feature_names,
                'label_names': self.label_names,
                'training_history': self.training_history,
                'created_at': self.created_at,
                'last_updated': self.last_updated
            }
            
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Mock model {self.model_id} saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving mock model {self.model_id}: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load mock model from file"""
        try:
            self.status = ModelStatus.LOADING
            
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.model_id = model_data['model_id']
            self.model_type = ModelType(model_data['model_type'])
            self.weights = np.array(model_data['weights']) if model_data['weights'] else None
            self.bias = model_data['bias']
            
            # Restore metrics
            metrics_data = model_data['metrics']
            self.metrics = ModelMetrics(**metrics_data)
            
            self.feature_names = model_data.get('feature_names', [])
            self.label_names = model_data.get('label_names', [])
            self.training_history = model_data.get('training_history', [])
            self.created_at = model_data.get('created_at', time.time())
            self.last_updated = model_data.get('last_updated', time.time())
            
            self.status = ModelStatus.READY
            logger.info(f"Mock model {self.model_id} loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading mock model {self.model_id}: {e}")
            self.status = ModelStatus.ERROR
            return False

class DataCollector:
    """Collects training data during simulation"""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.data_buffer = deque(maxlen=max_samples)
        self.feature_extractors: Dict[str, Callable] = {}
        self.label_extractors: Dict[str, Callable] = {}
        self.collection_active = False
        self.collection_stats = {
            'total_samples': 0,
            'samples_per_second': 0.0,
            'last_collection_time': 0.0
        }
    
    def add_feature_extractor(self, name: str, extractor: Callable[[Dict[str, Any]], float]):
        """Add feature extractor function"""
        self.feature_extractors[name] = extractor
    
    def add_label_extractor(self, name: str, extractor: Callable[[Dict[str, Any]], Union[float, int]]):
        """Add label extractor function"""
        self.label_extractors[name] = extractor
    
    def start_collection(self):
        """Start data collection"""
        self.collection_active = True
        logger.info("Data collection started")
    
    def stop_collection(self):
        """Stop data collection"""
        self.collection_active = False
        logger.info("Data collection stopped")
    
    def collect_sample(self, simulation_state: Dict[str, Any]):
        """Collect a single sample from simulation state"""
        if not self.collection_active:
            return
        
        try:
            # Extract features
            features = {}
            for name, extractor in self.feature_extractors.items():
                features[name] = extractor(simulation_state)
            
            # Extract labels
            labels = {}
            for name, extractor in self.label_extractors.items():
                labels[name] = extractor(simulation_state)
            
            # Create sample
            sample = {
                'features': features,
                'labels': labels,
                'timestamp': time.time(),
                'metadata': simulation_state.get('metadata', {})
            }
            
            self.data_buffer.append(sample)
            self.collection_stats['total_samples'] += 1
            self.collection_stats['last_collection_time'] = time.time()
            
        except Exception as e:
            logger.error(f"Error collecting sample: {e}")
    
    def get_training_data(self, feature_names: List[str], label_names: List[str]) -> Optional[TrainingData]:
        """Get training data for specified features and labels"""
        if len(self.data_buffer) == 0:
            logger.warning("No data collected yet")
            return None
        
        try:
            # Extract features and labels
            features_list = []
            labels_list = []
            
            for sample in self.data_buffer:
                # Extract requested features
                feature_vector = []
                for feature_name in feature_names:
                    if feature_name in sample['features']:
                        feature_vector.append(sample['features'][feature_name])
                    else:
                        logger.warning(f"Feature {feature_name} not found in sample")
                        feature_vector.append(0.0)  # Default value
                
                # Extract requested labels
                label_vector = []
                for label_name in label_names:
                    if label_name in sample['labels']:
                        label_vector.append(sample['labels'][label_name])
                    else:
                        logger.warning(f"Label {label_name} not found in sample")
                        label_vector.append(0.0)  # Default value
                
                features_list.append(feature_vector)
                labels_list.append(label_vector)
            
            features_array = np.array(features_list)
            labels_array = np.array(labels_list)
            
            # Flatten labels if single label
            if labels_array.shape[1] == 1:
                labels_array = labels_array.flatten()
            
            metadata = {
                'feature_names': feature_names,
                'label_names': label_names,
                'collection_time': time.time(),
                'sample_count': len(features_list)
            }
            
            return TrainingData(features_array, labels_array, metadata)
            
        except Exception as e:
            logger.error(f"Error creating training data: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get data collection statistics"""
        return self.collection_stats.copy()
    
    def clear_buffer(self):
        """Clear data buffer"""
        self.data_buffer.clear()
        logger.info("Data buffer cleared")

class ModelEvaluator:
    """Evaluates ML model performance"""
    
    def __init__(self):
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_classification(self, model: MLModel, test_data: TrainingData) -> ModelMetrics:
        """Evaluate classification model"""
        try:
            predictions = []
            inference_times = []
            
            # Make predictions
            for i in range(len(test_data.features)):
                features = test_data.features[i:i+1]
                result = model.predict(features)
                predictions.append(result.prediction)
                inference_times.append(result.inference_time)
            
            predictions = np.array(predictions)
            true_labels = test_data.labels
            
            # Calculate metrics
            accuracy = np.mean(predictions == true_labels)
            
            # Calculate precision, recall, F1 for binary classification
            if len(np.unique(true_labels)) == 2:
                tp = np.sum((predictions == 1) & (true_labels == 1))
                fp = np.sum((predictions == 1) & (true_labels == 0))
                fn = np.sum((predictions == 0) & (true_labels == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                precision = accuracy  # Simplified for multi-class
                recall = accuracy
                f1_score = accuracy
            
            avg_inference_time = np.mean(inference_times)
            
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                inference_time=avg_inference_time
            )
            
            # Record evaluation
            evaluation_record = {
                'model_id': model.model_id,
                'model_type': model.model_type.value,
                'timestamp': time.time(),
                'test_samples': len(test_data.features),
                'metrics': metrics.to_dict()
            }
            self.evaluation_history.append(evaluation_record)
            
            logger.info(f"Classification evaluation completed for {model.model_id}: Accuracy={accuracy:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating classification model {model.model_id}: {e}")
            return ModelMetrics()
    
    def evaluate_regression(self, model: MLModel, test_data: TrainingData) -> ModelMetrics:
        """Evaluate regression model"""
        try:
            predictions = []
            inference_times = []
            
            # Make predictions
            for i in range(len(test_data.features)):
                features = test_data.features[i:i+1]
                result = model.predict(features)
                predictions.append(result.prediction)
                inference_times.append(result.inference_time)
            
            predictions = np.array(predictions)
            true_labels = test_data.labels
            
            # Calculate regression metrics
            mse = np.mean((predictions - true_labels) ** 2)
            mae = np.mean(np.abs(predictions - true_labels))
            avg_inference_time = np.mean(inference_times)
            
            # R-squared (coefficient of determination)
            ss_res = np.sum((true_labels - predictions) ** 2)
            ss_tot = np.sum((true_labels - np.mean(true_labels)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            metrics = ModelMetrics(
                mse=mse,
                mae=mae,
                accuracy=r_squared,  # Use R-squared as accuracy for regression
                inference_time=avg_inference_time
            )
            
            # Record evaluation
            evaluation_record = {
                'model_id': model.model_id,
                'model_type': model.model_type.value,
                'timestamp': time.time(),
                'test_samples': len(test_data.features),
                'metrics': metrics.to_dict()
            }
            self.evaluation_history.append(evaluation_record)
            
            logger.info(f"Regression evaluation completed for {model.model_id}: MSE={mse:.3f}, MAE={mae:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating regression model {model.model_id}: {e}")
            return ModelMetrics()
    
    def evaluate_model(self, model: MLModel, test_data: TrainingData) -> ModelMetrics:
        """Evaluate model based on its type"""
        if model.model_type == ModelType.CLASSIFICATION:
            return self.evaluate_classification(model, test_data)
        elif model.model_type == ModelType.REGRESSION:
            return self.evaluate_regression(model, test_data)
        else:
            logger.warning(f"Evaluation not implemented for model type {model.model_type}")
            return ModelMetrics()
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history"""
        return self.evaluation_history.copy()
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare performance of multiple models"""
        comparison = {}
        
        for model_id in model_ids:
            model_evaluations = [eval_record for eval_record in self.evaluation_history 
                               if eval_record['model_id'] == model_id]
            
            if model_evaluations:
                latest_eval = max(model_evaluations, key=lambda x: x['timestamp'])
                comparison[model_id] = latest_eval['metrics']
        
        return comparison

class MLManager:
    """Main ML integration manager"""
    
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.data_collector = DataCollector()
        self.model_evaluator = ModelEvaluator()
        self.training_queue = deque()
        self.training_thread: Optional[threading.Thread] = None
        self.training_active = False
        
    def register_model(self, model: MLModel):
        """Register a new ML model"""
        self.models[model.model_id] = model
        logger.info(f"Registered model {model.model_id} of type {model.model_type.value}")
    
    def unregister_model(self, model_id: str):
        """Unregister an ML model"""
        if model_id in self.models:
            del self.models[model_id]
            logger.info(f"Unregistered model {model_id}")
    
    def get_model(self, model_id: str) -> Optional[MLModel]:
        """Get model by ID"""
        return self.models.get(model_id)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return [model.get_info() for model in self.models.values()]
    
    def predict(self, model_id: str, features: np.ndarray) -> Optional[PredictionResult]:
        """Make prediction using specified model"""
        model = self.get_model(model_id)
        if model is None:
            logger.error(f"Model {model_id} not found")
            return None
        
        if model.status != ModelStatus.READY:
            logger.error(f"Model {model_id} is not ready (status: {model.status.value})")
            return None
        
        try:
            return model.predict(features)
        except Exception as e:
            logger.error(f"Error making prediction with model {model_id}: {e}")
            return None
    
    def train_model_async(self, model_id: str, training_data: TrainingData):
        """Queue model for asynchronous training"""
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return
        
        training_task = {
            'model_id': model_id,
            'training_data': training_data,
            'timestamp': time.time()
        }
        
        self.training_queue.append(training_task)
        logger.info(f"Queued training task for model {model_id}")
        
        # Start training thread if not already running
        if not self.training_active:
            self._start_training_thread()
    
    def train_model_sync(self, model_id: str, training_data: TrainingData) -> bool:
        """Train model synchronously"""
        model = self.get_model(model_id)
        if model is None:
            logger.error(f"Model {model_id} not found")
            return False
        
        return model.train(training_data)
    
    def evaluate_model(self, model_id: str, test_data: TrainingData) -> Optional[ModelMetrics]:
        """Evaluate model performance"""
        model = self.get_model(model_id)
        if model is None:
            logger.error(f"Model {model_id} not found")
            return None
        
        metrics = self.model_evaluator.evaluate_model(model, test_data)
        model.update_metrics(metrics)
        return metrics
    
    def save_model(self, model_id: str, filepath: str) -> bool:
        """Save model to file"""
        model = self.get_model(model_id)
        if model is None:
            logger.error(f"Model {model_id} not found")
            return False
        
        return model.save_model(filepath)
    
    def load_model(self, model_id: str, filepath: str) -> bool:
        """Load model from file"""
        model = self.get_model(model_id)
        if model is None:
            logger.error(f"Model {model_id} not found")
            return False
        
        return model.load_model(filepath)
    
    def start_data_collection(self):
        """Start data collection"""
        self.data_collector.start_collection()
    
    def stop_data_collection(self):
        """Stop data collection"""
        self.data_collector.stop_collection()
    
    def collect_simulation_data(self, simulation_state: Dict[str, Any]):
        """Collect data from simulation state"""
        self.data_collector.collect_sample(simulation_state)
    
    def get_training_data(self, feature_names: List[str], label_names: List[str]) -> Optional[TrainingData]:
        """Get collected training data"""
        return self.data_collector.get_training_data(feature_names, label_names)
    
    def add_feature_extractor(self, name: str, extractor: Callable[[Dict[str, Any]], float]):
        """Add feature extractor for data collection"""
        self.data_collector.add_feature_extractor(name, extractor)
    
    def add_label_extractor(self, name: str, extractor: Callable[[Dict[str, Any]], Union[float, int]]):
        """Add label extractor for data collection"""
        self.data_collector.add_label_extractor(name, extractor)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get ML system status"""
        return {
            'models': {model_id: model.get_info() for model_id, model in self.models.items()},
            'data_collection': {
                'active': self.data_collector.collection_active,
                'stats': self.data_collector.get_collection_stats(),
                'buffer_size': len(self.data_collector.data_buffer),
                'max_samples': self.data_collector.max_samples
            },
            'training': {
                'active': self.training_active,
                'queue_size': len(self.training_queue)
            },
            'evaluation_history_size': len(self.model_evaluator.evaluation_history)
        }
    
    def _start_training_thread(self):
        """Start background training thread"""
        if self.training_active:
            return
        
        self.training_active = True
        self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.training_thread.start()
        logger.info("Training thread started")
    
    def _training_worker(self):
        """Background training worker"""
        while self.training_active:
            try:
                if len(self.training_queue) > 0:
                    task = self.training_queue.popleft()
                    model_id = task['model_id']
                    training_data = task['training_data']
                    
                    logger.info(f"Starting background training for model {model_id}")
                    model = self.get_model(model_id)
                    if model:
                        success = model.train(training_data)
                        if success:
                            logger.info(f"Background training completed for model {model_id}")
                        else:
                            logger.error(f"Background training failed for model {model_id}")
                else:
                    time.sleep(0.1)  # Wait for new tasks
                    
            except Exception as e:
                logger.error(f"Error in training worker: {e}")
                time.sleep(1.0)
    
    def shutdown(self):
        """Shutdown ML manager"""
        self.training_active = False
        self.data_collector.stop_collection()
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)
        
        logger.info("ML Manager shutdown completed")

# Utility functions for common feature extractors
def create_vehicle_feature_extractors() -> Dict[str, Callable]:
    """Create common vehicle feature extractors"""
    extractors = {
        'speed': lambda state: state.get('vehicle', {}).get('speed', 0.0),
        'acceleration': lambda state: state.get('vehicle', {}).get('acceleration', 0.0),
        'steering_angle': lambda state: state.get('vehicle', {}).get('steering_angle', 0.0),
        'throttle': lambda state: state.get('vehicle', {}).get('throttle', 0.0),
        'brake': lambda state: state.get('vehicle', {}).get('brake', 0.0),
        'distance_to_obstacle': lambda state: min(state.get('sensors', {}).get('lidar', {}).get('distances', [100.0])),
        'lane_deviation': lambda state: state.get('navigation', {}).get('lane_deviation', 0.0),
        'time_to_collision': lambda state: state.get('safety', {}).get('time_to_collision', 10.0)
    }
    return extractors

def create_vehicle_label_extractors() -> Dict[str, Callable]:
    """Create common vehicle label extractors"""
    extractors = {
        'should_brake': lambda state: 1 if state.get('actions', {}).get('brake', 0.0) > 0.5 else 0,
        'steering_command': lambda state: state.get('actions', {}).get('steering', 0.0),
        'throttle_command': lambda state: state.get('actions', {}).get('throttle', 0.0),
        'collision_risk': lambda state: 1 if state.get('safety', {}).get('collision_imminent', False) else 0
    }
    return extractors