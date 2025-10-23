"""
Unit tests for ML integration framework
"""

import unittest
import numpy as np
import tempfile
import os
import time
from unittest.mock import Mock, patch

from src.core.ml_integration import (
    ModelType, ModelStatus, ModelMetrics, TrainingData, PredictionResult,
    MLModel, MockMLModel, DataCollector, ModelEvaluator, MLManager,
    create_vehicle_feature_extractors, create_vehicle_label_extractors
)

class TestModelMetrics(unittest.TestCase):
    """Test ModelMetrics class"""
    
    def test_metrics_creation(self):
        """Test metrics creation with default values"""
        metrics = ModelMetrics()
        self.assertEqual(metrics.accuracy, 0.0)
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.f1_score, 0.0)
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary"""
        metrics = ModelMetrics(accuracy=0.85, precision=0.80, recall=0.90)
        metrics_dict = metrics.to_dict()
        
        self.assertEqual(metrics_dict['accuracy'], 0.85)
        self.assertEqual(metrics_dict['precision'], 0.80)
        self.assertEqual(metrics_dict['recall'], 0.90)
        self.assertIn('f1_score', metrics_dict)

class TestTrainingData(unittest.TestCase):
    """Test TrainingData class"""
    
    def test_valid_training_data(self):
        """Test creation of valid training data"""
        features = np.array([[1, 2], [3, 4], [5, 6]])
        labels = np.array([0, 1, 0])
        
        training_data = TrainingData(features, labels)
        
        self.assertEqual(len(training_data.features), 3)
        self.assertEqual(len(training_data.labels), 3)
        self.assertIsInstance(training_data.metadata, dict)
    
    def test_mismatched_lengths(self):
        """Test error when features and labels have different lengths"""
        features = np.array([[1, 2], [3, 4]])
        labels = np.array([0, 1, 0])  # Different length
        
        with self.assertRaises(ValueError):
            TrainingData(features, labels)
    
    def test_empty_data(self):
        """Test error when data is empty"""
        features = np.array([])
        labels = np.array([])
        
        with self.assertRaises(ValueError):
            TrainingData(features, labels)

class TestPredictionResult(unittest.TestCase):
    """Test PredictionResult class"""
    
    def test_prediction_result_creation(self):
        """Test prediction result creation"""
        result = PredictionResult(
            prediction=1,
            confidence=0.85,
            inference_time=0.001,
            model_id="test_model"
        )
        
        self.assertEqual(result.prediction, 1)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.model_id, "test_model")
    
    def test_prediction_result_to_dict(self):
        """Test prediction result conversion to dictionary"""
        probabilities = np.array([0.2, 0.8])
        result = PredictionResult(
            prediction=1,
            confidence=0.8,
            probabilities=probabilities,
            model_id="test_model"
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict['prediction'], 1)
        self.assertEqual(result_dict['confidence'], 0.8)
        self.assertEqual(result_dict['model_id'], "test_model")
        self.assertEqual(result_dict['probabilities'], [0.2, 0.8])

class TestMockMLModel(unittest.TestCase):
    """Test MockMLModel implementation"""
    
    def setUp(self):
        self.model = MockMLModel("test_model", ModelType.CLASSIFICATION)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.model_id, "test_model")
        self.assertEqual(self.model.model_type, ModelType.CLASSIFICATION)
        self.assertEqual(self.model.status, ModelStatus.UNLOADED)
    
    def test_model_training(self):
        """Test model training"""
        features = np.random.randn(100, 5)
        labels = np.random.randint(0, 2, 100)
        training_data = TrainingData(features, labels)
        
        success = self.model.train(training_data)
        
        self.assertTrue(success)
        self.assertEqual(self.model.status, ModelStatus.READY)
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)
        self.assertGreater(self.model.metrics.accuracy, 0)
    
    def test_model_prediction(self):
        """Test model prediction"""
        # Train model first
        features = np.random.randn(50, 3)
        labels = np.random.randint(0, 2, 50)
        training_data = TrainingData(features, labels)
        self.model.train(training_data)
        
        # Make prediction
        test_features = np.random.randn(1, 3)
        result = self.model.predict(test_features)
        
        self.assertIsInstance(result, PredictionResult)
        self.assertIn(result.prediction, [0, 1])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        self.assertEqual(result.model_id, "test_model")
    
    def test_prediction_without_training(self):
        """Test prediction fails when model not trained"""
        test_features = np.random.randn(1, 3)
        
        with self.assertRaises(RuntimeError):
            self.model.predict(test_features)
    
    def test_model_save_load(self):
        """Test model save and load functionality"""
        # Train model
        features = np.random.randn(30, 4)
        labels = np.random.randint(0, 2, 30)
        training_data = TrainingData(features, labels)
        self.model.train(training_data)
        
        # Save model
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filepath = f.name
        
        try:
            success = self.model.save_model(temp_filepath)
            self.assertTrue(success)
            
            # Create new model and load
            new_model = MockMLModel("loaded_model", ModelType.CLASSIFICATION)
            success = new_model.load_model(temp_filepath)
            self.assertTrue(success)
            
            # Verify loaded model
            self.assertEqual(new_model.status, ModelStatus.READY)
            self.assertIsNotNone(new_model.weights)
            self.assertIsNotNone(new_model.bias)
            
            # Test prediction with loaded model
            test_features = np.random.randn(1, 4)
            result = new_model.predict(test_features)
            self.assertIsInstance(result, PredictionResult)
            
        finally:
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
    
    def test_model_info(self):
        """Test model info retrieval"""
        info = self.model.get_info()
        
        self.assertEqual(info['model_id'], "test_model")
        self.assertEqual(info['model_type'], ModelType.CLASSIFICATION.value)
        self.assertEqual(info['status'], ModelStatus.UNLOADED.value)
        self.assertIn('metrics', info)
        self.assertIn('created_at', info)

class TestDataCollector(unittest.TestCase):
    """Test DataCollector class"""
    
    def setUp(self):
        self.collector = DataCollector(max_samples=100)
    
    def test_feature_extractor_addition(self):
        """Test adding feature extractors"""
        def speed_extractor(state):
            return state.get('speed', 0.0)
        
        self.collector.add_feature_extractor('speed', speed_extractor)
        self.assertIn('speed', self.collector.feature_extractors)
    
    def test_label_extractor_addition(self):
        """Test adding label extractors"""
        def brake_extractor(state):
            return 1 if state.get('brake', 0.0) > 0.5 else 0
        
        self.collector.add_label_extractor('should_brake', brake_extractor)
        self.assertIn('should_brake', self.collector.label_extractors)
    
    def test_data_collection(self):
        """Test data collection process"""
        # Add extractors
        self.collector.add_feature_extractor('speed', lambda state: state.get('speed', 0.0))
        self.collector.add_feature_extractor('throttle', lambda state: state.get('throttle', 0.0))
        self.collector.add_label_extractor('action', lambda state: state.get('action', 0))
        
        # Start collection
        self.collector.start_collection()
        self.assertTrue(self.collector.collection_active)
        
        # Collect samples
        for i in range(10):
            simulation_state = {
                'speed': i * 10.0,
                'throttle': i * 0.1,
                'action': i % 2
            }
            self.collector.collect_sample(simulation_state)
        
        # Check collection
        self.assertEqual(len(self.collector.data_buffer), 10)
        self.assertEqual(self.collector.collection_stats['total_samples'], 10)
    
    def test_training_data_generation(self):
        """Test training data generation from collected samples"""
        # Setup extractors
        self.collector.add_feature_extractor('speed', lambda state: state.get('speed', 0.0))
        self.collector.add_feature_extractor('throttle', lambda state: state.get('throttle', 0.0))
        self.collector.add_label_extractor('action', lambda state: state.get('action', 0))
        
        # Collect data
        self.collector.start_collection()
        for i in range(20):
            simulation_state = {
                'speed': i * 5.0,
                'throttle': i * 0.05,
                'action': i % 2
            }
            self.collector.collect_sample(simulation_state)
        
        # Generate training data
        training_data = self.collector.get_training_data(['speed', 'throttle'], ['action'])
        
        self.assertIsNotNone(training_data)
        self.assertEqual(training_data.features.shape, (20, 2))
        self.assertEqual(training_data.labels.shape, (20,))
        self.assertIn('feature_names', training_data.metadata)
    
    def test_collection_without_active(self):
        """Test that collection doesn't work when not active"""
        self.collector.add_feature_extractor('speed', lambda state: state.get('speed', 0.0))
        
        # Don't start collection
        simulation_state = {'speed': 10.0}
        self.collector.collect_sample(simulation_state)
        
        # Should not collect anything
        self.assertEqual(len(self.collector.data_buffer), 0)

class TestModelEvaluator(unittest.TestCase):
    """Test ModelEvaluator class"""
    
    def setUp(self):
        self.evaluator = ModelEvaluator()
        self.model = MockMLModel("test_model", ModelType.CLASSIFICATION)
        
        # Train the model
        features = np.random.randn(100, 3)
        labels = np.random.randint(0, 2, 100)
        training_data = TrainingData(features, labels)
        self.model.train(training_data)
    
    def test_classification_evaluation(self):
        """Test classification model evaluation"""
        # Create test data
        test_features = np.random.randn(50, 3)
        test_labels = np.random.randint(0, 2, 50)
        test_data = TrainingData(test_features, test_labels)
        
        # Evaluate model
        metrics = self.evaluator.evaluate_classification(self.model, test_data)
        
        self.assertIsInstance(metrics, ModelMetrics)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertGreaterEqual(metrics.precision, 0.0)
        self.assertGreaterEqual(metrics.recall, 0.0)
        self.assertGreater(metrics.inference_time, 0.0)
    
    def test_regression_evaluation(self):
        """Test regression model evaluation"""
        # Create regression model
        reg_model = MockMLModel("reg_model", ModelType.REGRESSION)
        features = np.random.randn(100, 3)
        labels = np.random.randn(100)  # Continuous labels
        training_data = TrainingData(features, labels)
        reg_model.train(training_data)
        
        # Create test data
        test_features = np.random.randn(50, 3)
        test_labels = np.random.randn(50)
        test_data = TrainingData(test_features, test_labels)
        
        # Evaluate model
        metrics = self.evaluator.evaluate_regression(reg_model, test_data)
        
        self.assertIsInstance(metrics, ModelMetrics)
        self.assertGreaterEqual(metrics.mse, 0.0)
        self.assertGreaterEqual(metrics.mae, 0.0)
        self.assertGreater(metrics.inference_time, 0.0)
    
    def test_evaluation_history(self):
        """Test evaluation history tracking"""
        # Create test data
        test_features = np.random.randn(30, 3)
        test_labels = np.random.randint(0, 2, 30)
        test_data = TrainingData(test_features, test_labels)
        
        # Evaluate model
        self.evaluator.evaluate_model(self.model, test_data)
        
        # Check history
        history = self.evaluator.get_evaluation_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['model_id'], "test_model")
        self.assertIn('metrics', history[0])
    
    def test_model_comparison(self):
        """Test model comparison functionality"""
        # Create second model
        model2 = MockMLModel("test_model_2", ModelType.CLASSIFICATION)
        features = np.random.randn(80, 3)
        labels = np.random.randint(0, 2, 80)
        training_data = TrainingData(features, labels)
        model2.train(training_data)
        
        # Evaluate both models
        test_features = np.random.randn(40, 3)
        test_labels = np.random.randint(0, 2, 40)
        test_data = TrainingData(test_features, test_labels)
        
        self.evaluator.evaluate_model(self.model, test_data)
        self.evaluator.evaluate_model(model2, test_data)
        
        # Compare models
        comparison = self.evaluator.compare_models(["test_model", "test_model_2"])
        
        self.assertIn("test_model", comparison)
        self.assertIn("test_model_2", comparison)
        self.assertIn('accuracy', comparison["test_model"])

class TestMLManager(unittest.TestCase):
    """Test MLManager class"""
    
    def setUp(self):
        self.manager = MLManager()
        self.model = MockMLModel("test_model", ModelType.CLASSIFICATION)
    
    def tearDown(self):
        self.manager.shutdown()
    
    def test_model_registration(self):
        """Test model registration and retrieval"""
        self.manager.register_model(self.model)
        
        retrieved_model = self.manager.get_model("test_model")
        self.assertEqual(retrieved_model, self.model)
        
        models_list = self.manager.list_models()
        self.assertEqual(len(models_list), 1)
        self.assertEqual(models_list[0]['model_id'], "test_model")
    
    def test_model_unregistration(self):
        """Test model unregistration"""
        self.manager.register_model(self.model)
        self.assertIsNotNone(self.manager.get_model("test_model"))
        
        self.manager.unregister_model("test_model")
        self.assertIsNone(self.manager.get_model("test_model"))
    
    def test_synchronous_training(self):
        """Test synchronous model training"""
        self.manager.register_model(self.model)
        
        features = np.random.randn(50, 4)
        labels = np.random.randint(0, 2, 50)
        training_data = TrainingData(features, labels)
        
        success = self.manager.train_model_sync("test_model", training_data)
        self.assertTrue(success)
        self.assertEqual(self.model.status, ModelStatus.READY)
    
    def test_prediction(self):
        """Test model prediction through manager"""
        self.manager.register_model(self.model)
        
        # Train model first
        features = np.random.randn(50, 4)
        labels = np.random.randint(0, 2, 50)
        training_data = TrainingData(features, labels)
        self.manager.train_model_sync("test_model", training_data)
        
        # Make prediction
        test_features = np.random.randn(1, 4)
        result = self.manager.predict("test_model", test_features)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.model_id, "test_model")
    
    def test_prediction_with_unready_model(self):
        """Test prediction fails with unready model"""
        self.manager.register_model(self.model)
        
        test_features = np.random.randn(1, 4)
        result = self.manager.predict("test_model", test_features)
        
        self.assertIsNone(result)
    
    def test_model_evaluation(self):
        """Test model evaluation through manager"""
        self.manager.register_model(self.model)
        
        # Train model
        features = np.random.randn(60, 3)
        labels = np.random.randint(0, 2, 60)
        training_data = TrainingData(features, labels)
        self.manager.train_model_sync("test_model", training_data)
        
        # Evaluate model
        test_features = np.random.randn(30, 3)
        test_labels = np.random.randint(0, 2, 30)
        test_data = TrainingData(test_features, test_labels)
        
        metrics = self.manager.evaluate_model("test_model", test_data)
        
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, ModelMetrics)
    
    def test_data_collection_integration(self):
        """Test data collection integration"""
        # Add feature and label extractors
        self.manager.add_feature_extractor('speed', lambda state: state.get('speed', 0.0))
        self.manager.add_label_extractor('action', lambda state: state.get('action', 0))
        
        # Start collection
        self.manager.start_data_collection()
        
        # Collect some data
        for i in range(15):
            simulation_state = {
                'speed': i * 2.0,
                'action': i % 2
            }
            self.manager.collect_simulation_data(simulation_state)
        
        # Get training data
        training_data = self.manager.get_training_data(['speed'], ['action'])
        
        self.assertIsNotNone(training_data)
        self.assertEqual(training_data.features.shape, (15, 1))
        self.assertEqual(training_data.labels.shape, (15,))
    
    def test_system_status(self):
        """Test system status reporting"""
        self.manager.register_model(self.model)
        
        status = self.manager.get_system_status()
        
        self.assertIn('models', status)
        self.assertIn('data_collection', status)
        self.assertIn('training', status)
        self.assertIn('evaluation_history_size', status)
        
        self.assertIn('test_model', status['models'])

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_vehicle_feature_extractors(self):
        """Test vehicle feature extractors creation"""
        extractors = create_vehicle_feature_extractors()
        
        self.assertIn('speed', extractors)
        self.assertIn('acceleration', extractors)
        self.assertIn('steering_angle', extractors)
        
        # Test extractor functionality
        test_state = {
            'vehicle': {'speed': 25.0, 'acceleration': 2.0},
            'sensors': {'lidar': {'distances': [5.0, 10.0, 15.0]}}
        }
        
        speed = extractors['speed'](test_state)
        self.assertEqual(speed, 25.0)
        
        min_distance = extractors['distance_to_obstacle'](test_state)
        self.assertEqual(min_distance, 5.0)
    
    def test_vehicle_label_extractors(self):
        """Test vehicle label extractors creation"""
        extractors = create_vehicle_label_extractors()
        
        self.assertIn('should_brake', extractors)
        self.assertIn('steering_command', extractors)
        self.assertIn('throttle_command', extractors)
        
        # Test extractor functionality
        test_state = {
            'actions': {'brake': 0.8, 'steering': 0.2, 'throttle': 0.3},
            'safety': {'collision_imminent': True}
        }
        
        should_brake = extractors['should_brake'](test_state)
        self.assertEqual(should_brake, 1)
        
        collision_risk = extractors['collision_risk'](test_state)
        self.assertEqual(collision_risk, 1)
        
        steering = extractors['steering_command'](test_state)
        self.assertEqual(steering, 0.2)

if __name__ == '__main__':
    unittest.main()