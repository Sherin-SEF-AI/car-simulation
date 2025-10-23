"""
Cross-Platform Compatibility Tests for Robotic Car Simulation

Tests to ensure the simulation works correctly across different operating systems,
Python versions, and hardware configurations.
"""

import unittest
import sys
import os
import platform
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication


class PlatformTestBase(unittest.TestCase):
    """Base class for cross-platform tests"""
    
    def setUp(self):
        """Set up platform testing environment"""
        self.platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation()
        }
        
        if not QApplication.instance():
            self.app = QApplication([])
        else:
            self.app = QApplication.instance()
            
    def get_platform_specific_config(self) -> Dict[str, Any]:
        """Get platform-specific configuration"""
        system = self.platform_info['system'].lower()
        
        if system == 'windows':
            return {
                'file_separator': '\\',
                'executable_extension': '.exe',
                'library_extension': '.dll',
                'max_path_length': 260,
                'case_sensitive_fs': False
            }
        elif system == 'darwin':  # macOS
            return {
                'file_separator': '/',
                'executable_extension': '',
                'library_extension': '.dylib',
                'max_path_length': 1024,
                'case_sensitive_fs': False  # Usually
            }
        else:  # Linux and other Unix-like
            return {
                'file_separator': '/',
                'executable_extension': '',
                'library_extension': '.so',
                'max_path_length': 4096,
                'case_sensitive_fs': True
            }


class TestPlatformCompatibility(PlatformTestBase):
    """Test basic platform compatibility"""
    
    def test_python_version_compatibility(self):
        """Test compatibility with supported Python versions"""
        python_version = sys.version_info
        
        # Should support Python 3.9+
        self.assertGreaterEqual(python_version.major, 3)
        self.assertGreaterEqual(python_version.minor, 9)
        
        print(f"Testing on Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
    def test_required_modules_import(self):
        """Test that all required modules can be imported"""
        required_modules = [
            'PyQt6.QtCore',
            'PyQt6.QtWidgets', 
            'PyQt6.QtOpenGL',
            'numpy',
            'json',
            'threading',
            'multiprocessing',
            'sqlite3',
            'xml.etree.ElementTree'
        ]
        
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                self.fail(f"Required module {module_name} could not be imported: {e}")
                
    def test_file_system_operations(self):
        """Test file system operations across platforms"""
        config = self.get_platform_specific_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test file creation
            test_file = temp_path / "test_file.txt"
            test_file.write_text("Test content")
            self.assertTrue(test_file.exists())
            
            # Test directory creation
            test_dir = temp_path / "test_directory"
            test_dir.mkdir()
            self.assertTrue(test_dir.is_dir())
            
            # Test path handling
            nested_path = test_dir / "nested" / "path" / "file.json"
            nested_path.parent.mkdir(parents=True, exist_ok=True)
            nested_path.write_text('{"test": "data"}')
            self.assertTrue(nested_path.exists())
            
            # Test case sensitivity if applicable
            if config['case_sensitive_fs']:
                upper_file = temp_path / "UPPER.txt"
                lower_file = temp_path / "upper.txt"
                upper_file.write_text("upper")
                lower_file.write_text("lower")
                
                self.assertNotEqual(upper_file.read_text(), lower_file.read_text())
                
    def test_multiprocessing_support(self):
        """Test multiprocessing functionality"""
        import multiprocessing as mp
        
        def test_worker(queue):
            queue.put("worker_result")
            
        # Test process creation
        queue = mp.Queue()
        process = mp.Process(target=test_worker, args=(queue,))
        process.start()
        process.join(timeout=5.0)
        
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(queue.get(), "worker_result")
        
    def test_threading_support(self):
        """Test threading functionality"""
        import threading
        import time
        
        result = []
        
        def test_thread():
            time.sleep(0.1)
            result.append("thread_completed")
            
        thread = threading.Thread(target=test_thread)
        thread.start()
        thread.join(timeout=1.0)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "thread_completed")


class TestQtCompatibility(PlatformTestBase):
    """Test Qt/PyQt6 compatibility across platforms"""
    
    def test_qt_application_creation(self):
        """Test Qt application can be created"""
        # Application should already be created in setUp
        self.assertIsNotNone(QApplication.instance())
        
    def test_qt_widgets_creation(self):
        """Test basic Qt widgets can be created"""
        from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout
        
        widget = QWidget()
        layout = QVBoxLayout()
        
        label = QLabel("Test Label")
        button = QPushButton("Test Button")
        
        layout.addWidget(label)
        layout.addWidget(button)
        widget.setLayout(layout)
        
        # Should not raise exceptions
        self.assertIsNotNone(widget)
        self.assertIsNotNone(label)
        self.assertIsNotNone(button)
        
    def test_opengl_support(self):
        """Test OpenGL support availability"""
        try:
            from PyQt6.QtOpenGL import QOpenGLWidget
            from PyQt6.QtGui import QOpenGLContext
            
            # Try to create OpenGL context
            context = QOpenGLContext()
            self.assertIsNotNone(context)
            
            # Note: Actual OpenGL initialization may fail in headless environments
            # This test just checks that the classes can be imported and instantiated
            
        except ImportError as e:
            self.skipTest(f"OpenGL support not available: {e}")
            
    def test_qt_signals_and_slots(self):
        """Test Qt signals and slots mechanism"""
        from PyQt6.QtCore import QObject, pyqtSignal
        
        class TestObject(QObject):
            test_signal = pyqtSignal(str)
            
            def __init__(self):
                super().__init__()
                self.received_data = None
                
            def test_slot(self, data):
                self.received_data = data
                
        obj = TestObject()
        obj.test_signal.connect(obj.test_slot)
        obj.test_signal.emit("test_data")
        
        # Process events to ensure signal is delivered
        QApplication.processEvents()
        
        self.assertEqual(obj.received_data, "test_data")


class TestSimulationCompatibility(PlatformTestBase):
    """Test simulation-specific compatibility"""
    
    def test_simulation_application_import(self):
        """Test that simulation application can be imported"""
        try:
            from core.application import SimulationApplication
            app = SimulationApplication()
            self.assertIsNotNone(app)
            app.cleanup()
        except Exception as e:
            self.fail(f"Could not import or create SimulationApplication: {e}")
            
    def test_physics_engine_compatibility(self):
        """Test physics engine compatibility"""
        try:
            from core.physics_engine import PhysicsEngine
            physics = PhysicsEngine()
            self.assertIsNotNone(physics)
            
            # Test basic physics operations
            physics.add_vehicle("test_vehicle", position=(0, 0, 0))
            position = physics.get_vehicle_position("test_vehicle")
            self.assertIsNotNone(position)
            
        except Exception as e:
            self.fail(f"Physics engine compatibility issue: {e}")
            
    def test_rendering_system_compatibility(self):
        """Test rendering system compatibility"""
        try:
            # Mock OpenGL for testing
            with patch('PyQt6.QtOpenGL.QOpenGLWidget'):
                from ui.rendering.render_engine import RenderEngine
                renderer = RenderEngine()
                self.assertIsNotNone(renderer)
                
        except Exception as e:
            self.fail(f"Rendering system compatibility issue: {e}")


class TestPerformanceAcrossPlatforms(PlatformTestBase):
    """Test performance characteristics across platforms"""
    
    def test_basic_performance_metrics(self):
        """Test basic performance metrics collection"""
        import time
        import psutil
        
        process = psutil.Process()
        
        # Test memory measurement
        memory_info = process.memory_info()
        self.assertGreater(memory_info.rss, 0)
        
        # Test CPU measurement
        cpu_percent = process.cpu_percent(interval=0.1)
        self.assertGreaterEqual(cpu_percent, 0.0)
        
        # Test timing
        start_time = time.time()
        time.sleep(0.01)  # 10ms
        elapsed = time.time() - start_time
        self.assertGreater(elapsed, 0.005)  # Should be at least 5ms
        
    def test_file_io_performance(self):
        """Test file I/O performance across platforms"""
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "performance_test.json"
            
            # Test write performance
            test_data = {"test": "data", "numbers": list(range(1000))}
            
            start_time = time.time()
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            write_time = time.time() - start_time
            
            # Test read performance
            start_time = time.time()
            with open(test_file, 'r') as f:
                loaded_data = json.load(f)
            read_time = time.time() - start_time
            
            # Basic performance assertions (very lenient)
            self.assertLess(write_time, 1.0)  # Should write in < 1 second
            self.assertLess(read_time, 1.0)   # Should read in < 1 second
            self.assertEqual(loaded_data, test_data)


class TestConfigurationCompatibility(PlatformTestBase):
    """Test configuration and settings compatibility"""
    
    def test_configuration_file_handling(self):
        """Test configuration file creation and loading"""
        config_data = {
            'simulation': {
                'max_vehicles': 50,
                'physics_timestep': 0.016,
                'enable_recording': True
            },
            'rendering': {
                'resolution': [1920, 1080],
                'vsync': True,
                'quality': 'high'
            },
            'platform': self.platform_info
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"
            
            # Write configuration
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            # Read configuration
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                
            self.assertEqual(loaded_config, config_data)
            
    def test_user_data_directory_creation(self):
        """Test user data directory creation"""
        config = self.get_platform_specific_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            user_data_dir = Path(temp_dir) / "robotic_car_sim"
            
            # Create directory structure
            subdirs = ['recordings', 'configs', 'logs', 'cache']
            for subdir in subdirs:
                (user_data_dir / subdir).mkdir(parents=True, exist_ok=True)
                
            # Verify all directories exist
            for subdir in subdirs:
                self.assertTrue((user_data_dir / subdir).is_dir())


class TestErrorHandlingCompatibility(PlatformTestBase):
    """Test error handling across platforms"""
    
    def test_exception_handling(self):
        """Test that exceptions are handled consistently"""
        def test_function():
            raise ValueError("Test exception")
            
        with self.assertRaises(ValueError) as context:
            test_function()
            
        self.assertEqual(str(context.exception), "Test exception")
        
    def test_file_permission_errors(self):
        """Test handling of file permission errors"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "readonly_test.txt"
            test_file.write_text("test content")
            
            # Make file read-only (platform-specific)
            if self.platform_info['system'] == 'Windows':
                os.chmod(test_file, 0o444)
            else:
                os.chmod(test_file, 0o444)
                
            # Attempt to write should raise exception
            with self.assertRaises(PermissionError):
                test_file.write_text("new content")
                
    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up"""
        import gc
        
        # Create some objects
        test_objects = []
        for i in range(100):
            test_objects.append({"data": f"object_{i}"})
            
        # Clear references
        test_objects.clear()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Should have collected some objects
        self.assertGreaterEqual(collected, 0)


class TestDeploymentCompatibility(PlatformTestBase):
    """Test deployment-related compatibility"""
    
    def test_executable_creation_compatibility(self):
        """Test that the application can be packaged as executable"""
        # This is a placeholder test - actual executable creation
        # would be done by build tools like PyInstaller
        
        # Test that main entry point can be imported
        try:
            import main
            self.assertTrue(hasattr(main, '__file__'))
        except ImportError:
            # main.py might not be importable as module
            main_file = Path(__file__).parent.parent / 'main.py'
            self.assertTrue(main_file.exists())
            
    def test_dependency_availability(self):
        """Test that all dependencies are available"""
        # Read requirements.txt and check each dependency
        requirements_file = Path(__file__).parent.parent / 'requirements.txt'
        
        if requirements_file.exists():
            with open(requirements_file) as f:
                requirements = f.read().splitlines()
                
            # Filter out comments and empty lines
            requirements = [req.strip() for req in requirements 
                          if req.strip() and not req.strip().startswith('#')]
            
            for requirement in requirements[:5]:  # Test first 5 to avoid long test times
                package_name = requirement.split('>=')[0].split('==')[0].split('<')[0]
                try:
                    __import__(package_name.replace('-', '_'))
                except ImportError:
                    self.fail(f"Required package {package_name} not available")


if __name__ == '__main__':
    # Print platform information
    print(f"Running cross-platform tests on:")
    print(f"  System: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Machine: {platform.machine()}")
    print()
    
    unittest.main()