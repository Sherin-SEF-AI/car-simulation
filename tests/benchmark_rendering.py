"""
Performance benchmarks for the 3D rendering system
"""

import time
import numpy as np
import sys
from unittest.mock import MagicMock
from PyQt6.QtWidgets import QApplication

# Mock OpenGL for benchmarking
sys.modules['OpenGL'] = MagicMock()
sys.modules['OpenGL.GL'] = MagicMock()

from src.ui.rendering.scene_manager import SceneManager, RenderableObject, Material, Mesh
from src.ui.rendering.lighting_system import LightingSystem, Light, LightType
from src.ui.rendering.camera_manager import CameraManager, CameraMode


class RenderingBenchmark:
    """Benchmark suite for rendering components"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_scene_manager_object_creation(self, num_objects=1000):
        """Benchmark scene manager object creation and management"""
        print(f"Benchmarking scene manager with {num_objects} objects...")
        
        scene_manager = SceneManager()
        mesh = scene_manager.get_mesh("cube")
        material = scene_manager.get_material("default")
        
        # Benchmark object creation
        start_time = time.time()
        
        objects = []
        for i in range(num_objects):
            obj = RenderableObject(
                name=f"object_{i}",
                mesh=mesh,
                material=material,
                transform=self._random_transform()
            )
            objects.append(obj)
        
        creation_time = time.time() - start_time
        
        # Benchmark object addition to scene
        start_time = time.time()
        
        for obj in objects:
            # Mock the buffer setup to avoid OpenGL calls
            with MagicMock():
                scene_manager.add_object(obj)
        
        addition_time = time.time() - start_time
        
        # Benchmark visible object filtering
        start_time = time.time()
        
        for _ in range(100):  # Run multiple times for average
            visible_objects = scene_manager.get_visible_objects()
        
        filtering_time = (time.time() - start_time) / 100
        
        self.results['scene_manager'] = {
            'object_creation_time': creation_time,
            'objects_per_second_creation': num_objects / creation_time,
            'object_addition_time': addition_time,
            'objects_per_second_addition': num_objects / addition_time,
            'visible_filtering_time': filtering_time * 1000,  # Convert to ms
            'total_objects': len(scene_manager.objects)
        }
        
        print(f"  Object creation: {creation_time:.4f}s ({num_objects/creation_time:.0f} objects/s)")
        print(f"  Object addition: {addition_time:.4f}s ({num_objects/addition_time:.0f} objects/s)")
        print(f"  Visible filtering: {filtering_time*1000:.2f}ms per call")
        
        scene_manager.cleanup()
    
    def benchmark_lighting_system(self, num_lights=50):
        """Benchmark lighting system performance"""
        print(f"Benchmarking lighting system with {num_lights} lights...")
        
        lighting_system = LightingSystem()
        
        # Benchmark light creation and addition
        start_time = time.time()
        
        for i in range(num_lights):
            light = Light(
                light_type=LightType.POINT,
                position=np.random.rand(3) * 100,
                direction=np.random.rand(3) - 0.5,
                color=np.random.rand(3),
                intensity=np.random.rand()
            )
            lighting_system.add_light(light)
        
        light_creation_time = time.time() - start_time
        
        # Benchmark uniform generation
        start_time = time.time()
        
        for _ in range(1000):  # Run multiple times for average
            uniforms = lighting_system.get_lighting_uniforms()
        
        uniform_generation_time = (time.time() - start_time) / 1000
        
        # Benchmark time of day updates
        start_time = time.time()
        
        for hour in range(24):
            lighting_system.set_time_of_day(hour)
        
        time_of_day_update_time = time.time() - start_time
        
        self.results['lighting_system'] = {
            'light_creation_time': light_creation_time,
            'lights_per_second': num_lights / light_creation_time,
            'uniform_generation_time': uniform_generation_time * 1000,  # Convert to ms
            'time_of_day_update_time': time_of_day_update_time * 1000,  # Convert to ms
            'total_lights': len(lighting_system.lights)
        }
        
        print(f"  Light creation: {light_creation_time:.4f}s ({num_lights/light_creation_time:.0f} lights/s)")
        print(f"  Uniform generation: {uniform_generation_time*1000:.2f}ms per call")
        print(f"  Time of day updates: {time_of_day_update_time*1000:.2f}ms for 24 hours")
        
        lighting_system.cleanup()
    
    def benchmark_camera_manager(self, num_updates=10000):
        """Benchmark camera manager performance"""
        print(f"Benchmarking camera manager with {num_updates} updates...")
        
        camera_manager = CameraManager()
        
        # Setup test vehicle data
        vehicle_positions = {
            "test_vehicle": {
                "position": [0.0, 0.0, 0.0],
                "rotation": 0.0
            }
        }
        
        # Benchmark camera updates in different modes
        modes = [CameraMode.FIRST_PERSON, CameraMode.THIRD_PERSON, CameraMode.TOP_DOWN, CameraMode.FREE_ROAM]
        mode_times = {}
        
        for mode in modes:
            camera_manager.set_mode(mode, "test_vehicle")
            
            start_time = time.time()
            
            for i in range(num_updates):
                # Simulate vehicle movement
                vehicle_positions["test_vehicle"]["position"] = [
                    np.sin(i * 0.01) * 10,
                    0.0,
                    np.cos(i * 0.01) * 10
                ]
                vehicle_positions["test_vehicle"]["rotation"] = i * 0.1
                
                camera_manager.update(0.016, vehicle_positions)  # 60 FPS delta time
            
            mode_time = time.time() - start_time
            mode_times[mode.value] = mode_time
            
            print(f"  {mode.value}: {mode_time:.4f}s ({num_updates/mode_time:.0f} updates/s)")
        
        # Benchmark matrix generation
        start_time = time.time()
        
        for _ in range(num_updates):
            view_matrix = camera_manager.get_view_matrix()
            projection_matrix = camera_manager.get_projection_matrix()
        
        matrix_generation_time = time.time() - start_time
        
        self.results['camera_manager'] = {
            'mode_update_times': mode_times,
            'matrix_generation_time': matrix_generation_time,
            'matrices_per_second': (num_updates * 2) / matrix_generation_time,  # 2 matrices per call
            'avg_update_time': sum(mode_times.values()) / len(mode_times)
        }
        
        print(f"  Matrix generation: {matrix_generation_time:.4f}s ({(num_updates*2)/matrix_generation_time:.0f} matrices/s)")
    
    def benchmark_mesh_operations(self, num_meshes=100):
        """Benchmark mesh creation and normal calculation"""
        print(f"Benchmarking mesh operations with {num_meshes} meshes...")
        
        # Generate test mesh data
        mesh_data = []
        for i in range(num_meshes):
            # Create random triangle mesh
            num_vertices = 100 + (i % 900)  # Varying complexity
            vertices = np.random.rand(num_vertices, 3).astype(np.float32) * 10
            indices = np.random.randint(0, num_vertices, size=(num_vertices // 3 * 3,)).astype(np.uint32)
            mesh_data.append((vertices, indices))
        
        # Benchmark mesh creation with normal calculation
        start_time = time.time()
        
        meshes = []
        for vertices, indices in mesh_data:
            mesh = Mesh(vertices=vertices, indices=indices)
            meshes.append(mesh)
        
        mesh_creation_time = time.time() - start_time
        
        # Benchmark normal recalculation
        start_time = time.time()
        
        for mesh in meshes:
            mesh.calculate_normals()
        
        normal_calculation_time = time.time() - start_time
        
        # Calculate average mesh complexity
        avg_vertices = sum(len(mesh.vertices) for mesh in meshes) / len(meshes)
        total_vertices = sum(len(mesh.vertices) for mesh in meshes)
        
        self.results['mesh_operations'] = {
            'mesh_creation_time': mesh_creation_time,
            'meshes_per_second': num_meshes / mesh_creation_time,
            'normal_calculation_time': normal_calculation_time,
            'normals_per_second': total_vertices / normal_calculation_time,
            'avg_vertices_per_mesh': avg_vertices,
            'total_vertices': total_vertices
        }
        
        print(f"  Mesh creation: {mesh_creation_time:.4f}s ({num_meshes/mesh_creation_time:.0f} meshes/s)")
        print(f"  Normal calculation: {normal_calculation_time:.4f}s ({total_vertices/normal_calculation_time:.0f} vertices/s)")
        print(f"  Average vertices per mesh: {avg_vertices:.0f}")
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage of rendering components"""
        print("Benchmarking memory usage...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large scene
        scene_manager = SceneManager()
        mesh = scene_manager.get_mesh("cube")
        material = scene_manager.get_material("default")
        
        objects = []
        for i in range(5000):
            obj = RenderableObject(
                name=f"object_{i}",
                mesh=mesh,
                material=material,
                transform=self._random_transform()
            )
            objects.append(obj)
            
            # Mock buffer setup
            with MagicMock():
                scene_manager.add_object(obj)
        
        scene_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create lighting system with many lights
        lighting_system = LightingSystem()
        for i in range(100):
            light = Light(
                light_type=LightType.POINT,
                position=np.random.rand(3) * 100,
                direction=np.random.rand(3) - 0.5,
                color=np.random.rand(3),
                intensity=np.random.rand()
            )
            lighting_system.add_light(light)
        
        lighting_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        self.results['memory_usage'] = {
            'initial_memory_mb': initial_memory,
            'scene_memory_mb': scene_memory,
            'lighting_memory_mb': lighting_memory,
            'scene_memory_per_object_kb': (scene_memory - initial_memory) * 1024 / 5000,
            'lighting_memory_per_light_kb': (lighting_memory - scene_memory) * 1024 / 100
        }
        
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Scene memory (5000 objects): {scene_memory:.1f} MB")
        print(f"  Lighting memory (100 lights): {lighting_memory:.1f} MB")
        print(f"  Memory per object: {(scene_memory - initial_memory) * 1024 / 5000:.2f} KB")
        print(f"  Memory per light: {(lighting_memory - scene_memory) * 1024 / 100:.2f} KB")
        
        # Cleanup
        scene_manager.cleanup()
        lighting_system.cleanup()
    
    def _random_transform(self):
        """Generate a random transformation matrix"""
        transform = np.eye(4, dtype=np.float32)
        
        # Random translation
        transform[0:3, 3] = np.random.rand(3) * 100 - 50
        
        # Random scale
        scale = np.random.rand() * 2 + 0.5
        transform[0, 0] = scale
        transform[1, 1] = scale
        transform[2, 2] = scale
        
        return transform
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("Starting rendering system benchmarks...\n")
        
        self.benchmark_scene_manager_object_creation(1000)
        print()
        
        self.benchmark_lighting_system(50)
        print()
        
        self.benchmark_camera_manager(10000)
        print()
        
        self.benchmark_mesh_operations(100)
        print()
        
        self.benchmark_memory_usage()
        print()
        
        self._print_summary()
    
    def _print_summary(self):
        """Print benchmark summary"""
        print("=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        if 'scene_manager' in self.results:
            sm = self.results['scene_manager']
            print(f"Scene Manager:")
            print(f"  Object creation: {sm['objects_per_second_creation']:.0f} objects/s")
            print(f"  Visible filtering: {sm['visible_filtering_time']:.2f}ms")
        
        if 'lighting_system' in self.results:
            ls = self.results['lighting_system']
            print(f"Lighting System:")
            print(f"  Light creation: {ls['lights_per_second']:.0f} lights/s")
            print(f"  Uniform generation: {ls['uniform_generation_time']:.2f}ms")
        
        if 'camera_manager' in self.results:
            cm = self.results['camera_manager']
            print(f"Camera Manager:")
            print(f"  Matrix generation: {cm['matrices_per_second']:.0f} matrices/s")
            print(f"  Average update time: {cm['avg_update_time']*1000:.2f}ms")
        
        if 'mesh_operations' in self.results:
            mo = self.results['mesh_operations']
            print(f"Mesh Operations:")
            print(f"  Mesh creation: {mo['meshes_per_second']:.0f} meshes/s")
            print(f"  Normal calculation: {mo['normals_per_second']:.0f} vertices/s")
        
        if 'memory_usage' in self.results:
            mu = self.results['memory_usage']
            print(f"Memory Usage:")
            print(f"  Per object: {mu['scene_memory_per_object_kb']:.2f} KB")
            print(f"  Per light: {mu['lighting_memory_per_light_kb']:.2f} KB")
        
        print("=" * 60)


def main():
    """Run the benchmark suite"""
    # Create QApplication for Qt components
    app = QApplication([])
    
    benchmark = RenderingBenchmark()
    benchmark.run_all_benchmarks()
    
    return benchmark.results


if __name__ == "__main__":
    results = main()