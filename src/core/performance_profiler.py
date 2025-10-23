"""
Performance Profiler for Robotic Car Simulation

Comprehensive performance monitoring, profiling, and optimization system
to identify bottlenecks and suggest performance improvements.
"""

import time
import threading
import psutil
import gc
import cProfile
import pstats
import io
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import functools

from PyQt6.QtCore import QObject, pyqtSignal, QTimer


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: float
    cpu_usage: float
    memory_usage: float  # MB
    frame_time: float
    physics_time: float
    ai_time: float
    rendering_time: float
    vehicle_count: int
    active_systems: List[str] = field(default_factory=list)


@dataclass
class ProfileResult:
    """Profiling result for a specific operation"""
    operation_name: str
    total_time: float
    call_count: int
    avg_time: float
    max_time: float
    min_time: float
    memory_delta: float


class PerformanceProfiler(QObject):
    """Main performance profiler class"""
    
    # Signals for performance events
    performance_warning = pyqtSignal(str, float)  # warning_type, severity
    bottleneck_detected = pyqtSignal(str, dict)   # component, metrics
    optimization_suggestion = pyqtSignal(str)     # suggestion
    
    def __init__(self):
        super().__init__()
        
        self.process = psutil.Process()
        self.is_profiling = False
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 samples
        
        # Performance thresholds
        self.thresholds = {
            'max_frame_time': 0.033,    # 30 FPS
            'max_cpu_usage': 80.0,      # 80%
            'max_memory_growth': 100.0, # 100MB per minute
            'max_physics_time': 0.016,  # 16ms
            'max_ai_time': 0.010,       # 10ms
            'max_rendering_time': 0.020 # 20ms
        }
        
        # Profiling data
        self.operation_profiles: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: List[float] = []
        self.bottleneck_history: List[Dict[str, Any]] = []
        
        # Monitoring timer
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.collect_metrics)
        
    def start_profiling(self, sample_interval: float = 0.1):
        """Start performance profiling"""
        if self.is_profiling:
            return
            
        self.is_profiling = True
        self.metrics_history.clear()
        self.operation_profiles.clear()
        self.memory_snapshots.clear()
        
        # Start monitoring timer
        self.monitor_timer.start(int(sample_interval * 1000))
        
        print("Performance profiling started")
        
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return results"""
        if not self.is_profiling:
            return {}
            
        self.is_profiling = False
        self.monitor_timer.stop()
        
        # Analyze collected data
        analysis = self.analyze_performance_data()
        
        print("Performance profiling stopped")
        return analysis
        
    def collect_metrics(self):
        """Collect current performance metrics"""
        if not self.is_profiling:
            return
            
        try:
            # System metrics
            cpu_usage = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Create metrics object (frame times will be updated by components)
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_mb,
                frame_time=0.0,
                physics_time=0.0,
                ai_time=0.0,
                rendering_time=0.0,
                vehicle_count=0
            )
            
            self.metrics_history.append(metrics)
            self.memory_snapshots.append(memory_mb)
            
            # Check for performance issues
            self.check_performance_thresholds(metrics)
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            
    def update_component_timing(self, component: str, execution_time: float):
        """Update timing for a specific component"""
        if not self.is_profiling or not self.metrics_history:
            return
            
        # Update the latest metrics
        latest_metrics = self.metrics_history[-1]
        
        if component == 'physics':
            latest_metrics.physics_time = execution_time
        elif component == 'ai':
            latest_metrics.ai_time = execution_time
        elif component == 'rendering':
            latest_metrics.rendering_time = execution_time
        elif component == 'frame':
            latest_metrics.frame_time = execution_time
            
    def update_vehicle_count(self, count: int):
        """Update current vehicle count"""
        if not self.is_profiling or not self.metrics_history:
            return
            
        self.metrics_history[-1].vehicle_count = count
        
    def check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check if performance metrics exceed thresholds"""
        # Check frame time
        if metrics.frame_time > self.thresholds['max_frame_time']:
            self.performance_warning.emit('frame_time', metrics.frame_time)
            
        # Check CPU usage
        if metrics.cpu_usage > self.thresholds['max_cpu_usage']:
            self.performance_warning.emit('cpu_usage', metrics.cpu_usage)
            
        # Check memory growth
        if len(self.memory_snapshots) >= 60:  # 1 minute of samples at 0.1s interval
            recent_memory = self.memory_snapshots[-60:]
            memory_growth = (recent_memory[-1] - recent_memory[0]) * 60  # MB per minute
            
            if memory_growth > self.thresholds['max_memory_growth']:
                self.performance_warning.emit('memory_growth', memory_growth)
                
        # Check component-specific times
        if metrics.physics_time > self.thresholds['max_physics_time']:
            self.bottleneck_detected.emit('physics', {'time': metrics.physics_time})
            
        if metrics.ai_time > self.thresholds['max_ai_time']:
            self.bottleneck_detected.emit('ai', {'time': metrics.ai_time})
            
        if metrics.rendering_time > self.thresholds['max_rendering_time']:
            self.bottleneck_detected.emit('rendering', {'time': metrics.rendering_time})
            
    def analyze_performance_data(self) -> Dict[str, Any]:
        """Analyze collected performance data"""
        if not self.metrics_history:
            return {}
            
        metrics_list = list(self.metrics_history)
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in metrics_list) / len(metrics_list)
        avg_memory = sum(m.memory_usage for m in metrics_list) / len(metrics_list)
        avg_frame_time = sum(m.frame_time for m in metrics_list if m.frame_time > 0)
        avg_frame_time = avg_frame_time / max(1, len([m for m in metrics_list if m.frame_time > 0]))
        
        # Calculate peaks
        peak_cpu = max(m.cpu_usage for m in metrics_list)
        peak_memory = max(m.memory_usage for m in metrics_list)
        peak_frame_time = max(m.frame_time for m in metrics_list)
        
        # Memory growth analysis
        memory_growth = 0.0
        if len(self.memory_snapshots) > 1:
            memory_growth = self.memory_snapshots[-1] - self.memory_snapshots[0]
            
        # Identify bottlenecks
        bottlenecks = self.identify_bottlenecks(metrics_list)
        
        # Generate optimization suggestions
        suggestions = self.generate_optimization_suggestions(metrics_list, bottlenecks)
        
        analysis = {
            'summary': {
                'avg_cpu_usage': avg_cpu,
                'avg_memory_usage': avg_memory,
                'avg_frame_time': avg_frame_time,
                'peak_cpu_usage': peak_cpu,
                'peak_memory_usage': peak_memory,
                'peak_frame_time': peak_frame_time,
                'memory_growth': memory_growth,
                'sample_count': len(metrics_list)
            },
            'bottlenecks': bottlenecks,
            'optimization_suggestions': suggestions,
            'performance_score': self.calculate_performance_score(metrics_list)
        }
        
        return analysis
        
    def identify_bottlenecks(self, metrics_list: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Analyze component timing
        physics_times = [m.physics_time for m in metrics_list if m.physics_time > 0]
        ai_times = [m.ai_time for m in metrics_list if m.ai_time > 0]
        rendering_times = [m.rendering_time for m in metrics_list if m.rendering_time > 0]
        
        if physics_times:
            avg_physics = sum(physics_times) / len(physics_times)
            if avg_physics > self.thresholds['max_physics_time']:
                bottlenecks.append({
                    'component': 'physics',
                    'severity': 'high' if avg_physics > self.thresholds['max_physics_time'] * 2 else 'medium',
                    'avg_time': avg_physics,
                    'max_time': max(physics_times)
                })
                
        if ai_times:
            avg_ai = sum(ai_times) / len(ai_times)
            if avg_ai > self.thresholds['max_ai_time']:
                bottlenecks.append({
                    'component': 'ai',
                    'severity': 'high' if avg_ai > self.thresholds['max_ai_time'] * 2 else 'medium',
                    'avg_time': avg_ai,
                    'max_time': max(ai_times)
                })
                
        if rendering_times:
            avg_rendering = sum(rendering_times) / len(rendering_times)
            if avg_rendering > self.thresholds['max_rendering_time']:
                bottlenecks.append({
                    'component': 'rendering',
                    'severity': 'high' if avg_rendering > self.thresholds['max_rendering_time'] * 2 else 'medium',
                    'avg_time': avg_rendering,
                    'max_time': max(rendering_times)
                })
                
        return bottlenecks
        
    def generate_optimization_suggestions(self, metrics_list: List[PerformanceMetrics], 
                                        bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization suggestions based on analysis"""
        suggestions = []
        
        # Analyze vehicle count vs performance
        vehicle_counts = [m.vehicle_count for m in metrics_list if m.vehicle_count > 0]
        frame_times = [m.frame_time for m in metrics_list if m.frame_time > 0]
        
        if vehicle_counts and frame_times:
            avg_vehicles = sum(vehicle_counts) / len(vehicle_counts)
            avg_frame_time = sum(frame_times) / len(frame_times)
            
            if avg_vehicles > 20 and avg_frame_time > self.thresholds['max_frame_time']:
                suggestions.append("Consider reducing maximum vehicle count or implementing level-of-detail for distant vehicles")
                
        # Component-specific suggestions
        for bottleneck in bottlenecks:
            component = bottleneck['component']
            
            if component == 'physics':
                suggestions.append("Physics bottleneck detected: Consider spatial partitioning for collision detection")
                suggestions.append("Implement physics LOD (Level of Detail) for distant objects")
                
            elif component == 'ai':
                suggestions.append("AI bottleneck detected: Consider reducing AI update frequency for distant vehicles")
                suggestions.append("Implement behavior tree caching for similar AI states")
                
            elif component == 'rendering':
                suggestions.append("Rendering bottleneck detected: Enable frustum culling and occlusion culling")
                suggestions.append("Consider reducing rendering quality or resolution")
                
        # Memory suggestions
        if len(self.memory_snapshots) > 1:
            memory_growth = self.memory_snapshots[-1] - self.memory_snapshots[0]
            if memory_growth > 50:  # 50MB growth
                suggestions.append("Memory usage growing: Check for memory leaks in vehicle lifecycle")
                suggestions.append("Consider implementing object pooling for frequently created/destroyed objects")
                
        return suggestions
        
    def calculate_performance_score(self, metrics_list: List[PerformanceMetrics]) -> float:
        """Calculate overall performance score (0-100)"""
        if not metrics_list:
            return 0.0
            
        score = 100.0
        
        # Frame rate score (40% weight)
        frame_times = [m.frame_time for m in metrics_list if m.frame_time > 0]
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            target_frame_time = 1.0 / 60.0  # 60 FPS target
            
            if avg_frame_time > target_frame_time:
                frame_penalty = min(40.0, (avg_frame_time - target_frame_time) * 1000)
                score -= frame_penalty
                
        # CPU usage score (30% weight)
        avg_cpu = sum(m.cpu_usage for m in metrics_list) / len(metrics_list)
        if avg_cpu > 50.0:
            cpu_penalty = min(30.0, (avg_cpu - 50.0) * 0.6)
            score -= cpu_penalty
            
        # Memory efficiency score (20% weight)
        if len(self.memory_snapshots) > 1:
            memory_growth = self.memory_snapshots[-1] - self.memory_snapshots[0]
            if memory_growth > 10:  # 10MB growth penalty
                memory_penalty = min(20.0, memory_growth * 0.2)
                score -= memory_penalty
                
        # Stability score (10% weight)
        frame_time_variance = 0.0
        if len(frame_times) > 1:
            avg_frame_time = sum(frame_times) / len(frame_times)
            frame_time_variance = sum((ft - avg_frame_time) ** 2 for ft in frame_times) / len(frame_times)
            
            if frame_time_variance > 0.001:  # High variance penalty
                stability_penalty = min(10.0, frame_time_variance * 1000)
                score -= stability_penalty
                
        return max(0.0, score)


def profile_operation(operation_name: str):
    """Decorator to profile specific operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get profiler instance if available
            profiler = None
            if hasattr(args[0], 'performance_profiler'):
                profiler = args[0].performance_profiler
            elif hasattr(args[0], 'simulation_app') and hasattr(args[0].simulation_app, 'performance_profiler'):
                profiler = args[0].simulation_app.performance_profiler
                
            if profiler and profiler.is_profiling:
                start_time = time.time()
                start_memory = profiler.process.memory_info().rss / 1024 / 1024
                
                try:
                    result = func(*args, **kwargs)
                    
                    end_time = time.time()
                    end_memory = profiler.process.memory_info().rss / 1024 / 1024
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    # Record operation profile
                    profiler.operation_profiles[operation_name].append(execution_time)
                    
                    # Update component timing if applicable
                    if operation_name in ['physics_update', 'ai_update', 'render_frame']:
                        component = operation_name.split('_')[0]
                        profiler.update_component_timing(component, execution_time)
                        
                    return result
                    
                except Exception as e:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    profiler.operation_profiles[f"{operation_name}_error"].append(execution_time)
                    raise
            else:
                return func(*args, **kwargs)
                
        return wrapper
    return decorator


@contextmanager
def profile_context(profiler: PerformanceProfiler, operation_name: str):
    """Context manager for profiling code blocks"""
    if not profiler.is_profiling:
        yield
        return
        
    start_time = time.time()
    start_memory = profiler.process.memory_info().rss / 1024 / 1024
    
    try:
        yield
        
    finally:
        end_time = time.time()
        end_memory = profiler.process.memory_info().rss / 1024 / 1024
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        profiler.operation_profiles[operation_name].append(execution_time)


class MemoryProfiler:
    """Specialized memory profiler"""
    
    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []
        self.tracking_objects: Dict[str, int] = {}
        
    def take_snapshot(self, label: str = ""):
        """Take a memory snapshot"""
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Count objects by type
        object_counts = defaultdict(int)
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] += 1
            
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'objects_collected': collected,
            'object_counts': dict(object_counts),
            'total_objects': len(gc.get_objects())
        }
        
        self.snapshots.append(snapshot)
        return snapshot
        
    def compare_snapshots(self, snapshot1_idx: int, snapshot2_idx: int) -> Dict[str, Any]:
        """Compare two memory snapshots"""
        if snapshot1_idx >= len(self.snapshots) or snapshot2_idx >= len(self.snapshots):
            return {}
            
        snap1 = self.snapshots[snapshot1_idx]
        snap2 = self.snapshots[snapshot2_idx]
        
        memory_diff = snap2['rss_mb'] - snap1['rss_mb']
        object_diff = snap2['total_objects'] - snap1['total_objects']
        
        # Find object type differences
        type_differences = {}
        all_types = set(snap1['object_counts'].keys()) | set(snap2['object_counts'].keys())
        
        for obj_type in all_types:
            count1 = snap1['object_counts'].get(obj_type, 0)
            count2 = snap2['object_counts'].get(obj_type, 0)
            diff = count2 - count1
            
            if abs(diff) > 10:  # Only report significant differences
                type_differences[obj_type] = diff
                
        return {
            'memory_diff_mb': memory_diff,
            'object_count_diff': object_diff,
            'type_differences': type_differences,
            'time_diff': snap2['timestamp'] - snap1['timestamp']
        }
        
    def detect_memory_leaks(self) -> List[str]:
        """Detect potential memory leaks"""
        if len(self.snapshots) < 3:
            return ["Insufficient snapshots for leak detection"]
            
        leaks = []
        
        # Check for consistent memory growth
        recent_snapshots = self.snapshots[-5:]  # Last 5 snapshots
        memory_values = [s['rss_mb'] for s in recent_snapshots]
        
        # Simple trend analysis
        if len(memory_values) >= 3:
            growth_trend = all(memory_values[i] <= memory_values[i+1] 
                             for i in range(len(memory_values)-1))
            
            if growth_trend:
                total_growth = memory_values[-1] - memory_values[0]
                if total_growth > 20:  # 20MB growth
                    leaks.append(f"Consistent memory growth detected: {total_growth:.1f}MB")
                    
        # Check for object count growth
        if len(self.snapshots) >= 2:
            comparison = self.compare_snapshots(-2, -1)
            
            for obj_type, diff in comparison['type_differences'].items():
                if diff > 100:  # More than 100 new objects
                    leaks.append(f"Potential leak in {obj_type}: +{diff} objects")
                    
        return leaks