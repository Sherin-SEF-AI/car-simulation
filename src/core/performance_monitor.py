"""
Performance Monitor for Robotic Car Simulation

This module provides comprehensive performance monitoring capabilities for tracking
frame rates, system resources, and automatic optimization suggestions.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from PyQt6.QtCore import QObject, pyqtSignal, QTimer


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_usage: float  # Percentage
    memory_usage: float  # Percentage
    memory_used_mb: float  # MB
    memory_available_mb: float  # MB
    gpu_usage: Optional[float] = None  # Percentage (if available)
    gpu_memory_mb: Optional[float] = None  # MB (if available)
    disk_io_read_mb: float = 0.0  # MB/s
    disk_io_write_mb: float = 0.0  # MB/s
    network_sent_mb: float = 0.0  # MB/s
    network_recv_mb: float = 0.0  # MB/s


@dataclass
class FrameMetrics:
    """Frame rendering performance metrics"""
    timestamp: float
    frame_time_ms: float  # Milliseconds
    fps: float
    render_time_ms: float  # Time spent rendering
    physics_time_ms: float  # Time spent on physics
    ai_time_ms: float  # Time spent on AI processing
    frame_drops: int  # Number of dropped frames
    vsync_enabled: bool = True


@dataclass
class PerformanceProfile:
    """Performance profile for different quality settings"""
    name: str
    description: str
    target_fps: float
    max_vehicles: int
    physics_quality: str  # 'low', 'medium', 'high'
    render_quality: str  # 'low', 'medium', 'high', 'ultra'
    particle_density: float  # 0.0 to 1.0
    shadow_quality: str  # 'off', 'low', 'medium', 'high'
    texture_quality: str  # 'low', 'medium', 'high'
    anti_aliasing: str  # 'off', 'fxaa', 'msaa2x', 'msaa4x'


@dataclass
class OptimizationSuggestion:
    """Performance optimization suggestion"""
    category: str  # 'graphics', 'physics', 'ai', 'system'
    priority: str  # 'low', 'medium', 'high', 'critical'
    title: str
    description: str
    impact: str  # 'low', 'medium', 'high'
    difficulty: str  # 'easy', 'medium', 'hard'
    action: str  # Specific action to take
    estimated_improvement: float  # Percentage improvement estimate


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring"""
    monitoring_frequency: float = 1.0  # Hz
    metrics_history_size: int = 300  # 5 minutes at 1Hz
    enable_system_monitoring: bool = True
    enable_frame_monitoring: bool = True
    enable_auto_optimization: bool = False
    target_fps: float = 60.0
    min_fps_threshold: float = 30.0
    max_cpu_threshold: float = 80.0
    max_memory_threshold: float = 85.0
    optimization_check_interval: float = 30.0  # seconds


class PerformanceMonitor(QObject):
    """
    Comprehensive performance monitoring system
    
    Tracks frame rates, system resources, and provides automatic
    optimization suggestions for maintaining optimal performance.
    """
    
    # Signals
    metrics_updated = pyqtSignal(object, object)  # SystemMetrics, FrameMetrics
    performance_warning = pyqtSignal(str, str, float)  # category, message, severity
    optimization_suggested = pyqtSignal(object)  # OptimizationSuggestion
    profile_changed = pyqtSignal(str)  # profile_name
    bottleneck_detected = pyqtSignal(str, object)  # bottleneck_type, details dict
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        super().__init__()
        self.config = config or PerformanceConfig()
        
        # Metrics storage
        self.system_metrics_history: deque = deque(maxlen=self.config.metrics_history_size)
        self.frame_metrics_history: deque = deque(maxlen=self.config.metrics_history_size)
        
        # Performance profiles
        self.performance_profiles = self._create_default_profiles()
        self.current_profile = self.performance_profiles['medium']
        
        # Monitoring state
        self.is_monitoring = False
        self.last_optimization_check = 0.0
        
        # System monitoring
        self.process = psutil.Process()
        self.last_disk_io = None
        self.last_network_io = None
        
        # Frame timing
        self.frame_start_time = 0.0
        self.render_start_time = 0.0
        self.physics_start_time = 0.0
        self.ai_start_time = 0.0
        self.frame_times: deque = deque(maxlen=60)  # Last 60 frames for FPS calculation
        
        # Monitoring timer
        self.monitoring_timer = QTimer()
        self.monitoring_timer.timeout.connect(self._collect_metrics)
        
        # Optimization suggestions cache
        self.cached_suggestions: List[OptimizationSuggestion] = []
        self.last_suggestion_update = 0.0
        
        # Performance callbacks
        self.performance_callbacks: Dict[str, Callable] = {}
    
    def _create_default_profiles(self) -> Dict[str, PerformanceProfile]:
        """Create default performance profiles"""
        return {
            'low': PerformanceProfile(
                name='Low Quality',
                description='Optimized for older hardware and maximum performance',
                target_fps=60.0,
                max_vehicles=10,
                physics_quality='low',
                render_quality='low',
                particle_density=0.3,
                shadow_quality='off',
                texture_quality='low',
                anti_aliasing='off'
            ),
            'medium': PerformanceProfile(
                name='Medium Quality',
                description='Balanced performance and visual quality',
                target_fps=60.0,
                max_vehicles=25,
                physics_quality='medium',
                render_quality='medium',
                particle_density=0.6,
                shadow_quality='low',
                texture_quality='medium',
                anti_aliasing='fxaa'
            ),
            'high': PerformanceProfile(
                name='High Quality',
                description='High visual quality for modern hardware',
                target_fps=60.0,
                max_vehicles=50,
                physics_quality='high',
                render_quality='high',
                particle_density=0.8,
                shadow_quality='medium',
                texture_quality='high',
                anti_aliasing='msaa2x'
            ),
            'ultra': PerformanceProfile(
                name='Ultra Quality',
                description='Maximum visual quality for high-end hardware',
                target_fps=60.0,
                max_vehicles=100,
                physics_quality='high',
                render_quality='ultra',
                particle_density=1.0,
                shadow_quality='high',
                texture_quality='high',
                anti_aliasing='msaa4x'
            )
        }
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start monitoring timer
        interval_ms = int(1000 / self.config.monitoring_frequency)
        self.monitoring_timer.start(interval_ms)
        
        # Initialize baseline measurements
        self.last_disk_io = psutil.disk_io_counters()
        self.last_network_io = psutil.net_io_counters()
        
        print("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.monitoring_timer.stop()
        
        print("Performance monitoring stopped")
    
    def register_performance_callback(self, name: str, callback: Callable):
        """Register a callback for performance data collection"""
        self.performance_callbacks[name] = callback
    
    def unregister_performance_callback(self, name: str):
        """Unregister a performance callback"""
        if name in self.performance_callbacks:
            del self.performance_callbacks[name]
    
    def begin_frame(self):
        """Mark the beginning of a frame for timing"""
        self.frame_start_time = time.perf_counter()
    
    def begin_render(self):
        """Mark the beginning of rendering phase"""
        self.render_start_time = time.perf_counter()
    
    def end_render(self):
        """Mark the end of rendering phase"""
        if self.render_start_time > 0:
            render_time = (time.perf_counter() - self.render_start_time) * 1000
            return render_time
        return 0.0
    
    def begin_physics(self):
        """Mark the beginning of physics phase"""
        self.physics_start_time = time.perf_counter()
    
    def end_physics(self):
        """Mark the end of physics phase"""
        if self.physics_start_time > 0:
            physics_time = (time.perf_counter() - self.physics_start_time) * 1000
            return physics_time
        return 0.0
    
    def begin_ai(self):
        """Mark the beginning of AI processing phase"""
        self.ai_start_time = time.perf_counter()
    
    def end_ai(self):
        """Mark the end of AI processing phase"""
        if self.ai_start_time > 0:
            ai_time = (time.perf_counter() - self.ai_start_time) * 1000
            return ai_time
        return 0.0
    
    def end_frame(self, render_time_ms: float = 0.0, physics_time_ms: float = 0.0, 
                  ai_time_ms: float = 0.0, frame_drops: int = 0):
        """Mark the end of a frame and record timing"""
        if self.frame_start_time == 0:
            return
        
        current_time = time.perf_counter()
        frame_time_ms = (current_time - self.frame_start_time) * 1000
        
        # Record frame time for FPS calculation
        self.frame_times.append(current_time)
        
        # Calculate FPS
        fps = self._calculate_fps()
        
        # Create frame metrics
        frame_metrics = FrameMetrics(
            timestamp=time.time(),
            frame_time_ms=frame_time_ms,
            fps=fps,
            render_time_ms=render_time_ms,
            physics_time_ms=physics_time_ms,
            ai_time_ms=ai_time_ms,
            frame_drops=frame_drops
        )
        
        # Store metrics
        if self.config.enable_frame_monitoring:
            self.frame_metrics_history.append(frame_metrics)
        
        # Reset frame start time
        self.frame_start_time = 0.0
        
        return frame_metrics
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS based on recent frame times"""
        if len(self.frame_times) < 2:
            return 0.0
        
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span <= 0:
            return 0.0
        
        return (len(self.frame_times) - 1) / time_span
    
    def _collect_metrics(self):
        """Collect system and performance metrics"""
        try:
            current_time = time.time()
            
            # Collect system metrics
            if self.config.enable_system_monitoring:
                system_metrics = self._collect_system_metrics(current_time)
                self.system_metrics_history.append(system_metrics)
            else:
                system_metrics = None
            
            # Get latest frame metrics
            frame_metrics = None
            if self.frame_metrics_history:
                frame_metrics = self.frame_metrics_history[-1]
            
            # Emit metrics update
            self.metrics_updated.emit(system_metrics, frame_metrics)
            
            # Check for performance warnings
            self._check_performance_warnings(system_metrics, frame_metrics)
            
            # Check for optimization opportunities
            if (self.config.enable_auto_optimization and 
                current_time - self.last_optimization_check > self.config.optimization_check_interval):
                self._check_optimization_opportunities()
                self.last_optimization_check = current_time
            
        except Exception as e:
            print(f"Error collecting performance metrics: {e}")
    
    def _collect_system_metrics(self, timestamp: float) -> SystemMetrics:
        """Collect system performance metrics"""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk I/O
        disk_io_read_mb = 0.0
        disk_io_write_mb = 0.0
        try:
            current_disk_io = psutil.disk_io_counters()
            if self.last_disk_io:
                read_bytes = current_disk_io.read_bytes - self.last_disk_io.read_bytes
                write_bytes = current_disk_io.write_bytes - self.last_disk_io.write_bytes
                time_delta = 1.0 / self.config.monitoring_frequency
                
                disk_io_read_mb = (read_bytes / (1024 * 1024)) / time_delta
                disk_io_write_mb = (write_bytes / (1024 * 1024)) / time_delta
            
            self.last_disk_io = current_disk_io
        except Exception:
            pass
        
        # Network I/O
        network_sent_mb = 0.0
        network_recv_mb = 0.0
        try:
            current_network_io = psutil.net_io_counters()
            if self.last_network_io:
                sent_bytes = current_network_io.bytes_sent - self.last_network_io.bytes_sent
                recv_bytes = current_network_io.bytes_recv - self.last_network_io.bytes_recv
                time_delta = 1.0 / self.config.monitoring_frequency
                
                network_sent_mb = (sent_bytes / (1024 * 1024)) / time_delta
                network_recv_mb = (recv_bytes / (1024 * 1024)) / time_delta
            
            self.last_network_io = current_network_io
        except Exception:
            pass
        
        # GPU metrics (if available)
        gpu_usage = None
        gpu_memory_mb = None
        try:
            # This would require additional GPU monitoring libraries
            # For now, we'll leave it as None
            pass
        except Exception:
            pass
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            gpu_usage=gpu_usage,
            gpu_memory_mb=gpu_memory_mb,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
    
    def _check_performance_warnings(self, system_metrics: Optional[SystemMetrics], 
                                  frame_metrics: Optional[FrameMetrics]):
        """Check for performance warnings and emit signals"""
        if system_metrics:
            # CPU usage warning
            if system_metrics.cpu_usage > self.config.max_cpu_threshold:
                self.performance_warning.emit(
                    'cpu',
                    f'High CPU usage: {system_metrics.cpu_usage:.1f}%',
                    0.8
                )
            
            # Memory usage warning
            if system_metrics.memory_usage > self.config.max_memory_threshold:
                self.performance_warning.emit(
                    'memory',
                    f'High memory usage: {system_metrics.memory_usage:.1f}%',
                    0.8
                )
        
        if frame_metrics:
            # FPS warning
            if frame_metrics.fps < self.config.min_fps_threshold:
                self.performance_warning.emit(
                    'fps',
                    f'Low FPS: {frame_metrics.fps:.1f}',
                    0.9
                )
            
            # Frame time warning
            target_frame_time = 1000.0 / self.config.target_fps
            if frame_metrics.frame_time_ms > target_frame_time * 1.5:
                self.performance_warning.emit(
                    'frame_time',
                    f'High frame time: {frame_metrics.frame_time_ms:.1f}ms',
                    0.7
                )
    
    def _check_optimization_opportunities(self):
        """Check for optimization opportunities and generate suggestions"""
        suggestions = []
        
        # Analyze recent performance data
        if len(self.system_metrics_history) < 10 or len(self.frame_metrics_history) < 10:
            return
        
        recent_system = list(self.system_metrics_history)[-10:]
        recent_frames = list(self.frame_metrics_history)[-10:]
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_system) / len(recent_system)
        avg_memory = sum(m.memory_usage for m in recent_system) / len(recent_system)
        avg_fps = sum(m.fps for m in recent_frames) / len(recent_frames)
        avg_frame_time = sum(m.frame_time_ms for m in recent_frames) / len(recent_frames)
        
        # CPU-based suggestions
        if avg_cpu > 70:
            suggestions.append(OptimizationSuggestion(
                category='system',
                priority='high',
                title='High CPU Usage Detected',
                description='CPU usage is consistently high, consider reducing simulation complexity',
                impact='high',
                difficulty='medium',
                action='Reduce number of vehicles or lower physics quality',
                estimated_improvement=15.0
            ))
        
        # Memory-based suggestions
        if avg_memory > 75:
            suggestions.append(OptimizationSuggestion(
                category='system',
                priority='medium',
                title='High Memory Usage',
                description='Memory usage is high, consider reducing texture quality or particle density',
                impact='medium',
                difficulty='easy',
                action='Lower texture quality or reduce particle effects',
                estimated_improvement=10.0
            ))
        
        # FPS-based suggestions
        if avg_fps < self.config.target_fps * 0.8:
            # Analyze bottlenecks
            avg_render_time = sum(m.render_time_ms for m in recent_frames) / len(recent_frames)
            avg_physics_time = sum(m.physics_time_ms for m in recent_frames) / len(recent_frames)
            avg_ai_time = sum(m.ai_time_ms for m in recent_frames) / len(recent_frames)
            
            total_accounted = avg_render_time + avg_physics_time + avg_ai_time
            
            if avg_render_time > avg_frame_time * 0.4:
                suggestions.append(OptimizationSuggestion(
                    category='graphics',
                    priority='high',
                    title='Rendering Bottleneck Detected',
                    description='Rendering is taking too much time per frame',
                    impact='high',
                    difficulty='easy',
                    action='Reduce render quality, disable shadows, or lower anti-aliasing',
                    estimated_improvement=20.0
                ))
            
            if avg_physics_time > avg_frame_time * 0.3:
                suggestions.append(OptimizationSuggestion(
                    category='physics',
                    priority='medium',
                    title='Physics Bottleneck Detected',
                    description='Physics simulation is consuming too much CPU time',
                    impact='medium',
                    difficulty='medium',
                    action='Reduce physics quality or limit number of physics objects',
                    estimated_improvement=15.0
                ))
            
            if avg_ai_time > avg_frame_time * 0.2:
                suggestions.append(OptimizationSuggestion(
                    category='ai',
                    priority='medium',
                    title='AI Processing Bottleneck',
                    description='AI processing is taking too much time per frame',
                    impact='medium',
                    difficulty='hard',
                    action='Optimize AI algorithms or reduce AI update frequency',
                    estimated_improvement=12.0
                ))
        
        # Update cached suggestions
        self.cached_suggestions = suggestions
        self.last_suggestion_update = time.time()
        
        # Emit suggestions
        for suggestion in suggestions:
            self.optimization_suggested.emit(suggestion)
    
    def get_performance_summary(self, time_range_minutes: float = 5.0) -> Dict[str, Any]:
        """Get performance summary for the specified time range"""
        cutoff_time = time.time() - (time_range_minutes * 60)
        
        # Filter recent metrics
        recent_system = [m for m in self.system_metrics_history if m.timestamp >= cutoff_time]
        recent_frames = [m for m in self.frame_metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_system and not recent_frames:
            return {}
        
        summary = {
            'time_range_minutes': time_range_minutes,
            'sample_count': len(recent_system),
            'current_profile': self.current_profile.name
        }
        
        # System metrics summary
        if recent_system:
            cpu_values = [m.cpu_usage for m in recent_system]
            memory_values = [m.memory_usage for m in recent_system]
            
            summary['system'] = {
                'cpu_usage': {
                    'average': sum(cpu_values) / len(cpu_values),
                    'min': min(cpu_values),
                    'max': max(cpu_values)
                },
                'memory_usage': {
                    'average': sum(memory_values) / len(memory_values),
                    'min': min(memory_values),
                    'max': max(memory_values)
                },
                'memory_used_mb': recent_system[-1].memory_used_mb if recent_system else 0
            }
        
        # Frame metrics summary
        if recent_frames:
            fps_values = [m.fps for m in recent_frames]
            frame_time_values = [m.frame_time_ms for m in recent_frames]
            
            summary['performance'] = {
                'fps': {
                    'average': sum(fps_values) / len(fps_values),
                    'min': min(fps_values),
                    'max': max(fps_values)
                },
                'frame_time_ms': {
                    'average': sum(frame_time_values) / len(frame_time_values),
                    'min': min(frame_time_values),
                    'max': max(frame_time_values)
                },
                'target_fps': self.config.target_fps,
                'fps_stability': self._calculate_fps_stability(fps_values)
            }
        
        return summary
    
    def _calculate_fps_stability(self, fps_values: List[float]) -> float:
        """Calculate FPS stability score (0.0 to 1.0)"""
        if len(fps_values) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        mean_fps = sum(fps_values) / len(fps_values)
        if mean_fps == 0:
            return 0.0
        
        variance = sum((fps - mean_fps) ** 2 for fps in fps_values) / len(fps_values)
        std_dev = variance ** 0.5
        cv = std_dev / mean_fps
        
        # Convert to stability score (lower CV = higher stability)
        stability = max(0.0, 1.0 - cv)
        return stability
    
    def set_performance_profile(self, profile_name: str) -> bool:
        """Set the current performance profile"""
        if profile_name not in self.performance_profiles:
            return False
        
        self.current_profile = self.performance_profiles[profile_name]
        self.config.target_fps = self.current_profile.target_fps
        
        self.profile_changed.emit(profile_name)
        
        # Apply profile settings through callbacks
        for callback in self.performance_callbacks.values():
            try:
                callback('profile_changed', self.current_profile)
            except Exception as e:
                print(f"Error in performance callback: {e}")
        
        return True
    
    def get_optimization_suggestions(self, force_update: bool = False) -> List[OptimizationSuggestion]:
        """Get current optimization suggestions"""
        if force_update or time.time() - self.last_suggestion_update > 60.0:
            self._check_optimization_opportunities()
        
        return self.cached_suggestions.copy()
    
    def apply_optimization_suggestion(self, suggestion: OptimizationSuggestion) -> bool:
        """Apply an optimization suggestion"""
        try:
            # This would implement the actual optimization actions
            # For now, we'll just emit a signal and return success
            
            # Notify callbacks about the optimization
            for callback in self.performance_callbacks.values():
                try:
                    callback('optimization_applied', suggestion)
                except Exception as e:
                    print(f"Error in optimization callback: {e}")
            
            return True
            
        except Exception as e:
            print(f"Error applying optimization: {e}")
            return False
    
    def detect_bottlenecks(self) -> Dict[str, Any]:
        """Detect performance bottlenecks"""
        if len(self.frame_metrics_history) < 10:
            return {}
        
        recent_frames = list(self.frame_metrics_history)[-10:]
        
        # Calculate average times
        avg_frame_time = sum(m.frame_time_ms for m in recent_frames) / len(recent_frames)
        avg_render_time = sum(m.render_time_ms for m in recent_frames) / len(recent_frames)
        avg_physics_time = sum(m.physics_time_ms for m in recent_frames) / len(recent_frames)
        avg_ai_time = sum(m.ai_time_ms for m in recent_frames) / len(recent_frames)
        
        bottlenecks = {}
        
        # Identify bottlenecks
        if avg_render_time > avg_frame_time * 0.4:
            bottlenecks['rendering'] = {
                'severity': 'high' if avg_render_time > avg_frame_time * 0.6 else 'medium',
                'percentage': (avg_render_time / avg_frame_time) * 100,
                'avg_time_ms': avg_render_time
            }
        
        if avg_physics_time > avg_frame_time * 0.3:
            bottlenecks['physics'] = {
                'severity': 'high' if avg_physics_time > avg_frame_time * 0.5 else 'medium',
                'percentage': (avg_physics_time / avg_frame_time) * 100,
                'avg_time_ms': avg_physics_time
            }
        
        if avg_ai_time > avg_frame_time * 0.2:
            bottlenecks['ai'] = {
                'severity': 'medium',
                'percentage': (avg_ai_time / avg_frame_time) * 100,
                'avg_time_ms': avg_ai_time
            }
        
        # Check system bottlenecks
        if self.system_metrics_history:
            recent_system = list(self.system_metrics_history)[-10:]
            avg_cpu = sum(m.cpu_usage for m in recent_system) / len(recent_system)
            avg_memory = sum(m.memory_usage for m in recent_system) / len(recent_system)
            
            if avg_cpu > 80:
                bottlenecks['cpu'] = {
                    'severity': 'high' if avg_cpu > 90 else 'medium',
                    'percentage': avg_cpu,
                    'avg_usage': avg_cpu
                }
            
            if avg_memory > 85:
                bottlenecks['memory'] = {
                    'severity': 'high' if avg_memory > 95 else 'medium',
                    'percentage': avg_memory,
                    'avg_usage': avg_memory
                }
        
        # Emit bottleneck detection signal
        if bottlenecks:
            for bottleneck_type, details in bottlenecks.items():
                self.bottleneck_detected.emit(bottleneck_type, details)
        
        return bottlenecks
    
    def export_performance_data(self, output_file: str, format: str = 'json') -> bool:
        """Export performance data to file"""
        try:
            export_data = {
                'system_metrics': [asdict(m) for m in self.system_metrics_history],
                'frame_metrics': [asdict(m) for m in self.frame_metrics_history],
                'current_profile': asdict(self.current_profile),
                'optimization_suggestions': [asdict(s) for s in self.cached_suggestions],
                'config': asdict(self.config),
                'export_timestamp': datetime.now().isoformat()
            }
            
            if format.lower() == 'json':
                import json
                with open(output_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                print(f"Unsupported export format: {format}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error exporting performance data: {e}")
            return False
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.system_metrics_history.clear()
        self.frame_metrics_history.clear()
        self.frame_times.clear()
        self.cached_suggestions.clear()
        self.last_suggestion_update = 0.0
        self.last_optimization_check = 0.0
    
    def get_current_metrics(self) -> Tuple[Optional[SystemMetrics], Optional[FrameMetrics]]:
        """Get the most recent metrics"""
        system_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
        frame_metrics = self.frame_metrics_history[-1] if self.frame_metrics_history else None
        return system_metrics, frame_metrics