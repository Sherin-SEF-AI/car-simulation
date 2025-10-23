"""
Automated Test Execution Framework for Robotic Car Simulation

This module provides comprehensive test execution, reporting, and performance regression testing
for the entire simulation system.
"""

import unittest
import sys
import time
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback
import psutil
import gc

# Import all test modules
from . import (
    test_ai_system_behavior_tree,
    test_analytics_engine,
    test_behavior_editor,
    test_behavior_tree,
    test_camera_system,
    test_challenge_system,
    test_code_generator,
    test_collision_detection,
    test_custom_challenge_creation,
    test_data_visualization,
    test_enhanced_control_panel,
    test_environment_system,
    test_help_system,
    test_main_window_interface,
    test_map_editor,
    test_ml_integration,
    test_multi_vehicle_manager,
    test_particle_system,
    test_path_planning,
    test_performance_monitor,
    test_progress_tracking,
    test_recording_system,
    test_rendering_pipeline,
    test_replay_system,
    test_scenario_validation,
    test_sensor_simulation,
    test_surface_physics,
    test_traffic_simulation,
    test_vehicle_coordination,
    test_vehicle_dynamics,
    test_visual_programming_integration,
    test_weather_system_integration
)


@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    status: str  # 'passed', 'failed', 'error', 'skipped'
    execution_time: float
    memory_usage: float
    error_message: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Complete test suite result"""
    timestamp: datetime
    total_tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    total_time: float
    peak_memory: float
    test_results: List[TestResult]
    performance_baseline: Optional[Dict[str, float]] = None


class PerformanceMonitor:
    """Monitor system performance during test execution"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_file = "tests/performance_baseline.json"
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return self.process.cpu_percent()
        
    def load_baseline(self) -> Dict[str, float]:
        """Load performance baseline from file"""
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return {}
        
    def save_baseline(self, performance_data: Dict[str, float]):
        """Save performance baseline to file"""
        with open(self.baseline_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
            
    def check_regression(self, current_performance: Dict[str, float], 
                        threshold: float = 0.2) -> List[str]:
        """Check for performance regressions"""
        baseline = self.load_baseline()
        regressions = []
        
        for test_name, current_time in current_performance.items():
            if test_name in baseline:
                baseline_time = baseline[test_name]
                if current_time > baseline_time * (1 + threshold):
                    regression_pct = ((current_time - baseline_time) / baseline_time) * 100
                    regressions.append(f"{test_name}: {regression_pct:.1f}% slower")
                    
        return regressions


class TestFramework:
    """Comprehensive test execution framework"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.test_modules = [
            test_ai_system_behavior_tree,
            test_analytics_engine,
            test_behavior_editor,
            test_behavior_tree,
            test_camera_system,
            test_challenge_system,
            test_code_generator,
            test_collision_detection,
            test_custom_challenge_creation,
            test_data_visualization,
            test_enhanced_control_panel,
            test_environment_system,
            test_help_system,
            test_main_window_interface,
            test_map_editor,
            test_ml_integration,
            test_multi_vehicle_manager,
            test_particle_system,
            test_path_planning,
            test_performance_monitor,
            test_progress_tracking,
            test_recording_system,
            test_rendering_pipeline,
            test_replay_system,
            test_scenario_validation,
            test_sensor_simulation,
            test_surface_physics,
            test_traffic_simulation,
            test_vehicle_coordination,
            test_vehicle_dynamics,
            test_visual_programming_integration,
            test_weather_system_integration
        ]
        
    def run_single_test(self, test_case: unittest.TestCase) -> TestResult:
        """Run a single test case with performance monitoring"""
        test_name = f"{test_case.__class__.__name__}.{test_case._testMethodName}"
        
        # Clear memory before test
        gc.collect()
        initial_memory = self.performance_monitor.get_memory_usage()
        
        start_time = time.time()
        result = unittest.TestResult()
        
        try:
            test_case.run(result)
            execution_time = time.time() - start_time
            peak_memory = self.performance_monitor.get_memory_usage()
            
            if result.wasSuccessful():
                status = 'passed'
                error_message = None
                traceback_str = None
            elif result.failures:
                status = 'failed'
                error_message = result.failures[0][1]
                traceback_str = result.failures[0][1]
            elif result.errors:
                status = 'error'
                error_message = result.errors[0][1]
                traceback_str = result.errors[0][1]
            else:
                status = 'skipped'
                error_message = None
                traceback_str = None
                
        except Exception as e:
            execution_time = time.time() - start_time
            peak_memory = self.performance_monitor.get_memory_usage()
            status = 'error'
            error_message = str(e)
            traceback_str = traceback.format_exc()
            
        return TestResult(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            memory_usage=peak_memory - initial_memory,
            error_message=error_message,
            traceback=traceback_str
        )
        
    def run_test_module(self, module) -> List[TestResult]:
        """Run all tests in a module"""
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        results = []
        
        for test_group in suite:
            if hasattr(test_group, '__iter__'):
                for test_case in test_group:
                    if hasattr(test_case, '__iter__'):
                        for individual_test in test_case:
                            results.append(self.run_single_test(individual_test))
                    else:
                        results.append(self.run_single_test(test_case))
            else:
                results.append(self.run_single_test(test_group))
                
        return results
        
    def run_all_tests(self, update_baseline: bool = False) -> TestSuiteResult:
        """Run complete test suite"""
        print("Starting comprehensive test suite execution...")
        start_time = time.time()
        initial_memory = self.performance_monitor.get_memory_usage()
        
        all_results = []
        performance_data = {}
        
        for module in self.test_modules:
            print(f"Running tests in {module.__name__}...")
            module_results = self.run_test_module(module)
            all_results.extend(module_results)
            
            # Collect performance data
            for result in module_results:
                performance_data[result.test_name] = result.execution_time
                
        total_time = time.time() - start_time
        peak_memory = self.performance_monitor.get_memory_usage()
        
        # Calculate statistics
        passed = sum(1 for r in all_results if r.status == 'passed')
        failed = sum(1 for r in all_results if r.status == 'failed')
        errors = sum(1 for r in all_results if r.status == 'error')
        skipped = sum(1 for r in all_results if r.status == 'skipped')
        
        # Check for performance regressions
        regressions = self.performance_monitor.check_regression(performance_data)
        if regressions:
            print("\nPerformance Regressions Detected:")
            for regression in regressions:
                print(f"  - {regression}")
                
        # Update baseline if requested
        if update_baseline:
            self.performance_monitor.save_baseline(performance_data)
            print("Performance baseline updated.")
            
        suite_result = TestSuiteResult(
            timestamp=datetime.now(),
            total_tests=len(all_results),
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            total_time=total_time,
            peak_memory=peak_memory - initial_memory,
            test_results=all_results,
            performance_baseline=self.performance_monitor.load_baseline()
        )
        
        return suite_result
        
    def generate_report(self, result: TestSuiteResult, output_file: str = None):
        """Generate comprehensive test report"""
        report = []
        report.append("=" * 80)
        report.append("ROBOTIC CAR SIMULATION - COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Execution Time: {result.timestamp}")
        report.append(f"Total Tests: {result.total_tests}")
        report.append(f"Passed: {result.passed}")
        report.append(f"Failed: {result.failed}")
        report.append(f"Errors: {result.errors}")
        report.append(f"Skipped: {result.skipped}")
        report.append(f"Success Rate: {(result.passed / result.total_tests * 100):.1f}%")
        report.append(f"Total Execution Time: {result.total_time:.2f}s")
        report.append(f"Peak Memory Usage: {result.peak_memory:.2f}MB")
        report.append("")
        
        # Failed tests details
        if result.failed > 0 or result.errors > 0:
            report.append("FAILED/ERROR TESTS:")
            report.append("-" * 40)
            for test_result in result.test_results:
                if test_result.status in ['failed', 'error']:
                    report.append(f"Test: {test_result.test_name}")
                    report.append(f"Status: {test_result.status.upper()}")
                    report.append(f"Error: {test_result.error_message}")
                    report.append("")
                    
        # Performance summary
        report.append("PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        slowest_tests = sorted(result.test_results, 
                             key=lambda x: x.execution_time, reverse=True)[:10]
        for test_result in slowest_tests:
            report.append(f"{test_result.test_name}: {test_result.execution_time:.3f}s")
            
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        else:
            print(report_text)
            
        return report_text


def run_comprehensive_tests(update_baseline: bool = False, 
                          output_file: str = None) -> TestSuiteResult:
    """Main entry point for running comprehensive tests"""
    framework = TestFramework()
    result = framework.run_all_tests(update_baseline=update_baseline)
    framework.generate_report(result, output_file=output_file)
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--update-baseline", action="store_true",
                       help="Update performance baseline")
    parser.add_argument("--output", type=str,
                       help="Output file for test report")
    
    args = parser.parse_args()
    
    result = run_comprehensive_tests(
        update_baseline=args.update_baseline,
        output_file=args.output
    )
    
    # Exit with appropriate code
    if result.failed > 0 or result.errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)