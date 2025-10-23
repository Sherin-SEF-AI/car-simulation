#!/usr/bin/env python3
"""
Comprehensive Test Runner for Robotic Car Simulation

This script provides a unified interface for running all types of tests:
- Unit tests for individual components
- Integration tests for cross-component communication
- Performance regression tests
- End-to-end scenario tests
"""

import sys
import os
import argparse
import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import test framework
from tests.test_framework import TestFramework, run_comprehensive_tests


@dataclass
class TestRunResult:
    """Result of a test run"""
    test_type: str
    success: bool
    total_tests: int
    passed: int
    failed: int
    errors: int
    execution_time: float
    output: str
    error_details: List[str]


class TestRunner:
    """Comprehensive test runner for the simulation system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results: List[TestRunResult] = []
        
    def run_unit_tests(self, pattern: str = "test_*.py") -> TestRunResult:
        """Run unit tests for individual components"""
        print("Running unit tests...")
        
        start_time = time.time()
        
        # Discover and run unit tests
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            f"--pattern={pattern}",
            "--json-report",
            "--json-report-file=test_results_unit.json"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse pytest JSON output if available
            json_file = self.project_root / "test_results_unit.json"
            if json_file.exists():
                with open(json_file) as f:
                    pytest_data = json.load(f)
                    
                total_tests = pytest_data.get('summary', {}).get('total', 0)
                passed = pytest_data.get('summary', {}).get('passed', 0)
                failed = pytest_data.get('summary', {}).get('failed', 0)
                errors = pytest_data.get('summary', {}).get('error', 0)
            else:
                # Fallback parsing
                total_tests = result.stdout.count('PASSED') + result.stdout.count('FAILED')
                passed = result.stdout.count('PASSED')
                failed = result.stdout.count('FAILED')
                errors = result.stdout.count('ERROR')
            
            success = result.returncode == 0
            error_details = []
            
            if not success:
                error_details = [
                    line for line in result.stdout.split('\n') 
                    if 'FAILED' in line or 'ERROR' in line
                ]
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            success = False
            total_tests = 0
            passed = 0
            failed = 0
            errors = 1
            error_details = ["Unit tests timed out after 5 minutes"]
            result = type('Result', (), {'stdout': '', 'stderr': 'Timeout'})()
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            total_tests = 0
            passed = 0
            failed = 0
            errors = 1
            error_details = [f"Unit test execution failed: {str(e)}"]
            result = type('Result', (), {'stdout': '', 'stderr': str(e)})()
        
        test_result = TestRunResult(
            test_type="unit",
            success=success,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            errors=errors,
            execution_time=execution_time,
            output=result.stdout,
            error_details=error_details
        )
        
        self.test_results.append(test_result)
        return test_result
        
    def run_integration_tests(self) -> TestRunResult:
        """Run integration tests"""
        print("Running integration tests...")
        
        start_time = time.time()
        
        cmd = [
            sys.executable, "-m", "unittest", 
            "tests.test_integration", 
            "-v"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for integration tests
            )
            
            execution_time = time.time() - start_time
            
            # Parse unittest output
            lines = result.stderr.split('\n')  # unittest outputs to stderr
            test_lines = [line for line in lines if ' ... ' in line]
            
            total_tests = len(test_lines)
            passed = len([line for line in test_lines if 'ok' in line])
            failed = len([line for line in test_lines if 'FAIL' in line])
            errors = len([line for line in test_lines if 'ERROR' in line])
            
            success = result.returncode == 0
            error_details = []
            
            if not success:
                error_details = [
                    line for line in lines 
                    if 'FAIL:' in line or 'ERROR:' in line
                ]
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            success = False
            total_tests = 0
            passed = 0
            failed = 0
            errors = 1
            error_details = ["Integration tests timed out after 10 minutes"]
            result = type('Result', (), {'stdout': '', 'stderr': 'Timeout'})()
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            total_tests = 0
            passed = 0
            failed = 0
            errors = 1
            error_details = [f"Integration test execution failed: {str(e)}"]
            result = type('Result', (), {'stdout': '', 'stderr': str(e)})()
        
        test_result = TestRunResult(
            test_type="integration",
            success=success,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            errors=errors,
            execution_time=execution_time,
            output=result.stderr,  # unittest uses stderr
            error_details=error_details
        )
        
        self.test_results.append(test_result)
        return test_result
        
    def run_performance_tests(self, update_baselines: bool = False) -> TestRunResult:
        """Run performance regression tests"""
        print("Running performance regression tests...")
        
        start_time = time.time()
        
        cmd = [
            sys.executable, "-m", "unittest", 
            "tests.test_performance_regression", 
            "-v"
        ]
        
        if update_baselines:
            cmd.append("--update-baselines")
            
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout for performance tests
            )
            
            execution_time = time.time() - start_time
            
            # Parse unittest output
            lines = result.stderr.split('\n')
            test_lines = [line for line in lines if ' ... ' in line]
            
            total_tests = len(test_lines)
            passed = len([line for line in test_lines if 'ok' in line])
            failed = len([line for line in test_lines if 'FAIL' in line])
            errors = len([line for line in test_lines if 'ERROR' in line])
            
            success = result.returncode == 0
            error_details = []
            
            if not success:
                error_details = [
                    line for line in lines 
                    if 'regression' in line.lower() or 'FAIL:' in line
                ]
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            success = False
            total_tests = 0
            passed = 0
            failed = 0
            errors = 1
            error_details = ["Performance tests timed out after 15 minutes"]
            result = type('Result', (), {'stdout': '', 'stderr': 'Timeout'})()
            
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            total_tests = 0
            passed = 0
            failed = 0
            errors = 1
            error_details = [f"Performance test execution failed: {str(e)}"]
            result = type('Result', (), {'stdout': '', 'stderr': str(e)})()
        
        test_result = TestRunResult(
            test_type="performance",
            success=success,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            errors=errors,
            execution_time=execution_time,
            output=result.stderr,
            error_details=error_details
        )
        
        self.test_results.append(test_result)
        return test_result
        
    def run_comprehensive_framework_tests(self, update_baseline: bool = False) -> TestRunResult:
        """Run tests using the comprehensive test framework"""
        print("Running comprehensive test framework...")
        
        start_time = time.time()
        
        try:
            # Use the test framework directly
            suite_result = run_comprehensive_tests(
                update_baseline=update_baseline,
                output_file="comprehensive_test_report.txt"
            )
            
            execution_time = time.time() - start_time
            
            success = suite_result.failed == 0 and suite_result.errors == 0
            
            error_details = []
            if not success:
                for test_result in suite_result.test_results:
                    if test_result.status in ['failed', 'error']:
                        error_details.append(
                            f"{test_result.test_name}: {test_result.error_message}")
            
            test_result = TestRunResult(
                test_type="comprehensive",
                success=success,
                total_tests=suite_result.total_tests,
                passed=suite_result.passed,
                failed=suite_result.failed,
                errors=suite_result.errors,
                execution_time=execution_time,
                output=f"Total: {suite_result.total_tests}, "
                      f"Passed: {suite_result.passed}, "
                      f"Failed: {suite_result.failed}",
                error_details=error_details
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            test_result = TestRunResult(
                test_type="comprehensive",
                success=False,
                total_tests=0,
                passed=0,
                failed=0,
                errors=1,
                execution_time=execution_time,
                output="",
                error_details=[f"Comprehensive test framework failed: {str(e)}"]
            )
        
        self.test_results.append(test_result)
        return test_result
        
    def run_all_tests(self, update_baselines: bool = False, 
                     test_types: List[str] = None) -> Dict[str, TestRunResult]:
        """Run all specified test types"""
        if test_types is None:
            test_types = ["unit", "integration", "performance", "comprehensive"]
            
        results = {}
        
        print("=" * 80)
        print("ROBOTIC CAR SIMULATION - COMPREHENSIVE TEST EXECUTION")
        print("=" * 80)
        print(f"Starting test execution at {datetime.now()}")
        print(f"Test types: {', '.join(test_types)}")
        print()
        
        if "unit" in test_types:
            results["unit"] = self.run_unit_tests()
            
        if "integration" in test_types:
            results["integration"] = self.run_integration_tests()
            
        if "performance" in test_types:
            results["performance"] = self.run_performance_tests(update_baselines)
            
        if "comprehensive" in test_types:
            results["comprehensive"] = self.run_comprehensive_framework_tests(update_baselines)
            
        return results
        
    def generate_summary_report(self, output_file: str = None) -> str:
        """Generate a summary report of all test results"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ROBOTIC CAR SIMULATION - TEST EXECUTION SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append(f"Report generated: {datetime.now()}")
        report_lines.append("")
        
        total_tests = sum(r.total_tests for r in self.test_results)
        total_passed = sum(r.passed for r in self.test_results)
        total_failed = sum(r.failed for r in self.test_results)
        total_errors = sum(r.errors for r in self.test_results)
        total_time = sum(r.execution_time for r in self.test_results)
        
        report_lines.append("OVERALL SUMMARY:")
        report_lines.append(f"  Total Tests: {total_tests}")
        report_lines.append(f"  Passed: {total_passed}")
        report_lines.append(f"  Failed: {total_failed}")
        report_lines.append(f"  Errors: {total_errors}")
        report_lines.append(f"  Success Rate: {(total_passed / total_tests * 100):.1f}%" if total_tests > 0 else "  Success Rate: N/A")
        report_lines.append(f"  Total Execution Time: {total_time:.2f}s")
        report_lines.append("")
        
        # Individual test type results
        for result in self.test_results:
            report_lines.append(f"{result.test_type.upper()} TESTS:")
            report_lines.append(f"  Status: {'PASSED' if result.success else 'FAILED'}")
            report_lines.append(f"  Tests: {result.total_tests}")
            report_lines.append(f"  Passed: {result.passed}")
            report_lines.append(f"  Failed: {result.failed}")
            report_lines.append(f"  Errors: {result.errors}")
            report_lines.append(f"  Time: {result.execution_time:.2f}s")
            
            if result.error_details:
                report_lines.append("  Error Details:")
                for error in result.error_details[:5]:  # Limit to first 5 errors
                    report_lines.append(f"    - {error}")
                if len(result.error_details) > 5:
                    report_lines.append(f"    ... and {len(result.error_details) - 5} more errors")
                    
            report_lines.append("")
            
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        if total_failed > 0 or total_errors > 0:
            report_lines.append("  - Review failed tests and fix underlying issues")
            report_lines.append("  - Check error logs for detailed failure information")
            
        performance_result = next((r for r in self.test_results if r.test_type == "performance"), None)
        if performance_result and not performance_result.success:
            report_lines.append("  - Performance regressions detected - review recent changes")
            
        if all(r.success for r in self.test_results):
            report_lines.append("  - All tests passed! System is ready for deployment")
            
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
                
        return report_text


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for Robotic Car Simulation"
    )
    
    parser.add_argument(
        "--test-types", 
        nargs="+", 
        choices=["unit", "integration", "performance", "comprehensive", "all"],
        default=["all"],
        help="Types of tests to run"
    )
    
    parser.add_argument(
        "--update-baselines", 
        action="store_true",
        help="Update performance baselines"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file for test report"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle "all" test type
    if "all" in args.test_types:
        test_types = ["unit", "integration", "performance", "comprehensive"]
    else:
        test_types = args.test_types
        
    # Create and run tests
    runner = TestRunner()
    results = runner.run_all_tests(
        update_baselines=args.update_baselines,
        test_types=test_types
    )
    
    # Generate report
    report = runner.generate_summary_report(args.output)
    
    if args.verbose or not args.output:
        print(report)
        
    # Exit with appropriate code
    all_passed = all(result.success for result in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()