# Makefile for Robotic Car Simulation Testing

.PHONY: help test test-unit test-integration test-performance test-all clean install lint format coverage docs

# Default target
help:
	@echo "Robotic Car Simulation - Test Management"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@echo "  install           Install dependencies"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-performance  Run performance tests only"
	@echo "  test-comprehensive Run comprehensive test framework"
	@echo "  test-fast         Run fast tests (unit + integration)"
	@echo "  lint              Run code linting"
	@echo "  format            Format code with black"
	@echo "  coverage          Generate coverage report"
	@echo "  clean             Clean test artifacts"
	@echo "  docs              Generate documentation"
	@echo "  baseline-update   Update performance baselines"
	@echo ""

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -r tests/requirements.txt

# Test execution
test: test-all

test-all:
	@echo "Running all tests..."
	python run_tests.py --test-types all --output test_report.txt

test-unit:
	@echo "Running unit tests..."
	python run_tests.py --test-types unit --output unit_test_report.txt

test-integration:
	@echo "Running integration tests..."
	python run_tests.py --test-types integration --output integration_test_report.txt

test-performance:
	@echo "Running performance tests..."
	python run_tests.py --test-types performance --output performance_test_report.txt

test-comprehensive:
	@echo "Running comprehensive test framework..."
	python run_tests.py --test-types comprehensive --output comprehensive_test_report.txt

test-fast:
	@echo "Running fast tests (unit + integration)..."
	python run_tests.py --test-types unit integration --output fast_test_report.txt

# Performance baseline management
baseline-update:
	@echo "Updating performance baselines..."
	python run_tests.py --test-types performance --update-baselines

# Code quality
lint:
	@echo "Running code linting..."
	flake8 src tests --max-line-length=127 --extend-ignore=E203,W503
	mypy src --ignore-missing-imports --no-strict-optional

format:
	@echo "Formatting code..."
	black src tests --line-length=127
	isort src tests --profile black

# Coverage
coverage:
	@echo "Generating coverage report..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# Documentation
docs:
	@echo "Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html
	@echo "Documentation generated in docs/_build/html/"

# Cleanup
clean:
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf test-results.xml
	rm -rf test-results.json
	rm -rf coverage.xml
	rm -rf *_test_report.txt
	rm -rf test_report.txt
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Development helpers
dev-setup: install
	@echo "Setting up development environment..."
	pre-commit install

# Continuous Integration simulation
ci-test:
	@echo "Running CI test simulation..."
	make lint
	make test-fast
	make coverage

# Stress testing
stress-test:
	@echo "Running stress tests..."
	python -m pytest tests/test_performance_regression.py::TestSystemIntegrationPerformance::test_stress_test_performance -v

# Memory leak detection
memory-test:
	@echo "Running memory leak detection..."
	python -m pytest tests/test_performance_regression.py::TestMemoryLeakDetection -v

# Benchmark specific components
benchmark-physics:
	@echo "Benchmarking physics engine..."
	python -m pytest tests/test_performance_regression.py::TestPhysicsPerformance -v

benchmark-ai:
	@echo "Benchmarking AI system..."
	python -m pytest tests/test_performance_regression.py::TestAIPerformance -v

benchmark-rendering:
	@echo "Benchmarking rendering system..."
	python -m pytest tests/test_performance_regression.py::TestRenderingPerformance -v

# Test specific scenarios
test-multi-vehicle:
	@echo "Testing multi-vehicle scenarios..."
	python -m pytest tests/test_integration.py::TestMultiVehicleIntegration -v

test-ui-integration:
	@echo "Testing UI integration..."
	python -m pytest tests/test_integration.py::TestUIIntegration -v

# Performance monitoring
monitor-performance:
	@echo "Monitoring performance during test execution..."
	python -c "
import subprocess
import psutil
import time
import threading

def monitor():
    process = psutil.Process()
    while True:
        cpu = process.cpu_percent()
        memory = process.memory_info().rss / 1024 / 1024
        print(f'CPU: {cpu:.1f}%, Memory: {memory:.1f}MB')
        time.sleep(1)

monitor_thread = threading.Thread(target=monitor, daemon=True)
monitor_thread.start()

subprocess.run(['python', 'run_tests.py', '--test-types', 'performance'])
"

# Generate test matrix for different Python versions
test-matrix:
	@echo "Testing with different Python versions..."
	@for version in 3.9 3.10 3.11; do \
		echo "Testing with Python $$version..."; \
		python$$version -m venv venv_$$version; \
		source venv_$$version/bin/activate; \
		pip install -r requirements.txt; \
		pip install -r tests/requirements.txt; \
		python run_tests.py --test-types unit; \
		deactivate; \
	done

# Quick smoke test
smoke-test:
	@echo "Running smoke test..."
	python -c "
import sys
sys.path.insert(0, 'src')
from core.application import SimulationApplication
app = SimulationApplication()
print('✓ Application imports successfully')
app.cleanup()
print('✓ Application cleanup successful')
print('✓ Smoke test passed')
"