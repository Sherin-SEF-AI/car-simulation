"""
Setup script for Robotic Car Simulation

Provides packaging and distribution configuration for the simulation application.
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Read version from version file
version_file = Path(__file__).parent / "src" / "version.py"
version_info = {}
if version_file.exists():
    exec(version_file.read_text(), version_info)
    VERSION = version_info.get('__version__', '1.0.0')
else:
    VERSION = '1.0.0'

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    LONG_DESCRIPTION = readme_file.read_text(encoding='utf-8')
else:
    LONG_DESCRIPTION = "Robotic Car Simulation - Advanced autonomous vehicle simulation platform"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    REQUIREMENTS = [
        line.strip() for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]
else:
    REQUIREMENTS = [
        'PyQt6>=6.4.0',
        'numpy>=1.21.0',
        'psutil>=5.9.0',
    ]

# Development requirements
DEV_REQUIREMENTS = [
    'pytest>=7.0.0',
    'pytest-qt>=4.0.0',
    'pytest-cov>=4.0.0',
    'black>=22.0.0',
    'flake8>=5.0.0',
    'mypy>=0.991',
    'sphinx>=5.0.0',
    'sphinx-rtd-theme>=1.0.0',
]

# Platform-specific requirements
PLATFORM_REQUIREMENTS = {
    'win32': [
        'pywin32>=304',
    ],
    'darwin': [
        'pyobjc-framework-Cocoa>=8.0',
    ],
    'linux': [
        'python-xlib>=0.31',
    ]
}

# Add platform-specific requirements
current_platform = sys.platform
if current_platform in PLATFORM_REQUIREMENTS:
    REQUIREMENTS.extend(PLATFORM_REQUIREMENTS[current_platform])

setup(
    name="robotic-car-simulation",
    version=VERSION,
    author="Robotic Car Simulation Team",
    author_email="team@roboticarsim.com",
    description="Advanced autonomous vehicle simulation platform",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/roboticarsim/robotic-car-simulation",
    project_urls={
        "Bug Tracker": "https://github.com/roboticarsim/robotic-car-simulation/issues",
        "Documentation": "https://roboticarsim.readthedocs.io/",
        "Source Code": "https://github.com/roboticarsim/robotic-car-simulation",
    },
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
        "ui": ["*.ui", "*.qrc"],
        "assets": ["*.png", "*.jpg", "*.svg", "*.obj", "*.mtl"],
        "shaders": ["*.vert", "*.frag", "*.glsl"],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "robotic-car-sim=main:main",
            "robosim=main:main",
            "robosim-test=run_tests:main",
        ],
        "gui_scripts": [
            "robotic-car-sim-gui=main:main",
        ],
    },
    
    # Dependencies
    install_requires=REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "test": [
            'pytest>=7.0.0',
            'pytest-qt>=4.0.0',
            'pytest-cov>=4.0.0',
            'pytest-benchmark>=4.0.0',
        ],
        "docs": [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.19.0',
        ],
        "performance": [
            'py-spy>=0.3.14',
            'memory-profiler>=0.60.0',
            'line-profiler>=4.0.0',
        ],
    },
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: X11 Applications :: Qt",
    ],
    
    # Keywords
    keywords=[
        "autonomous vehicles", "simulation", "robotics", "ai", "machine learning",
        "physics simulation", "3d graphics", "pyqt", "opengl", "computer vision"
    ],
    
    # License
    license="MIT",
    
    # Zip safety
    zip_safe=False,
)