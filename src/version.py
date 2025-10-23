"""Version information for Robotic Car Simulation"""

__version__ = "1.0.0"
__version_info__ = (1, 0, 0)

# Build information
__build__ = "stable"
__build_date__ = "2024-01-01"

# API version for compatibility checking
__api_version__ = "1.0"

def get_version_string():
    """Get formatted version string"""
    return f"{__version__} ({__build__})"

def get_full_version_info():
    """Get complete version information"""
    return {
        "version": __version__,
        "version_info": __version_info__,
        "build": __build__,
        "build_date": __build_date__,
        "api_version": __api_version__
    }