"""
Monitoring package for the trading application
"""
from .core.monitoring import monitoring_service

# Export the monitoring service for easy access
__all__ = ['monitoring_service']