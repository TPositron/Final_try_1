"""
Main source package for the Image Analysis application.

This is the root package that organizes the entire SEM/GDS alignment application into three main modules:
- core: Data models, utilities, and core business logic
- services: Business logic services that coordinate between UI and core
- ui: User interface components and controllers

The package follows a layered architecture:
UI Layer -> Services Layer -> Core Layer

Dependencies:
- Used by: main.py (application entry point)
- Uses: core, services, ui subpackages
"""

from . import core
from . import services
from . import ui

__all__ = ['core', 'services', 'ui']
