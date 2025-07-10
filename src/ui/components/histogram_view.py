"""
Histogram View Component - Matplotlib Embedded Histogram Display

This module provides a matplotlib embedded histogram view component
for image analysis and filter visualization.

Main Class:
- HistogramView: Widget for displaying image histograms

Key Methods:
- setup_ui(): Initializes UI components
- update_histogram(): Updates histogram display with new image data
- clear_displays(): Clears histogram display
- save_plots(): Saves current histogram plot to file
- get_histogram_data(): Gets current histogram data

Signals Emitted:
- histogram_updated(): Histogram display updated

Dependencies:
- Uses: numpy (numerical operations)
- Uses: matplotlib.pyplot, matplotlib.backends.backend_qt5agg (plotting)
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Called by: UI main window and filtering components
- Coordinates with: Image processing and analysis workflows

Features:
- Matplotlib embedded histogram display
- Dark theme styling to match application
- Automatic grayscale conversion for color images
- Responsive sizing with proper layout management
- Histogram data export capabilities
- Real-time updates during image processing
- Grid and axis styling for better visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal
from typing import Optional


class HistogramView(QWidget):
    """Widget for displaying image histograms and filter kernels."""
    
    # Signals
    histogram_updated = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_image = None
        
    def setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for better space usage
        
        # Create matplotlib figure with increased size for better visibility
        self.figure = Figure(figsize=(6, 4))  # Increased from (5, 3)
        self.canvas = FigureCanvas(self.figure)
        
        # Set size policies for responsive behavior
        from PySide6.QtWidgets import QSizePolicy
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy.setHorizontalStretch(1)
        size_policy.setVerticalStretch(1)
        self.canvas.setSizePolicy(size_policy)
        
        layout.addWidget(self.canvas)
        
        # Create single subplot for histogram only
        self.hist_ax = self.figure.add_subplot(1, 1, 1)
        
        # Set dark theme styling to match application
        self.figure.patch.set_facecolor('#2b2b2b')
        self.hist_ax.set_facecolor('#2b2b2b')
        
        # Improve layout spacing for better presentation
        self.figure.tight_layout(pad=1.5)
        
    def update_histogram(self, image_data: np.ndarray) -> None:
        """Update the histogram display with new image data."""
        if image_data is None:
            return
            
        self.current_image = image_data.copy()
        
        # Clear previous plots
        self.hist_ax.clear()
        
        # Convert image to grayscale if it's color
        if len(image_data.shape) == 3:
            # Convert to grayscale
            gray_image = np.dot(image_data[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray_image = image_data
        
        # Ensure data is in 0-255 range
        if gray_image.max() <= 1.0:
            gray_image = gray_image * 255
        
        # Calculate histogram
        hist, bins = np.histogram(gray_image.flatten(), bins=256, range=(0, 255))
        
        # Plot histogram with improved styling
        self.hist_ax.bar(bins[:-1], hist, width=1, alpha=0.8, color='#4a90e2', edgecolor='none')
        self.hist_ax.set_title('SEM Image Histogram', color='white', fontsize=10)
        self.hist_ax.grid(True, alpha=0.3, color='gray')
        
        # Remove axis labels and tick marks
        self.hist_ax.set_xticks([])
        self.hist_ax.set_yticks([])
        self.hist_ax.spines['bottom'].set_color('white')
        self.hist_ax.spines['top'].set_color('white')
        self.hist_ax.spines['right'].set_color('white')
        self.hist_ax.spines['left'].set_color('white')
        
        # Update canvas
        self.canvas.draw()
        self.histogram_updated.emit()
        
    def clear_displays(self) -> None:
        """Clear histogram display."""
        self.hist_ax.clear()
        self.canvas.draw()
        
    def save_plots(self, filepath: str) -> None:
        """Save the current histogram plot to file."""
        self.figure.savefig(filepath, dpi=150, bbox_inches='tight')
        
    def get_histogram_data(self) -> Optional[tuple]:
        """Get the current histogram data."""
        if self.current_image is None:
            return None
            
        # Convert to grayscale if needed
        if len(self.current_image.shape) == 3:
            gray_image = np.dot(self.current_image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray_image = self.current_image
            
        # Ensure data is in 0-255 range
        if gray_image.max() <= 1.0:
            gray_image = gray_image * 255
        
        hist, bins = np.histogram(gray_image.flatten(), bins=256, range=(0, 255))
        return hist, bins
