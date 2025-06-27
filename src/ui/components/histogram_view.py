"""
Histogram View Component
Matplotlib embedded histogram and kernel view for filter analysis.
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
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Create subplots
        self.hist_ax = self.figure.add_subplot(2, 1, 1)
        self.kernel_ax = self.figure.add_subplot(2, 1, 2)
        
        self.figure.tight_layout()
        
    def update_histogram(self, image_data: np.ndarray) -> None:
        """Update the histogram display with new image data."""
        if image_data is None:
            return
            
        self.current_image = image_data.copy()
        
        # Clear previous plots
        self.hist_ax.clear()
        
        # Calculate histogram
        hist, bins = np.histogram(image_data.flatten(), bins=256, range=(0, 255))
        
        # Plot histogram
        self.hist_ax.bar(bins[:-1], hist, width=1, alpha=0.7, color='blue')
        self.hist_ax.set_title('Image Histogram')
        self.hist_ax.set_xlabel('Pixel Intensity')
        self.hist_ax.set_ylabel('Frequency')
        self.hist_ax.grid(True, alpha=0.3)
        
        # Update canvas
        self.canvas.draw()
        self.histogram_updated.emit()
        
    def update_kernel_view(self, kernel: np.ndarray, title: str = "Filter Kernel") -> None:
        """Update the kernel visualization."""
        if kernel is None:
            return
            
        # Clear previous plot
        self.kernel_ax.clear()
        
        # Display kernel as image
        im = self.kernel_ax.imshow(kernel, cmap='RdBu_r', aspect='equal')
        self.kernel_ax.set_title(title)
        
        # Add colorbar if kernel is large enough
        if kernel.shape[0] > 3 or kernel.shape[1] > 3:
            self.figure.colorbar(im, ax=self.kernel_ax, shrink=0.6)
        
        # Add text annotations for small kernels
        if kernel.shape[0] <= 5 and kernel.shape[1] <= 5:
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    text = self.kernel_ax.text(j, i, f'{kernel[i, j]:.2f}',
                                             ha="center", va="center", color="black", fontsize=8)
        
        # Update canvas
        self.canvas.draw()
        
    def clear_displays(self) -> None:
        """Clear both histogram and kernel displays."""
        self.hist_ax.clear()
        self.kernel_ax.clear()
        self.canvas.draw()
        
    def save_plots(self, filepath: str) -> None:
        """Save the current plots to file."""
        self.figure.savefig(filepath, dpi=150, bbox_inches='tight')
        
    def get_histogram_data(self) -> Optional[tuple]:
        """Get the current histogram data."""
        if self.current_image is None:
            return None
            
        hist, bins = np.histogram(self.current_image.flatten(), bins=256, range=(0, 255))
        return hist, bins
