"""
Scoring Operations Manager - Comprehensive Scoring and Metrics Calculation

This module handles all scoring operations, metrics calculation, and scoring-related
functionality, providing centralized scoring management capabilities.

Main Class:
- ScoringOperationsManager: Manages all scoring operations

Key Methods:
- calculate_scores(): Calculates alignment scores using multiple metrics
- set_scoring_method(): Sets current scoring method
- get_scoring_results(): Gets current scoring results
- has_scores(): Checks if scoring results are available
- export_scores(): Exports scoring results to file
- compare_methods(): Compares all scoring methods
- clear_scores(): Clears current scoring results

Private Methods:
- _calculate_all_metrics(): Calculates all scoring metrics
- _normalize_arrays(): Normalizes arrays for comparison
- _calculate_edge_overlap(): Calculates edge overlap ratio
- _calculate_cross_correlation(): Calculates cross-correlation
- _calculate_mutual_information(): Calculates mutual information

Signals Emitted:
- scores_calculated(dict): Scores calculated with results
- scoring_completed(str): Scoring completed with method
- scoring_error(str, str): Scoring error with details

Dependencies:
- Uses: cv2, numpy (image processing and calculations)
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt framework)
- Uses: skimage.metrics (SSIM, MSE calculations)
- Uses: ui/view_manager.ViewMode
- Called by: UI main window and scoring components
- Coordinates with: Alignment operations and image viewers

Features:
- Multiple scoring metrics (SSIM, MSE, Edge Overlap, Cross-Correlation, Mutual Information)
- Array normalization and preprocessing
- Comprehensive error handling and validation
- Export capabilities with detailed information
- Method comparison and analysis
- Integration with alignment workflow
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QObject, Signal
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from src.ui.view_manager import ViewMode


class ScoringOperationsManager(QObject):
    """Manages all scoring operations for the application."""
    
    # Signals for scoring operations
    scores_calculated = Signal(dict)  # scores_dict
    scoring_completed = Signal(str)  # scoring_method
    scoring_error = Signal(str, str)  # operation, error_message
    
    def __init__(self, main_window):
        """Initialize with reference to main window."""
        super().__init__()
        self.main_window = main_window
        
        # Scoring state
        self.current_scoring_results = {}
        self.current_scoring_method = "SSIM"  # Default method
        
    def calculate_scores(self, show_validation_warning=True):
        """Calculate alignment scores using multiple metrics."""
        if not self._validate_required_data(alignment_required=True, show_warning=show_validation_warning):
            return
            
        try:
            # Get current image from image processing service
            if hasattr(self.main_window, 'image_processing_service'):
                current_image = self.main_window.image_processing_service.get_current_image()
            else:
                current_image = self.main_window.current_sem_image
            
            # Convert image to array if needed
            if hasattr(current_image, 'to_array'):
                sem_array = current_image.to_array()
            else:
                sem_array = current_image
                
            # Get transformed GDS from alignment result
            if not hasattr(self.main_window, 'alignment_operations_manager'):
                # Fallback to current alignment result
                if hasattr(self.main_window, 'current_alignment_result'):
                    alignment_result = self.main_window.current_alignment_result
                else:
                    raise RuntimeError("No alignment result available")
            else:
                alignment_result = self.main_window.alignment_operations_manager.current_alignment_result
            
            if alignment_result is None:
                raise RuntimeError("No alignment result available for scoring")
            
            gds_array = alignment_result['transformed_gds']
            
            # Calculate multiple scoring metrics
            scores = self._calculate_all_metrics(sem_array, gds_array)
            
            # Add alignment score from the alignment result
            if 'alignment_score' in alignment_result:
                scores['alignment_score'] = alignment_result['alignment_score']
            
            # Store scores
            self.current_scoring_results = scores
            
            # Store in main window for compatibility
            if hasattr(self.main_window, 'current_scoring_results'):
                self.main_window.current_scoring_results = scores
            
            # Update status
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage("Scores calculated")
            
            # Update score overlays if in scoring view
            if (hasattr(self.main_window, 'view_manager') and 
                self.main_window.view_manager.current_view == ViewMode.SCORING):
                if hasattr(self.main_window, '_update_score_overlays'):
                    self.main_window._update_score_overlays()
            
            # Emit signals
            self.scores_calculated.emit(scores)
            self.scoring_completed.emit(self.current_scoring_method)
            
            print(f"✓ Scores calculated: {scores}")

        except Exception as e:
            self._handle_service_error("Score calculation", e)
    
    def _calculate_all_metrics(self, sem_array, gds_array):
        """Calculate all scoring metrics."""
        scores = {}
        
        try:
            # Ensure arrays are the same size and type
            sem_array, gds_array = self._normalize_arrays(sem_array, gds_array)
            
            # SSIM Score
            try:
                ssim_score = ssim(sem_array, gds_array, data_range=1.0)
                scores['ssim'] = float(ssim_score)
            except Exception as e:
                print(f"Error calculating SSIM: {e}")
                scores['ssim'] = 0.0
            
            # MSE Score
            try:
                mse_score = mse(sem_array, gds_array)
                scores['mse'] = float(mse_score)
            except Exception as e:
                print(f"Error calculating MSE: {e}")
                scores['mse'] = 1.0
            
            # Edge overlap ratio
            try:
                edge_ratio = self._calculate_edge_overlap(sem_array, gds_array)
                scores['edge_overlap_ratio'] = float(edge_ratio)
            except Exception as e:
                print(f"Error calculating edge overlap: {e}")
                scores['edge_overlap_ratio'] = 0.0
            
            # Cross-correlation
            try:
                correlation = self._calculate_cross_correlation(sem_array, gds_array)
                scores['cross_correlation'] = float(correlation)
            except Exception as e:
                print(f"Error calculating cross-correlation: {e}")
                scores['cross_correlation'] = 0.0
            
            # Mutual information
            try:
                mutual_info = self._calculate_mutual_information(sem_array, gds_array)
                scores['mutual_information'] = float(mutual_info)
            except Exception as e:
                print(f"Error calculating mutual information: {e}")
                scores['mutual_information'] = 0.0
                
        except Exception as e:
            print(f"Error in metric calculation: {e}")
            # Return default scores
            scores = {
                'ssim': 0.0,
                'mse': 1.0,
                'edge_overlap_ratio': 0.0,
                'cross_correlation': 0.0,
                'mutual_information': 0.0
            }
        
        return scores
    
    def _normalize_arrays(self, sem_array, gds_array):
        """Normalize arrays to the same size and type."""
        # Convert to grayscale if needed
        if len(sem_array.shape) == 3:
            sem_array = cv2.cvtColor(sem_array, cv2.COLOR_BGR2GRAY)
        if len(gds_array.shape) == 3:
            gds_array = cv2.cvtColor(gds_array, cv2.COLOR_BGR2GRAY)
        
        # Ensure same size
        if sem_array.shape != gds_array.shape:
            # Resize GDS to match SEM
            gds_array = cv2.resize(gds_array, (sem_array.shape[1], sem_array.shape[0]))
        
        # Normalize to 0-1 range
        sem_norm = sem_array.astype(np.float32) / 255.0
        gds_norm = gds_array.astype(np.float32) / 255.0
        
        return sem_norm, gds_norm
    
    def _calculate_edge_overlap(self, sem_array, gds_array):
        """Calculate edge overlap ratio."""
        try:
            # Convert to uint8 for edge detection
            sem_uint8 = (sem_array * 255).astype(np.uint8)
            gds_uint8 = (gds_array * 255).astype(np.uint8)
            
            # Calculate edges
            sem_edges = cv2.Canny(sem_uint8, 50, 150)
            gds_edges = cv2.Canny(gds_uint8, 50, 150)
            
            # Calculate overlap
            edge_overlap = (sem_edges & gds_edges).sum()
            total_edges = sem_edges.sum() + gds_edges.sum()
            
            if total_edges > 0:
                return edge_overlap / total_edges
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating edge overlap: {e}")
            return 0.0
    
    def _calculate_cross_correlation(self, sem_array, gds_array):
        """Calculate normalized cross-correlation."""
        try:
            # Calculate correlation coefficient
            correlation = np.corrcoef(sem_array.flatten(), gds_array.flatten())[0, 1]
            
            # Handle NaN case
            if np.isnan(correlation):
                return 0.0
            
            return correlation
            
        except Exception as e:
            print(f"Error calculating cross-correlation: {e}")
            return 0.0
    
    def _calculate_mutual_information(self, sem_array, gds_array):
        """Calculate mutual information between images."""
        try:
            # Convert to integer histograms
            sem_hist = (sem_array * 255).astype(np.uint8)
            gds_hist = (gds_array * 255).astype(np.uint8)
            
            # Calculate joint histogram
            joint_hist, _, _ = np.histogram2d(sem_hist.flatten(), gds_hist.flatten(), bins=256)
            
            # Normalize to probabilities
            joint_hist = joint_hist / joint_hist.sum()
            
            # Calculate marginal histograms
            sem_marginal = joint_hist.sum(axis=1)
            gds_marginal = joint_hist.sum(axis=0)
            
            # Calculate mutual information
            mutual_info = 0.0
            for i in range(256):
                for j in range(256):
                    if joint_hist[i, j] > 0 and sem_marginal[i] > 0 and gds_marginal[j] > 0:
                        mutual_info += joint_hist[i, j] * np.log2(
                            joint_hist[i, j] / (sem_marginal[i] * gds_marginal[j])
                        )
            
            return mutual_info
            
        except Exception as e:
            print(f"Error calculating mutual information: {e}")
            return 0.0
    
    def set_scoring_method(self, method):
        """Set the current scoring method."""
        available_methods = ['SSIM', 'MSE', 'Edge Overlap', 'Cross-Correlation', 'Mutual Information']
        if method in available_methods:
            self.current_scoring_method = method
            print(f"Scoring method set to: {method}")
        else:
            print(f"Warning: Unknown scoring method: {method}")
    
    def get_scoring_results(self):
        """Get the current scoring results."""
        return self.current_scoring_results.copy()
    
    def has_scores(self):
        """Check if scoring results are available."""
        return bool(self.current_scoring_results)
    
    def export_scores(self, file_path, include_details=True):
        """Export scoring results to a file."""
        try:
            if not self.current_scoring_results:
                raise ValueError("No scoring results to export")
            
            export_data = {
                'scoring_method': self.current_scoring_method,
                'scores': self.current_scoring_results,
                'timestamp': str(np.datetime64('now'))
            }
            
            # Add additional details if requested
            if include_details:
                export_data['image_info'] = self._get_image_info()
                export_data['alignment_info'] = self._get_alignment_info()
            
            # Export as JSON
            import json
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✓ Scoring results exported to: {file_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to export scoring results: {str(e)}"
            print(error_msg)
            self.scoring_error.emit("export_scores", str(e))
            return False
    
    def compare_methods(self):
        """Compare all scoring methods and return a summary."""
        if not self.current_scoring_results:
            return None
        
        try:
            methods = ['ssim', 'mse', 'edge_overlap_ratio', 'cross_correlation', 'mutual_information']
            comparison = {}
            
            for method in methods:
                if method in self.current_scoring_results:
                    comparison[method] = self.current_scoring_results[method]
            
            # Find best and worst scores (higher is better for most metrics except MSE)
            best_scores = {}
            worst_scores = {}
            
            for method, score in comparison.items():
                if method == 'mse':
                    # For MSE, lower is better
                    best_scores[method] = score if score == min(comparison.values()) else None
                    worst_scores[method] = score if score == max(comparison.values()) else None
                else:
                    # For other metrics, higher is better
                    best_scores[method] = score if score == max(comparison.values()) else None
                    worst_scores[method] = score if score == min(comparison.values()) else None
            
            return {
                'scores': comparison,
                'best_method': max(comparison.items(), key=lambda x: x[1] if x[0] != 'mse' else -x[1]),
                'summary': {
                    'total_methods': len(comparison),
                    'average_score': np.mean(list(comparison.values())),
                    'score_range': {
                        'min': min(comparison.values()),
                        'max': max(comparison.values())
                    }
                }
            }
            
        except Exception as e:
            print(f"Error comparing methods: {e}")
            return None
    
    def clear_scores(self):
        """Clear current scoring results."""
        self.current_scoring_results = {}
        if hasattr(self.main_window, 'current_scoring_results'):
            self.main_window.current_scoring_results = {}
        print("Scoring results cleared")
    
    def _get_image_info(self):
        """Get information about current images."""
        info = {}
        
        if hasattr(self.main_window, 'current_sem_image') and self.main_window.current_sem_image is not None:
            img = self.main_window.current_sem_image
            info['sem_image'] = {
                'shape': img.shape,
                'dtype': str(img.dtype),
                'min': float(np.min(img)),
                'max': float(np.max(img))
            }
        
        return info
    
    def _get_alignment_info(self):
        """Get information about current alignment."""
        if hasattr(self.main_window, 'alignment_operations_manager'):
            return self.main_window.alignment_operations_manager.get_alignment_info()
        return None
    
    def _validate_required_data(self, alignment_required=False, show_warning=True):
        """Validate that required data is available for operations."""
        if not hasattr(self.main_window, 'current_sem_image') or self.main_window.current_sem_image is None:
            if show_warning:
                QMessageBox.warning(
                    self.main_window,
                    "No SEM Image",
                    "Please load a SEM image first."
                )
            return False
        
        if alignment_required:
            has_alignment = False
            if hasattr(self.main_window, 'alignment_operations_manager'):
                has_alignment = self.main_window.alignment_operations_manager.is_aligned()
            elif hasattr(self.main_window, 'current_alignment_result'):
                has_alignment = self.main_window.current_alignment_result is not None
            
            if not has_alignment:
                if show_warning:
                    QMessageBox.warning(
                        self.main_window,
                        "No Alignment",
                        "Please perform alignment first."
                    )
                return False
        
        return True
    
    def _handle_service_error(self, operation_name: str, error: Exception):
        """Handle service errors consistently."""
        error_msg = f"{operation_name} failed: {str(error)}"
        print(f"Scoring service error: {error_msg}")
        
        if hasattr(self.main_window, 'status_bar'):
            self.main_window.status_bar.showMessage(error_msg)
        
        QMessageBox.critical(self.main_window, f"{operation_name} Error", error_msg)
        self.scoring_error.emit(operation_name.lower().replace(" ", "_"), str(error))
