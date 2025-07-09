"""
Scoring Calculator - Comprehensive Metrics Calculation and Evaluation

This module handles all scoring operations, metrics calculation, and evaluation
functionality for SEM/GDS alignment assessment.

Main Class:
- ScoringCalculator: Qt-based handler for scoring operations

Key Methods:
- calculate_scores(): Calculates comprehensive alignment scores
- batch_scoring(): Performs batch scoring on multiple image pairs
- set_scoring_method(): Sets primary scoring method
- get_current_scores(): Returns current scoring results
- export_scores(): Exports scoring results to file
- compare_scoring_methods(): Compares all scoring methods
- get_best_score(): Returns best score from current results

Signals Emitted:
- scores_calculated(dict): Scoring results calculated
- batch_scoring_completed(list): Batch scoring completed
- scoring_progress(str): Progress message during operations

Dependencies:
- Uses: cv2, numpy (image processing and calculations)
- Uses: PySide6.QtWidgets, PySide6.QtCore (Qt integration)
- Uses: skimage.metrics (SSIM, MSE calculations)
- Uses: services/simple_scoring_service.ScoringService
- Called by: ui/main_window.py (scoring operations)
- Coordinates with: UI scoring components

Scoring Methods:
- SSIM: Structural Similarity Index
- MSE: Mean Squared Error
- PSNR: Peak Signal-to-Noise Ratio
- Cross-Correlation: Template matching correlation
- Mutual Information: Information theory metric
- Edge Overlap: Edge detection overlap analysis

Features:
- Comprehensive multi-metric scoring
- Batch processing capabilities
- Scoring history tracking
- Export functionality with context information
- Method comparison and analysis
- Error handling with user feedback
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QObject, Signal
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from src.services.simple_scoring_service import ScoringService


class ScoringCalculator(QObject):
    """Handles scoring operations and metrics calculation."""
    
    # Signals
    scores_calculated = Signal(dict)  # scoring results
    batch_scoring_completed = Signal(list)  # batch results
    scoring_progress = Signal(str)  # progress message
    
    def __init__(self, main_window):
        """Initialize scoring calculator with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.scoring_service = ScoringService()
        
        # Scoring state
        self.current_scoring_method = "SSIM"
        self.current_scoring_results = {}
        self.scoring_history = []
        
        # Available scoring methods
        self.available_methods = [
            "SSIM", "MSE", "PSNR", "Cross-Correlation", 
            "Mutual Information", "Edge Overlap"
        ]
    
    def calculate_scores(self):
        """Calculate alignment scores using multiple methods."""
        try:
            # Validate prerequisites
            if not self._validate_scoring_prerequisites():
                return False
            
            print("Calculating alignment scores...")
            self.scoring_progress.emit("Calculating alignment scores...")
            
            # Get current images
            sem_image = self._get_current_sem_image()
            aligned_overlay = self._get_aligned_overlay()
            
            if sem_image is None or aligned_overlay is None:
                raise ValueError("Required images not available for scoring")
            
            # Calculate comprehensive scores
            scores = self._calculate_comprehensive_scores(sem_image, aligned_overlay)
            
            if scores:
                # Store results
                self.current_scoring_results = scores
                
                # Add to history
                self.scoring_history.append({
                    'scores': scores.copy(),
                    'timestamp': str(np.datetime64('now')),
                    'method': 'comprehensive'
                })
                
                # Update score overlays
                self._update_score_overlays(scores)
                
                # Update status
                primary_score = scores.get(self.current_scoring_method, 'N/A')
                if hasattr(self.main_window, 'status_bar'):
                    self.main_window.status_bar.showMessage(
                        f"Scoring completed - {self.current_scoring_method}: {primary_score}"
                    )
                
                # Emit signal
                self.scores_calculated.emit(scores)
                
                print(f"✓ Scoring completed: {scores}")
                return True
            else:
                raise RuntimeError("Score calculation returned empty results")
                
        except Exception as e:
            error_msg = f"Failed to calculate scores: {str(e)}"
            print(f"Scoring error: {error_msg}")
            QMessageBox.critical(self.main_window, "Scoring Error", error_msg)
            return False
    
    def _calculate_comprehensive_scores(self, sem_image, gds_overlay):
        """Calculate comprehensive scoring metrics."""
        try:
            scores = {}
            
            # Ensure images are in compatible format
            sem_array = self._prepare_image_for_scoring(sem_image)
            gds_array = self._prepare_image_for_scoring(gds_overlay)
            
            # Ensure same size
            if sem_array.shape != gds_array.shape:
                gds_array = cv2.resize(gds_array, (sem_array.shape[1], sem_array.shape[0]))
            
            # Calculate SSIM
            try:
                ssim_score = ssim(sem_array, gds_array, data_range=1.0)
                scores['SSIM'] = round(float(ssim_score), 4)
            except Exception as e:
                print(f"SSIM calculation failed: {e}")
                scores['SSIM'] = 0.0
            
            # Calculate MSE
            try:
                mse_score = mse(sem_array, gds_array)
                scores['MSE'] = round(float(mse_score), 4)
            except Exception as e:
                print(f"MSE calculation failed: {e}")
                scores['MSE'] = 1.0
            
            # Calculate PSNR
            try:
                if scores.get('MSE', 1.0) > 0:
                    psnr_score = 20 * np.log10(1.0 / np.sqrt(scores['MSE']))
                    scores['PSNR'] = round(float(psnr_score), 4)
                else:
                    scores['PSNR'] = 100.0  # Perfect match
            except Exception as e:
                print(f"PSNR calculation failed: {e}")
                scores['PSNR'] = 0.0
            
            # Calculate Cross-Correlation
            try:
                correlation = cv2.matchTemplate(sem_array, gds_array, cv2.TM_CCOEFF_NORMED)
                scores['Cross-Correlation'] = round(float(np.max(correlation)), 4)
            except Exception as e:
                print(f"Cross-correlation calculation failed: {e}")
                scores['Cross-Correlation'] = 0.0
            
            # Calculate Edge Overlap
            try:
                edge_score = self._calculate_edge_overlap(sem_array, gds_array)
                scores['Edge Overlap'] = round(float(edge_score), 4)
            except Exception as e:
                print(f"Edge overlap calculation failed: {e}")
                scores['Edge Overlap'] = 0.0
            
            # Calculate Mutual Information (simplified)
            try:
                mi_score = self._calculate_mutual_information(sem_array, gds_array)
                scores['Mutual Information'] = round(float(mi_score), 4)
            except Exception as e:
                print(f"Mutual information calculation failed: {e}")
                scores['Mutual Information'] = 0.0
            
            return scores
            
        except Exception as e:
            print(f"Error in comprehensive scoring: {e}")
            return {}
    
    def _calculate_edge_overlap(self, sem_array, gds_array):
        """Calculate edge overlap score."""
        try:
            # Convert to uint8 for edge detection
            sem_uint8 = (sem_array * 255).astype(np.uint8)
            gds_uint8 = (gds_array * 255).astype(np.uint8)
            
            # Apply edge detection
            sem_edges = cv2.Canny(sem_uint8, 50, 150)
            gds_edges = cv2.Canny(gds_uint8, 50, 150)
            
            # Calculate overlap
            edge_overlap = np.sum(sem_edges & gds_edges)
            total_edges = np.sum(sem_edges) + np.sum(gds_edges)
            
            if total_edges > 0:
                return edge_overlap / total_edges
            else:
                return 0.0
                
        except Exception as e:
            print(f"Edge overlap calculation error: {e}")
            return 0.0
    
    def _calculate_mutual_information(self, sem_array, gds_array):
        """Calculate simplified mutual information."""
        try:
            # Convert to integer bins for histogram
            sem_int = (sem_array * 255).astype(np.uint8)
            gds_int = (gds_array * 255).astype(np.uint8)
            
            # Calculate joint histogram
            hist_2d, _, _ = np.histogram2d(sem_int.ravel(), gds_int.ravel(), bins=50)
            
            # Calculate probabilities
            hist_2d_norm = hist_2d / np.sum(hist_2d)
            hist_sem = np.sum(hist_2d_norm, axis=1)
            hist_gds = np.sum(hist_2d_norm, axis=0)
            
            # Calculate mutual information
            mi = 0.0
            for i in range(len(hist_sem)):
                for j in range(len(hist_gds)):
                    if hist_2d_norm[i, j] > 0 and hist_sem[i] > 0 and hist_gds[j] > 0:
                        mi += hist_2d_norm[i, j] * np.log2(
                            hist_2d_norm[i, j] / (hist_sem[i] * hist_gds[j])
                        )
            
            return mi
            
        except Exception as e:
            print(f"Mutual information calculation error: {e}")
            return 0.0
    
    def _prepare_image_for_scoring(self, image):
        """Prepare image for scoring calculations."""
        try:
            # Handle different image types
            if hasattr(image, 'to_array'):
                array = image.to_array()
            else:
                array = image
            
            # Ensure it's a numpy array
            if not isinstance(array, np.ndarray):
                array = np.array(array)
            
            # Normalize to 0-1 range
            if array.dtype == np.uint8:
                array = array.astype(np.float32) / 255.0
            elif array.dtype in [np.uint16, np.uint32]:
                array = array.astype(np.float32) / np.max(array)
            elif array.max() > 1.0:
                array = array / np.max(array)
            
            # Ensure single channel
            if len(array.shape) > 2:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            
            return array.astype(np.float32)
            
        except Exception as e:
            print(f"Error preparing image for scoring: {e}")
            return None
    
    def _update_score_overlays(self, scores):
        """Update UI overlays with scoring information."""
        try:
            # Update scoring panel if available
            if hasattr(self.main_window, 'panel_manager'):
                scoring_panel = self.main_window.panel_manager.left_panels.get('SCORING')
                if scoring_panel and hasattr(scoring_panel, 'update_scores'):
                    scoring_panel.update_scores(scores)
            
        except Exception as e:
            print(f"Error updating score overlays: {e}")
    
    def batch_scoring(self, image_pairs, methods=None):
        """Perform batch scoring on multiple image pairs."""
        try:
            if not image_pairs:
                raise ValueError("No image pairs provided for batch scoring")
            
            if methods is None:
                methods = self.available_methods
            
            print(f"Starting batch scoring for {len(image_pairs)} image pairs...")
            self.scoring_progress.emit(f"Starting batch scoring for {len(image_pairs)} pairs...")
            
            batch_results = []
            
            for i, (sem_img, gds_overlay) in enumerate(image_pairs):
                try:
                    progress_msg = f"Processing pair {i+1}/{len(image_pairs)}"
                    print(progress_msg)
                    self.scoring_progress.emit(progress_msg)
                    
                    # Calculate scores for this pair
                    pair_scores = self._calculate_comprehensive_scores(sem_img, gds_overlay)
                    
                    batch_results.append({
                        'pair_index': i,
                        'scores': pair_scores,
                        'success': True
                    })
                    
                except Exception as e:
                    print(f"Error processing pair {i+1}: {e}")
                    batch_results.append({
                        'pair_index': i,
                        'scores': {},
                        'success': False,
                        'error': str(e)
                    })
            
            # Emit signal
            self.batch_scoring_completed.emit(batch_results)
            
            successful_pairs = sum(1 for result in batch_results if result['success'])
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage(
                    f"Batch scoring completed: {successful_pairs}/{len(image_pairs)} pairs processed"
                )
            
            print(f"✓ Batch scoring completed: {successful_pairs}/{len(image_pairs)} successful")
            return batch_results
            
        except Exception as e:
            error_msg = f"Batch scoring failed: {str(e)}"
            print(f"Batch scoring error: {error_msg}")
            QMessageBox.critical(self.main_window, "Batch Scoring Error", error_msg)
            return []
    
    def set_scoring_method(self, method):
        """Set the primary scoring method."""
        if method in self.available_methods:
            self.current_scoring_method = method
            print(f"Primary scoring method set to: {method}")
        else:
            print(f"Warning: Unknown scoring method: {method}")
    
    def get_scoring_method(self):
        """Get the current primary scoring method."""
        return self.current_scoring_method
    
    def get_available_methods(self):
        """Get list of available scoring methods."""
        return self.available_methods.copy()
    
    def get_current_scores(self):
        """Get the current scoring results."""
        return self.current_scoring_results.copy()
    
    def get_scoring_history(self):
        """Get the scoring history."""
        return self.scoring_history.copy()
    
    def export_scores(self, file_path, include_history=True):
        """Export scoring results to a file."""
        try:
            if not self.current_scoring_results:
                raise ValueError("No scoring results to export")
            
            export_data = {
                'primary_method': self.current_scoring_method,
                'current_scores': self.current_scoring_results,
                'timestamp': str(np.datetime64('now')),
                'available_methods': self.available_methods
            }
            
            if include_history:
                export_data['scoring_history'] = self.scoring_history
            
            # Add context information
            export_data['context'] = {
                'sem_image_info': self._get_sem_image_info(),
                'gds_info': self._get_gds_info(),
                'alignment_info': self._get_alignment_info()
            }
            
            # Save as JSON
            import json
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✓ Scoring results exported to: {file_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to export scores: {str(e)}"
            print(f"Export error: {error_msg}")
            QMessageBox.critical(self.main_window, "Export Error", error_msg)
            return False
    
    def compare_scoring_methods(self):
        """Compare all scoring methods and return a comparison report."""
        try:
            if not self.current_scoring_results:
                self.calculate_scores()
            
            if not self.current_scoring_results:
                return None
            
            scores = self.current_scoring_results
            
            # Filter numerical scores for comparison
            numerical_scores = {
                method: score for method, score in scores.items()
                if isinstance(score, (int, float)) and not np.isnan(score)
            }
            
            if not numerical_scores:
                return None
            
            # Create comparison report
            comparison = {
                'methods_compared': list(numerical_scores.keys()),
                'scores': numerical_scores,
                'statistics': {
                    'best_score': max(numerical_scores.items(), key=lambda x: x[1]),
                    'worst_score': min(numerical_scores.items(), key=lambda x: x[1]),
                    'average_score': sum(numerical_scores.values()) / len(numerical_scores),
                    'score_range': {
                        'min': min(numerical_scores.values()),
                        'max': max(numerical_scores.values())
                    }
                }
            }
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing scoring methods: {e}")
            return None
    
    def _validate_scoring_prerequisites(self):
        """Validate that all required data is available for scoring."""
        if self.main_window.current_sem_image is None:
            QMessageBox.warning(
                self.main_window,
                "No SEM Image",
                "Please load a SEM image first."
            )
            return False
        
        if not self._get_aligned_overlay():
            QMessageBox.warning(
                self.main_window,
                "No Alignment",
                "Please perform alignment first."
            )
            return False
        
        return True
    
    def _get_current_sem_image(self):
        """Get the current SEM image for scoring."""
        # Try processed image first, then original
        if hasattr(self.main_window, 'image_processor'):
            if self.main_window.image_processor.has_filtered_image():
                return self.main_window.image_processor.filtered_sem_image
        
        return self.main_window.current_sem_image
    
    def _get_aligned_overlay(self):
        """Get the aligned overlay for scoring."""
        # Try to get from alignment result first
        if hasattr(self.main_window, 'alignment_controller'):
            if self.main_window.alignment_controller.current_alignment_result:
                alignment_result = self.main_window.alignment_controller.current_alignment_result
                if 'transformed_gds' in alignment_result:
                    return alignment_result['transformed_gds']
        
        # Fallback to image viewer overlay
        if hasattr(self.main_window, 'image_viewer'):
            return getattr(self.main_window.image_viewer, 'gds_overlay', None)
        
        return None
    
    def _get_sem_image_info(self):
        """Get information about the current SEM image."""
        sem_image = self._get_current_sem_image()
        if sem_image is None:
            return None
        
        return {
            'shape': sem_image.shape,
            'dtype': str(sem_image.dtype),
            'min_value': float(np.min(sem_image)),
            'max_value': float(np.max(sem_image)),
            'mean_value': float(np.mean(sem_image))
        }
    
    def _get_gds_info(self):
        """Get information about the current GDS structure."""
        if hasattr(self.main_window, 'gds_manager'):
            return self.main_window.gds_manager.get_current_structure_info()
        return None
    
    def _get_alignment_info(self):
        """Get information about the current alignment."""
        if hasattr(self.main_window, 'alignment_controller'):
            return self.main_window.alignment_controller.get_alignment_info()
        return None
    
    def has_scores(self):
        """Check if scoring results are available."""
        return bool(self.current_scoring_results)
    
    def clear_scores(self):
        """Clear current scoring results."""
        self.current_scoring_results = {}
        self.scoring_history = []
        print("Scoring results cleared")
    
    def get_best_score(self):
        """Get the best score from current results."""
        if not self.current_scoring_results:
            return None
        
        # Find the highest score (assuming higher is better for most metrics)
        best_score = None
        best_method = None
        
        for method, score in self.current_scoring_results.items():
            if isinstance(score, (int, float)) and not np.isnan(score):
                if method == 'MSE':  # MSE is inverse (lower is better)
                    score = 1.0 / (1.0 + score)
                
                if best_score is None or score > best_score:
                    best_score = score
                    best_method = method
        
        return {'method': best_method, 'score': best_score} if best_method else None
