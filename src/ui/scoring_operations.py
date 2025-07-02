"""
Scoring Operations Module
Handles all scoring operations, metrics calculation, and scoring-related functionality.
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QObject, Signal
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from src.services.simple_scoring_service import ScoringService


class ScoringOperations(QObject):
    """Handles scoring operations and metrics calculation."""
    
    # Signals
    scores_calculated = Signal(dict)  # Emitted when scores are calculated
    batch_scoring_completed = Signal(list)  # Emitted when batch scoring is completed
    
    def __init__(self, main_window):
        """Initialize scoring operations with reference to main window."""
        super().__init__()
        self.main_window = main_window
        self.scoring_service = ScoringService()
        
        # Scoring state
        self.current_scoring_method = "SSIM"
        self.current_scoring_results = {}
        self.available_methods = ["SSIM", "MSE", "PSNR", "Cross-Correlation", "Mutual Information"]
        
    def calculate_scores(self):
        """Calculate alignment scores using the current method."""
        try:
            # Validate prerequisites
            if not self._validate_scoring_prerequisites():
                return
            
            print(f"Calculating scores using method: {self.current_scoring_method}")
            
            sem_image = self.main_window.current_sem_image
            
            # Get aligned GDS overlay
            aligned_overlay = self._get_aligned_overlay()
            if aligned_overlay is None:
                raise RuntimeError("No aligned GDS overlay available")
            
            # Calculate scores using scoring service
            scoring_results = self.scoring_service.calculate_alignment_scores(
                sem_image, aligned_overlay, methods=[self.current_scoring_method]
            )
            
            if scoring_results:
                # Store results
                self.current_scoring_results = scoring_results
                
                # Update status
                primary_score = scoring_results.get(self.current_scoring_method, 'N/A')
                self.main_window.status_bar.showMessage(
                    f"Scoring completed - {self.current_scoring_method}: {primary_score}"
                )
                
                # Emit signal
                self.scores_calculated.emit(scoring_results)
                
                print(f"✓ Scoring completed: {scoring_results}")
                
                return scoring_results
            else:
                raise RuntimeError("Scoring service returned empty results")
                
        except Exception as e:
            error_msg = f"Failed to calculate scores: {str(e)}"
            print(f"Scoring error: {error_msg}")
            QMessageBox.critical(self.main_window, "Scoring Error", error_msg)
            return None
    
    def calculate_all_scores(self):
        """Calculate scores using all available methods."""
        try:
            if not self._validate_scoring_prerequisites():
                return
            
            print("Calculating scores using all available methods...")
            
            sem_image = self.main_window.current_sem_image
            aligned_overlay = self._get_aligned_overlay()
            
            if aligned_overlay is None:
                raise RuntimeError("No aligned GDS overlay available")
            
            # Calculate scores for all methods
            scoring_results = self.scoring_service.calculate_alignment_scores(
                sem_image, aligned_overlay, methods=self.available_methods
            )
            
            if scoring_results:
                self.current_scoring_results = scoring_results
                
                # Create summary message
                summary = ", ".join([f"{method}: {score}" for method, score in scoring_results.items()])
                self.main_window.status_bar.showMessage(f"All scores calculated - {summary}")
                
                # Emit signal
                self.scores_calculated.emit(scoring_results)
                
                print(f"✓ All scores calculated: {scoring_results}")
                
                return scoring_results
            else:
                raise RuntimeError("Scoring service returned empty results")
                
        except Exception as e:
            error_msg = f"Failed to calculate all scores: {str(e)}"
            print(f"Scoring error: {error_msg}")
            QMessageBox.critical(self.main_window, "Scoring Error", error_msg)
            return None
    
    def set_scoring_method(self, method):
        """Set the current scoring method."""
        if method in self.available_methods:
            self.current_scoring_method = method
            print(f"Scoring method set to: {method}")
        else:
            print(f"Warning: Unknown scoring method: {method}")
    
    def get_scoring_method(self):
        """Get the current scoring method."""
        return self.current_scoring_method
    
    def get_available_methods(self):
        """Get list of available scoring methods."""
        return self.available_methods.copy()
    
    def get_current_scores(self):
        """Get the current scoring results."""
        return self.current_scoring_results.copy()
    
    def calculate_custom_score(self, metric_func, metric_name="Custom"):
        """Calculate a custom scoring metric."""
        try:
            if not self._validate_scoring_prerequisites():
                return None
            
            sem_image = self.main_window.current_sem_image
            aligned_overlay = self._get_aligned_overlay()
            
            if aligned_overlay is None:
                raise RuntimeError("No aligned GDS overlay available")
            
            # Apply custom metric function
            score = metric_func(sem_image, aligned_overlay)
            
            # Store result
            self.current_scoring_results[metric_name] = score
            
            self.main_window.status_bar.showMessage(f"Custom metric {metric_name}: {score}")
            
            # Emit signal
            self.scores_calculated.emit({metric_name: score})
            
            print(f"✓ Custom metric {metric_name} calculated: {score}")
            
            return score
            
        except Exception as e:
            error_msg = f"Failed to calculate custom score: {str(e)}"
            print(f"Custom scoring error: {error_msg}")
            QMessageBox.critical(self.main_window, "Custom Scoring Error", error_msg)
            return None
    
    def batch_scoring(self, image_pairs, methods=None):
        """Perform batch scoring on multiple image pairs."""
        try:
            if not image_pairs:
                raise ValueError("No image pairs provided for batch scoring")
            
            if methods is None:
                methods = [self.current_scoring_method]
            
            print(f"Starting batch scoring for {len(image_pairs)} image pairs...")
            
            batch_results = []
            
            for i, (sem_img, gds_overlay) in enumerate(image_pairs):
                try:
                    print(f"Processing pair {i+1}/{len(image_pairs)}")
                    
                    # Calculate scores for this pair
                    pair_scores = self.scoring_service.calculate_alignment_scores(
                        sem_img, gds_overlay, methods=methods
                    )
                    
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
    
    def export_scores(self, file_path, include_images=False):
        """Export scoring results to a file."""
        try:
            if not self.current_scoring_results:
                raise ValueError("No scoring results to export")
            
            export_data = {
                'scoring_method': self.current_scoring_method,
                'scores': self.current_scoring_results,
                'timestamp': str(np.datetime64('now')),
                'sem_image_info': self._get_sem_image_info(),
                'gds_info': self._get_gds_info(),
                'alignment_info': self._get_alignment_info()
            }
            
            # Add image data if requested
            if include_images:
                export_data['images'] = self._get_image_data_for_export()
            
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
            all_scores = self.calculate_all_scores()
            if not all_scores:
                return None
            
            # Create comparison report
            comparison = {
                'methods_compared': list(all_scores.keys()),
                'scores': all_scores,
                'best_method': max(all_scores.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0),
                'worst_method': min(all_scores.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else float('inf')),
                'score_range': {
                    'min': min(score for score in all_scores.values() if isinstance(score, (int, float))),
                    'max': max(score for score in all_scores.values() if isinstance(score, (int, float)))
                }
            }
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing scoring methods: {e}")
            return None
    
    def _get_aligned_overlay(self):
        """Get the current aligned GDS overlay."""
        try:
            # Try to get aligned overlay from image viewer
            if hasattr(self.main_window.image_viewer, 'gds_overlay'):
                return self.main_window.image_viewer.gds_overlay
            
            # Fallback to current overlay
            if hasattr(self.main_window, 'gds_operations'):
                return self.main_window.gds_operations.current_gds_overlay
            
            return None
            
        except Exception as e:
            print(f"Error getting aligned overlay: {e}")
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
        
        if not hasattr(self.main_window, 'gds_operations') or not self.main_window.gds_operations.is_structure_selected():
            QMessageBox.warning(
                self.main_window,
                "No Structure Selected",
                "Please select a GDS structure first."
            )
            return False
        
        return True
    
    def _get_sem_image_info(self):
        """Get information about the current SEM image."""
        if self.main_window.current_sem_image is None:
            return None
        
        img = self.main_window.current_sem_image
        return {
            'shape': img.shape,
            'dtype': str(img.dtype),
            'min_value': float(np.min(img)),
            'max_value': float(np.max(img)),
            'mean_value': float(np.mean(img))
        }
    
    def _get_gds_info(self):
        """Get information about the current GDS structure."""
        if hasattr(self.main_window, 'gds_operations'):
            return self.main_window.gds_operations.get_current_structure_info()
        return None
    
    def _get_alignment_info(self):
        """Get information about the current alignment."""
        if hasattr(self.main_window, 'alignment_operations'):
            return self.main_window.alignment_operations.get_alignment_info()
        return None
    
    def _get_image_data_for_export(self):
        """Get image data for export (encoded as base64)."""
        try:
            import base64
            
            data = {}
            
            # Encode SEM image
            if self.main_window.current_sem_image is not None:
                _, buffer = cv2.imencode('.png', self.main_window.current_sem_image)
                data['sem_image'] = base64.b64encode(buffer).decode('utf-8')
            
            # Encode GDS overlay
            overlay = self._get_aligned_overlay()
            if overlay is not None:
                _, buffer = cv2.imencode('.png', overlay)
                data['gds_overlay'] = base64.b64encode(buffer).decode('utf-8')
            
            return data
            
        except Exception as e:
            print(f"Error encoding images for export: {e}")
            return {}
    
    def has_scores(self):
        """Check if scoring results are available."""
        return bool(self.current_scoring_results)
    
    def clear_scores(self):
        """Clear current scoring results."""
        self.current_scoring_results = {}
        print("Scoring results cleared")
