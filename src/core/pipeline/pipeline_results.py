"""
Pipeline Result Structure for SEM/GDS Comparison Tool

This module defines the unified result structure that aggregates all pipeline outputs
for both in-app display and export functionality.
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import os
import json
try:
    import imageio
except ImportError:
    imageio = None


class PipelineResults:
    """
    Unified result structure for pipeline execution results.
    
    Aggregates filter chain details, transform matrices, and scoring results
    in a format suitable for both in-app display and export.
    """
    
    def __init__(self):
        """Initialize empty result structure."""
        self.results = {
            # Core processing results
            'filter_chain': [],
            'transform_matrix': None,
            'scoring_results': {},
            
            # Intermediate images for display
            'intermediate_images': {
                'original_sem': None,
                'filtered_sem': None,
                'original_gds': None,
                'aligned_gds': None,
                'overlay': None
            },
            
            # Execution metadata
            'metadata': {
                'timestamp': None,
                'execution_mode': None,
                'total_processing_time': None,
                'stages_completed': [],
                'config_used': {},
                'early_exit': False,
                'success': False
            },
            
            # Stage-specific results
            'stage_results': {
                'filtering': {},
                'alignment': {},
                'scoring': {}
            },
            
            # Export-ready data
            'export_data': {
                'json_report': None,
                'overlay_paths': [],
                'filter_image_paths': [],
                'aligned_image_paths': [],
                'alignment_matrix_path': None
            },
            
            # Reproducibility information
            'parameter_history': {},
            'pipeline_log': []
        }
    
    def set_filter_results(self, filter_chain: List[Dict], filtered_image: np.ndarray):
        """
        Set filtering stage results.
        
        Args:
            filter_chain: List of applied filters with parameters
            filtered_image: Resulting filtered image
        """
        self.results['filter_chain'] = filter_chain
        self.results['intermediate_images']['filtered_sem'] = filtered_image
        self.results['stage_results']['filtering'] = {
            'filter_count': len(filter_chain),
            'successful_filters': len([f for f in filter_chain if f.get('success', False)]),
            'failed_filters': len([f for f in filter_chain if not f.get('success', True)]),
            'processing_time': None
        }
    
    def set_alignment_results(self, transform_matrix: np.ndarray, aligned_image: np.ndarray, 
                            confidence: float, method: str):
        """
        Set alignment stage results.
        
        Args:
            transform_matrix: 3x3 transformation matrix
            aligned_image: Aligned GDS image
            confidence: Alignment confidence score
            method: Alignment method used
        """
        self.results['transform_matrix'] = transform_matrix
        self.results['intermediate_images']['aligned_gds'] = aligned_image
        self.results['stage_results']['alignment'] = {
            'confidence': confidence,
            'method_used': method,
            'transform_parameters': self._extract_transform_parameters(transform_matrix),
            'success': confidence > 0.1
        }
    
    def set_scoring_results(self, scoring_results: Dict[str, float]):
        """
        Set scoring stage results.
        
        Args:
            scoring_results: Dictionary of scoring method results
        """
        self.results['scoring_results'] = scoring_results
        self.results['stage_results']['scoring'] = {
            'methods_used': list(scoring_results.keys()),
            'best_score': max(scoring_results.values()) if scoring_results else 0.0,
            'average_score': sum(scoring_results.values()) / len(scoring_results) if scoring_results else 0.0,
            'all_scores': scoring_results
        }
    
    def set_metadata(self, mode: str, config: Dict, start_time: float, stages_completed: List[str]):
        """
        Set execution metadata.
        
        Args:
            mode: Execution mode ('manual' or 'automatic')
            config: Configuration used for execution
            start_time: Pipeline start timestamp
            stages_completed: List of completed stage names
        """
        current_time = time.time()
        self.results['metadata'].update({
            'timestamp': datetime.now().isoformat(),
            'execution_mode': mode,
            'total_processing_time': current_time - start_time,
            'stages_completed': stages_completed,
            'config_used': config.copy(),
            'early_exit': len(stages_completed) < 3,  # Less than filtering, alignment, scoring
            'success': len(stages_completed) > 0
        })
    
    def set_original_images(self, sem_image: np.ndarray, gds_image: np.ndarray):
        """
        Set original input images.
        
        Args:
            sem_image: Original SEM image
            gds_image: Original GDS image
        """
        self.results['intermediate_images']['original_sem'] = sem_image
        self.results['intermediate_images']['original_gds'] = gds_image
    
    def create_overlay_image(self) -> Optional[np.ndarray]:
        """
        Create overlay image from filtered SEM and aligned GDS.
        
        Returns:
            Overlay image if both images available, None otherwise
        """
        filtered_sem = self.results['intermediate_images'].get('filtered_sem')
        aligned_gds = self.results['intermediate_images'].get('aligned_gds')
        
        if filtered_sem is not None and aligned_gds is not None:
            # Simple overlay: blend images
            try:
                # Ensure both images are same size
                if filtered_sem.shape != aligned_gds.shape:
                    # Resize aligned GDS to match filtered SEM
                    from skimage.transform import resize
                    aligned_gds = resize(aligned_gds, filtered_sem.shape, preserve_range=True)
                
                # Create RGB overlay (SEM in red, GDS in green)
                overlay = np.zeros((*filtered_sem.shape, 3), dtype=np.uint8)
                overlay[:, :, 0] = (filtered_sem * 255).astype(np.uint8)  # Red channel
                overlay[:, :, 1] = (aligned_gds * 255).astype(np.uint8)   # Green channel
                
                self.results['intermediate_images']['overlay'] = overlay
                return overlay
                
            except Exception as e:
                print(f"Failed to create overlay: {e}")
                return None
        
        return None
    
    def get_display_summary(self) -> Dict[str, Any]:
        """
        Get summary for in-app display.
        
        Returns:
            Dictionary with key metrics for UI display
        """
        return {
            'execution_mode': self.results['metadata'].get('execution_mode', 'unknown'),
            'processing_time': self.results['metadata'].get('total_processing_time', 0),
            'stages_completed': len(self.results['metadata'].get('stages_completed', [])),
            'filter_count': len(self.results['filter_chain']),
            'alignment_confidence': self.results['stage_results']['alignment'].get('confidence', 0.0),
            'best_score': self.results['stage_results']['scoring'].get('best_score', 0.0),
            'success': self.results['metadata'].get('success', False),
            'early_exit': self.results['metadata'].get('early_exit', False)
        }
    
    def prepare_export_data(self) -> Dict[str, Any]:
        """
        Prepare data structure for JSON export.
        
        Returns:
            Export-ready dictionary with serializable data
        """
        export_dict = {}
        
        # Copy basic results, excluding non-serializable numpy arrays
        for key, value in self.results.items():
            if key != 'intermediate_images':
                if isinstance(value, np.ndarray):
                    export_dict[key] = {
                        'shape': value.shape,
                        'dtype': str(value.dtype),
                        'data_summary': 'array_excluded_from_json'
                    }
                else:
                    export_dict[key] = value
        
        # Add image metadata without actual arrays
        export_dict['intermediate_images'] = {}
        for img_name, img_array in self.results['intermediate_images'].items():
            if img_array is not None:
                export_dict['intermediate_images'][img_name] = {
                    'shape': getattr(img_array, 'shape', None),
                    'dtype': str(getattr(img_array, 'dtype', 'unknown')),
                    'available': True
                }
            else:
                export_dict['intermediate_images'][img_name] = {'available': False}
        
        # Convert numpy arrays in transform matrix to lists
        if 'transform_matrix' in export_dict and export_dict['transform_matrix'] is not None:
            if isinstance(self.results['transform_matrix'], np.ndarray):
                export_dict['transform_matrix'] = self.results['transform_matrix'].tolist()
        
        return export_dict
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the complete results dictionary.
        
        Returns:
            Complete results dictionary
        """
        return self.results
    
    def _extract_transform_parameters(self, transform_matrix: np.ndarray) -> Dict[str, float]:
        """
        Extract transformation parameters from matrix.
        
        Args:
            transform_matrix: 3x3 transformation matrix
            
        Returns:
            Dictionary with translation, rotation, scale parameters
        """
        try:
            if transform_matrix is None or transform_matrix.shape != (3, 3):
                return {'translation_x': 0, 'translation_y': 0, 'rotation': 0, 'scale': 1}
            
            # Extract translation
            tx = transform_matrix[0, 2]
            ty = transform_matrix[1, 2]
            
            # Extract scale and rotation (simplified)
            scale_x = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[0, 1]**2)
            scale_y = np.sqrt(transform_matrix[1, 0]**2 + transform_matrix[1, 1]**2)
            scale = (scale_x + scale_y) / 2  # Average scale
            
            # Extract rotation (in degrees)
            rotation = np.degrees(np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0]))
            
            return {
                'translation_x': float(tx),
                'translation_y': float(ty),
                'rotation': float(rotation),
                'scale': float(scale)
            }
            
        except Exception:
            return {'translation_x': 0, 'translation_y': 0, 'rotation': 0, 'scale': 1}
    
    def export_json_report(self, path: str) -> bool:
        """
        Export pipeline results as a JSON report (excluding raw image arrays).
        
        Args:
            path: Path to save the JSON file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = self.prepare_export_data()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            self.results['export_data']['json_report'] = path
            return True
        except Exception as e:
            print(f"Failed to export JSON report: {e}")
            return False

    def export_overlay_image(self, path: str) -> bool:
        """
        Export overlay image (if available) as PNG.
        
        Args:
            path: Path to save the overlay image
        
        Returns:
            True if successful, False otherwise
        """
        overlay = self.results['intermediate_images'].get('overlay')
        if overlay is not None and imageio is not None:
            try:
                imageio.imwrite(path, overlay)
                self.results['export_data']['overlay_paths'].append(path)
                return True
            except Exception as e:
                print(f"Failed to export overlay image: {e}")
                return False
        return False

    def export_stage_images(self, out_dir: str) -> dict:
        """
        Export filtered, aligned, and overlay images to a directory as PNGs.
        
        Args:
            out_dir: Output directory
        
        Returns:
            Dict with paths to exported images
        """
        if imageio is None:
            print("imageio is not installed. Cannot export images.")
            return {}
        os.makedirs(out_dir, exist_ok=True)
        paths = {}
        for key in ['filtered_sem', 'aligned_gds', 'overlay']:
            img = self.results['intermediate_images'].get(key)
            if img is not None:
                out_path = os.path.join(out_dir, f"{key}.png")
                try:
                    imageio.imwrite(out_path, img)
                    paths[key] = out_path
                except Exception as e:
                    print(f"Failed to export {key}: {e}")
        self.results['export_data']['filter_image_paths'] = [paths.get('filtered_sem')] if 'filtered_sem' in paths else []
        self.results['export_data']['aligned_image_paths'] = [paths.get('aligned_gds')] if 'aligned_gds' in paths else []
        self.results['export_data']['overlay_paths'] = [paths.get('overlay')] if 'overlay' in paths else []
        return paths

    def export_transform_matrix(self, path: str) -> bool:
        """
        Export transform matrix as a CSV file.
        
        Args:
            path: Path to save the CSV file
        
        Returns:
            True if successful, False otherwise
        """
        matrix = self.results.get('transform_matrix')
        if matrix is not None:
            try:
                np.savetxt(path, matrix, delimiter=',')
                self.results['export_data']['alignment_matrix_path'] = path
                return True
            except Exception as e:
                print(f"Failed to export transform matrix: {e}")
                return False
        return False
