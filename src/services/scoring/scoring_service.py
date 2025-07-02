"""Scoring service that aggregates metrics and manages reports."""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime
from PySide6.QtCore import QObject, Signal

from .pixel_score_service import PixelScoreService
from .ssim_score_service import SSIMScoreService
from .iou_score_service import IOUScoreService
from src.core.utils import get_logger, get_results_path
import numpy as np


class ScoringService(QObject):
    """Main service for aggregating scoring metrics and managing reports."""
    
    # Signals
    scoring_completed = Signal(dict)  # Emitted when full scoring is completed
    report_saved = Signal(str)        # Emitted when report is saved
    scoring_error = Signal(str)       # Emitted on scoring error
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        
        # Initialize scoring services
        self.pixel_service = PixelScoreService()
        self.ssim_service = SSIMScoreService()
        self.iou_service = IOUScoreService()
        
        # Connect to error signals
        self.pixel_service.score_error.connect(self.scoring_error.emit)
        self.ssim_service.ssim_error.connect(self.scoring_error.emit)
        self.iou_service.iou_error.connect(self.scoring_error.emit)
        
        self._last_full_result = None
    
    def compute_full_score(self, image1: np.ndarray, image2: np.ndarray,
                          config: Optional[Dict] = None) -> Dict:
        """
        Compute all scoring metrics for two images.
        
        Args:
            image1: First image (typically SEM)
            image2: Second image (typically GDS)
            config: Configuration for scoring parameters
            
        Returns:
            Dictionary containing all computed metrics
        """
        if config is None:
            config = self._get_default_config()
        
        try:
            self.logger.info("Starting full scoring computation")
            
            # Initialize result structure
            full_result = {
                'timestamp': datetime.now().isoformat(),
                'config': config,
                'image_info': {
                    'image1_shape': image1.shape,
                    'image2_shape': image2.shape,
                    'images_same_size': image1.shape == image2.shape
                },
                'metrics': {}
            }
            
            # Compute pixel overlap metrics
            if config.get('compute_pixel_overlap', True):
                self.logger.info("Computing pixel overlap metrics")
                pixel_result = self.pixel_service.compute_pixel_overlap(
                    image1, image2, 
                    threshold=config.get('pixel_threshold', 127)
                )
                full_result['metrics']['pixel_overlap'] = pixel_result
            
            # Compute pixel differences
            if config.get('compute_pixel_differences', True):
                self.logger.info("Computing pixel difference metrics")
                diff_result = self.pixel_service.compute_pixel_difference(image1, image2)
                full_result['metrics']['pixel_differences'] = diff_result
            
            # Compute SSIM
            if config.get('compute_ssim', True):
                self.logger.info("Computing SSIM metrics")
                ssim_result = self.ssim_service.compute_ssim(
                    image1, image2,
                    window_size=config.get('ssim_window_size', 7)
                )
                full_result['metrics']['ssim'] = ssim_result
            
            # Compute Multi-Scale SSIM
            if config.get('compute_ms_ssim', True):
                self.logger.info("Computing Multi-Scale SSIM")
                ms_ssim_result = self.ssim_service.compute_multiscale_ssim(
                    image1, image2,
                    scales=config.get('ms_ssim_scales', [1.0, 0.5, 0.25])
                )
                full_result['metrics']['ms_ssim'] = ms_ssim_result
            
            # Compute IoU
            if config.get('compute_iou', True):
                self.logger.info("Computing IoU metrics")
                iou_result = self.iou_service.compute_iou(
                    image1, image2,
                    threshold=config.get('iou_threshold', 127)
                )
                full_result['metrics']['iou'] = iou_result
            
            # Compute multi-threshold IoU
            if config.get('compute_multi_threshold_iou', True):
                self.logger.info("Computing multi-threshold IoU")
                multi_iou_result = self.iou_service.compute_multi_threshold_iou(
                    image1, image2,
                    thresholds=config.get('iou_thresholds', [64, 96, 127, 160, 192])
                )
                full_result['metrics']['multi_threshold_iou'] = multi_iou_result
            
            # Compute local statistics if requested
            if config.get('compute_local_stats', False):
                self.logger.info("Computing local statistics")
                self._compute_local_statistics(image1, image2, config, full_result)
            
            # Compute aggregate scores
            self._compute_aggregate_scores(full_result)
            
            self._last_full_result = full_result
            self.scoring_completed.emit(full_result)
            self.logger.info("Full scoring computation completed")
            
            return full_result
            
        except Exception as e:
            error_msg = f"Failed to compute full score: {e}"
            self.logger.error(error_msg)
            self.scoring_error.emit(error_msg)
            return {}
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for scoring."""
        return {
            'compute_pixel_overlap': True,
            'compute_pixel_differences': True,
            'compute_ssim': True,
            'compute_ms_ssim': True,
            'compute_iou': True,
            'compute_multi_threshold_iou': True,
            'compute_local_stats': False,
            'pixel_threshold': 127,
            'ssim_window_size': 7,
            'ms_ssim_scales': [1.0, 0.5, 0.25],
            'iou_threshold': 127,
            'iou_thresholds': [64, 96, 127, 160, 192],
            'local_grid_size': 64
        }
    
    def _compute_local_statistics(self, image1: np.ndarray, image2: np.ndarray,
                                 config: Dict, full_result: Dict) -> None:
        """Compute local statistics for different metrics."""
        grid_size = config.get('local_grid_size', 64)
        
        # Local SSIM statistics
        local_ssim = self.ssim_service.compute_local_ssim_statistics(
            image1, image2, 
            window_size=config.get('ssim_window_size', 7),
            grid_size=grid_size
        )
        full_result['metrics']['local_ssim'] = local_ssim
        
        # Local IoU statistics
        local_iou = self.iou_service.compute_local_iou(
            image1, image2,
            grid_size=grid_size,
            threshold=config.get('iou_threshold', 127)
        )
        full_result['metrics']['local_iou'] = local_iou
    
    def _compute_aggregate_scores(self, full_result: Dict) -> None:
        """Compute aggregate scores from individual metrics."""
        metrics = full_result.get('metrics', {})
        
        # Collect individual scores
        scores = []
        weights = []
        
        # Pixel overlap score (use F1 score)
        if 'pixel_overlap' in metrics and 'f1_score' in metrics['pixel_overlap']:
            scores.append(metrics['pixel_overlap']['f1_score'])
            weights.append(0.3)
        
        # SSIM score
        if 'ssim' in metrics and 'ssim_score' in metrics['ssim']:
            scores.append(metrics['ssim']['ssim_score'])
            weights.append(0.3)
        
        # IoU score
        if 'iou' in metrics and 'iou_score' in metrics['iou']:
            scores.append(metrics['iou']['iou_score'])
            weights.append(0.4)
        
        # Compute weighted average
        if scores and weights:
            weights = np.array(weights[:len(scores)])
            weights = weights / np.sum(weights)  # Normalize weights
            
            aggregate_score = np.sum(np.array(scores) * weights)
            
            full_result['aggregate'] = {
                'overall_score': float(aggregate_score),
                'overall_percentage': float(aggregate_score * 100),
                'component_scores': {
                    'pixel_f1': scores[0] if len(scores) > 0 else 0.0,
                    'ssim': scores[1] if len(scores) > 1 else 0.0,
                    'iou': scores[2] if len(scores) > 2 else 0.0
                },
                'weights_used': weights.tolist(),
                'num_components': len(scores)
            }
    
    def save_report(self, result: Optional[Dict] = None, filename: str = "",
                   sem_name: str = "", gds_name: str = "", mode: str = "manual") -> bool:
        """
        Save a scoring report to disk.
        
        Args:
            result: Scoring result to save (uses last result if None)
            filename: Custom filename (auto-generated if empty)
            sem_name: SEM image name for filename
            gds_name: GDS structure name for filename
            mode: Alignment mode ('manual', 'auto', etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if result is None:
            result = self._last_full_result
        
        if not result:
            self.scoring_error.emit("No scoring result to save")
            return False
        
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if sem_name and gds_name:
                    filename = f"{sem_name}_{gds_name}_{mode}_score_{timestamp}.json"
                else:
                    filename = f"scoring_report_{mode}_{timestamp}.json"
            
            # Ensure .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            save_path = get_results_path("Scoring/reports") / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata to result
            save_result = result.copy()
            save_result['metadata'] = {
                'save_timestamp': datetime.now().isoformat(),
                'sem_name': sem_name,
                'gds_name': gds_name,
                'alignment_mode': mode,
                'software_version': '1.0.0'  # You might want to make this configurable
            }
            
            # Save to JSON
            with open(save_path, 'w') as f:
                json.dump(save_result, f, indent=2, default=str)
            
            self.report_saved.emit(str(save_path))
            self.logger.info(f"Saved scoring report to {save_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to save scoring report: {e}"
            self.logger.error(error_msg)
            self.scoring_error.emit(error_msg)
            return False
    
    def load_report(self, report_path: str) -> Optional[Dict]:
        """
        Load a previously saved scoring report.
        
        Args:
            report_path: Path to the report file
            
        Returns:
            Loaded report data or None if failed
        """
        try:
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            self.logger.info(f"Loaded scoring report from {report_path}")
            return report_data
            
        except Exception as e:
            error_msg = f"Failed to load scoring report: {e}"
            self.logger.error(error_msg)
            self.scoring_error.emit(error_msg)
            return None
    
    def get_last_result(self) -> Optional[Dict]:
        """Get the most recent full scoring result."""
        return self._last_full_result
    
    def create_summary_report(self, results: List[Dict]) -> Dict:
        """
        Create a summary report from multiple scoring results.
        
        Args:
            results: List of scoring result dictionaries
            
        Returns:
            Summary report dictionary
        """
        if not results:
            return {'error': 'No results provided for summary'}
        
        try:
            summary = {
                'summary_timestamp': datetime.now().isoformat(),
                'num_results': len(results),
                'statistics': {}
            }
            
            # Collect aggregate scores
            aggregate_scores = []
            ssim_scores = []
            iou_scores = []
            pixel_f1_scores = []
            
            for result in results:
                if 'aggregate' in result and 'overall_score' in result['aggregate']:
                    aggregate_scores.append(result['aggregate']['overall_score'])
                
                if 'metrics' in result:
                    metrics = result['metrics']
                    
                    if 'ssim' in metrics and 'ssim_score' in metrics['ssim']:
                        ssim_scores.append(metrics['ssim']['ssim_score'])
                    
                    if 'iou' in metrics and 'iou_score' in metrics['iou']:
                        iou_scores.append(metrics['iou']['iou_score'])
                    
                    if 'pixel_overlap' in metrics and 'f1_score' in metrics['pixel_overlap']:
                        pixel_f1_scores.append(metrics['pixel_overlap']['f1_score'])
            
            # Compute statistics for each metric
            for name, scores in [
                ('aggregate', aggregate_scores),
                ('ssim', ssim_scores),
                ('iou', iou_scores),
                ('pixel_f1', pixel_f1_scores)
            ]:
                if scores:
                    scores_array = np.array(scores)
                    summary['statistics'][name] = {
                        'mean': float(np.mean(scores_array)),
                        'std': float(np.std(scores_array)),
                        'min': float(np.min(scores_array)),
                        'max': float(np.max(scores_array)),
                        'median': float(np.median(scores_array)),
                        'count': len(scores)
                    }
            
            self.logger.info(f"Created summary report for {len(results)} results")
            return summary
            
        except Exception as e:
            error_msg = f"Failed to create summary report: {e}"
            self.logger.error(error_msg)
            return {'error': error_msg}
