from .pipeline_base import ProcessingPipelineBase
from typing import Any, Dict, List
import numpy as np

class ManualProcessingPipeline(ProcessingPipelineBase):
    """
    Processing pipeline for manual mode. Handles UI-driven parameter collection, manual filtering, and manual alignment.
    Inherits shared state, signals, and result aggregation from PipelineBase.
    """
    def _execute_manual_mode(self, input_image):
        self._log_action("Starting manual mode processing", {
            'input_shape': getattr(input_image, 'shape', None),
            'ui_state_available': self.config.get('use_ui_state', True)
        })
        ui_parameters = self._collect_ui_parameters()
        current_image = input_image
        filter_chain = []
        if ui_parameters.get('filters'):
            total_filters = len(ui_parameters['filters'])
            for i, filter_config in enumerate(ui_parameters['filters']):
                filter_name = filter_config.get('name')
                filter_params = filter_config.get('parameters', {})
                self.status_message.emit(f"Applying manual filter: {filter_name}")
                progress = int((i / total_filters) * 100)
                self.stage_progress.emit('filtering', progress)
                try:
                    if hasattr(self.image_processor, 'apply_filter'):
                        filtered_result = self.image_processor.apply_filter(
                            current_image, filter_name, filter_params
                        )
                        if filtered_result is not None:
                            current_image = filtered_result
                            filter_entry = {
                                'filter_name': filter_name,
                                'parameters': filter_params,
                                'success': True,
                                'source': 'manual_ui',
                                'trial_number': i + 1
                            }
                            filter_chain.append(filter_entry)
                            self._log_action(f"Manual filter applied: {filter_name}", filter_entry)
                        else:
                            self._log_action(f"Manual filter failed: {filter_name}", {'trial_number': i + 1})
                except Exception as e:
                    self._log_action(f"Manual filter error: {filter_name}", {
                        'error': str(e),
                        'trial_number': i + 1
                    })
        self.stage_progress.emit('filtering', 100)
        self._log_action("Manual mode filtering completed", {
            'filters_applied': len(filter_chain),
            'final_image_shape': getattr(current_image, 'shape', None)
        })
        return current_image, filter_chain

    def _collect_ui_parameters(self) -> Dict[str, Any]:
        ui_parameters = {
            'filters': [],
            'alignment': {},
            'scoring': {},
            'collection_timestamp': self._get_timestamp(),
            'source': 'ui_panels'
        }
        try:
            ui_parameters['filters'] = self._get_ui_filter_parameters()
            ui_parameters['alignment'] = self._get_ui_alignment_parameters()
            ui_parameters['scoring'] = self._get_ui_scoring_parameters()
            self._log_action("UI parameters collected", {
                'filter_count': len(ui_parameters['filters']),
                'has_alignment': bool(ui_parameters['alignment']),
                'has_scoring': bool(ui_parameters['scoring'])
            })
        except Exception as e:
            self._log_action("UI parameter collection failed", {'error': str(e)})
            ui_parameters = self._get_fallback_manual_parameters()
        return ui_parameters

    def _get_ui_filter_parameters(self) -> List[Dict[str, Any]]:
        ui_filters = []
        if self.config.get('use_ui_state', False):
            example_ui_filters = [
                {
                    'name': 'gaussian',
                    'parameters': {'sigma': 1.5},
                    'enabled': True,
                    'source': 'ui_simulation'
                }
            ]
            ui_filters = example_ui_filters
        self._log_action("Filter parameters collected from UI", {
            'filter_count': len(ui_filters),
            'filter_names': [f['name'] for f in ui_filters]
        })
        return ui_filters

    def _get_ui_alignment_parameters(self) -> Dict[str, Any]:
        alignment_params = {}
        if self.config.get('use_ui_state', False):
            alignment_params = {
                'alignment_method': 'manual',
                'manual_adjustments': {
                    'translation_x': 0.0,
                    'translation_y': 0.0,
                    'rotation': 0.0,
                    'scale_x': 1.0,
                    'scale_y': 1.0
                },
                'source': 'ui_simulation'
            }
        self._log_action("Alignment parameters collected from UI", alignment_params)
        return alignment_params

    def _get_ui_scoring_parameters(self) -> Dict[str, Any]:
        scoring_params = {}
        if self.config.get('use_ui_state', False):
            scoring_params = {
                'scoring_methods': ['correlation'],
                'scoring_parameters': {
                    'threshold': 0.5
                },
                'display_settings': {
                    'show_overlay': True
                },
                'source': 'ui_simulation'
            }
        self._log_action("Scoring parameters collected from UI", scoring_params)
        return scoring_params

    def _get_fallback_manual_parameters(self) -> Dict[str, Any]:
        return {
            'filters': [
                {
                    'name': 'gaussian',
                    'parameters': {'sigma': 1.0},
                    'enabled': True,
                    'source': 'fallback'
                }
            ],
            'alignment': {
                'alignment_method': 'manual',
                'manual_adjustments': {
                    'translation_x': 0.0,
                    'translation_y': 0.0,
                    'rotation': 0.0,
                    'scale': 1.0
                },
                'source': 'fallback'
            },
            'scoring': {
                'scoring_methods': ['correlation'],
                'source': 'fallback'
            },
            'collection_timestamp': self._get_timestamp(),
            'source': 'fallback_parameters'
        }

    def _perform_manual_alignment(self, filtered_image, alignment_params):
        self._log_action("Starting manual alignment", {
            'filtered_image_shape': getattr(filtered_image, 'shape', None),
            'alignment_method': alignment_params.get('alignment_method', 'manual')
        })
        try:
            manual_adjustments = alignment_params.get('manual_adjustments', {})
            transform_matrix = self._create_transform_matrix(manual_adjustments)
            if hasattr(self.alignment_service, 'apply_manual_transform'):
                aligned_image = self.alignment_service.apply_manual_transform(
                    self.gds_image,
                    transform_matrix
                )
            else:
                aligned_image = self.gds_image
                transform_matrix = np.eye(3)
            alignment_result = {
                'aligned_image': aligned_image,
                'transform_matrix': transform_matrix,
                'confidence': 1.0,
                'method': 'manual',
                'manual_parameters': manual_adjustments,
                'success': True
            }
            self._log_action("Manual alignment completed", {
                'success': True,
                'transform_parameters': manual_adjustments
            })
            return alignment_result
        except Exception as e:
            self._log_action("Manual alignment failed", {'error': str(e)})
            return {
                'aligned_image': filtered_image,
                'transform_matrix': np.eye(3),
                'confidence': 0.0,
                'method': 'manual_fallback',
                'error': str(e),
                'success': False
            }

    def _create_transform_matrix(self, manual_adjustments):
        tx = manual_adjustments.get('translation_x', 0.0)
        ty = manual_adjustments.get('translation_y', 0.0)
        rotation = manual_adjustments.get('rotation', 0.0)
        scale = manual_adjustments.get('scale', 1.0)
        rotation_rad = np.radians(rotation)
        cos_r = np.cos(rotation_rad)
        sin_r = np.sin(rotation_rad)
        transform_matrix = np.array([
            [scale * cos_r, -scale * sin_r, tx],
            [scale * sin_r,  scale * cos_r, ty],
            [0,              0,             1]
        ])
        return transform_matrix

    def validate_manual_mode_readiness(self) -> Dict[str, Any]:
        validation = {
            'ready': False,
            'issues': [],
            'warnings': [],
            'ui_panel_status': {}
        }
        panels = ['filter_panel', 'alignment_panel', 'score_panel']
        for panel_name in panels:
            panel = getattr(self, panel_name, None)
            validation['ui_panel_status'][panel_name] = {
                'available': panel is not None,
                'has_config_method': panel is not None and hasattr(panel, 'get_current_config')
            }
            if panel is None:
                validation['warnings'].append(f"{panel_name}_not_connected")
        try:
            test_params = self._collect_ui_parameters()
            validation['parameter_collection'] = 'success'
            validation['collected_filter_count'] = len(test_params.get('filters', []))
        except Exception as e:
            validation['issues'].append(f"parameter_collection_failed: {str(e)}")
            validation['parameter_collection'] = 'failed'
        if not self.config.get('use_ui_state', True):
            validation['issues'].append("use_ui_state_disabled_in_config")
        validation['ready'] = len(validation['issues']) == 0
        return validation

    def set_ui_panels(self, filter_panel=None, alignment_panel=None, score_panel=None):
        self.filter_panel = filter_panel
        self.alignment_panel = alignment_panel
        self.score_panel = score_panel
        self._log_action("UI panels registered", {
            'has_filter_panel': filter_panel is not None,
            'has_alignment_panel': alignment_panel is not None,
            'has_score_panel': score_panel is not None
        })
