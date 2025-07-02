from .pipeline_base import ProcessingPipelineBase
import numpy as np

class AutomaticProcessingPipeline(ProcessingPipelineBase):
    """
    Processing pipeline for automatic mode. Handles automatic filter sequence, alignment, and early exit logic.
    Inherits shared state, signals, and result aggregation from PipelineBase.
    """
    def _execute_automatic_mode(self, input_image):
        return self._apply_automatic_filters_with_progress(input_image)

    def _apply_automatic_filters_with_progress(self, input_image):
        filter_sequence = self.config.get('filter_sequence', ['clahe', 'total_variation'])
        max_trials = self.config.get('max_filter_trials', 5)
        self._log_action("Starting automatic filter sequence", {
            'filter_sequence': filter_sequence,
            'max_trials': max_trials,
            'input_shape': getattr(input_image, 'shape', None)
        })
        current_image = input_image
        filter_chain = []
        total_filters = len(filter_sequence)
        for i, filter_name in enumerate(filter_sequence):
            self.filter_trial_started.emit(filter_name, i + 1, total_filters)
            self.filter_sequence_progress.emit(i + 1, total_filters, filter_name)
            self.status_message.emit(f"Applying filter {i + 1}/{total_filters}: {filter_name}")
            progress = int((i / total_filters) * 100)
            self.stage_progress.emit('filtering', progress)
            try:
                filter_params = self._get_default_filter_params(filter_name)
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
                            'trial_number': i + 1
                        }
                        filter_chain.append(filter_entry)
                        self.filter_trial_completed.emit(filter_name, filter_entry)
                        self._log_action(f"Filter applied successfully: {filter_name}", filter_entry)
                    else:
                        self._log_action(f"Filter failed: {filter_name}", {'trial_number': i + 1})
                else:
                    self._log_action(f"Filter method unavailable: {filter_name}", {'fallback': True})
            except Exception as e:
                error_msg = f"Filter {filter_name} failed: {str(e)}"
                self._log_action(error_msg, {'trial_number': i + 1, 'error': str(e)})
        self.stage_progress.emit('filtering', 100)
        self._log_action("Automatic filter sequence completed", {
            'filters_applied': len(filter_chain),
            'total_attempted': len(filter_sequence),
            'final_image_shape': getattr(current_image, 'shape', None)
        })
        return current_image, filter_chain

    def _get_default_filter_params(self, filter_name):
        defaults = {
            'clahe': {'clip_limit': 2.0, 'tile_grid_size': 8},
            'total_variation': {'weight': 0.1, 'iterations': 10},
            'gaussian': {'sigma': 1.0},
            'threshold': {'threshold_value': 127, 'method': 'binary'},
            'canny': {'low_threshold': 50, 'high_threshold': 150}
        }
        return defaults.get(filter_name, {})

    def _perform_automatic_alignment_with_progress(self, filtered_image, stage_parameters):
        alignment_method = stage_parameters.get('alignment_method', 'orb_ransac')
        self._log_action("Starting automatic alignment", {
            'method': alignment_method,
            'filtered_image_shape': getattr(filtered_image, 'shape', None),
            'gds_image_shape': getattr(self.gds_image, 'shape', None)
        })
        self.alignment_method_started.emit(alignment_method)
        self.status_message.emit(f"Starting alignment using {alignment_method}")
        self.alignment_progress_detail.emit(alignment_method, "Initializing alignment", 10)
        self.stage_progress.emit('alignment', 10)
        alignment_result = None
        success = False
        confidence = 0.0
        try:
            if hasattr(self.alignment_service, 'align_images'):
                self.alignment_progress_detail.emit(alignment_method, "Processing keypoints", 30)
                self.status_message.emit(f"Processing keypoints with {alignment_method}")
                self.stage_progress.emit('alignment', 30)
                alignment_result = self.alignment_service.align_images(
                    filtered_image, 
                    self.gds_image, 
                    method=alignment_method
                )
                self.alignment_progress_detail.emit(alignment_method, "Computing transform matrix", 70)
                self.status_message.emit("Computing transformation matrix")
                self.stage_progress.emit('alignment', 70)
                if alignment_result and 'transform_matrix' in alignment_result:
                    success = True
                    confidence = alignment_result.get('confidence', 0.0)
                    self.alignment_method_completed.emit(alignment_method, success, confidence)
                    self._log_action(f"Alignment successful: {alignment_method}", {
                        'confidence': confidence,
                        'transform_matrix_shape': getattr(alignment_result.get('transform_matrix'), 'shape', None)
                    })
                else:
                    fallback_method = 'template_matching'
                    self.alignment_fallback.emit(alignment_method, fallback_method)
                    self._log_action(f"Primary alignment failed, trying fallback", {
                        'primary_method': alignment_method,
                        'fallback_method': fallback_method
                    })
                    self.stage_progress.emit('alignment', 80)
                    if hasattr(self.alignment_service, 'template_match'):
                        fallback_result = self.alignment_service.template_match(
                            filtered_image, self.gds_image
                        )
                        if fallback_result:
                            alignment_result = fallback_result
                            success = True
                            confidence = fallback_result.get('confidence', 0.3)
                            self.alignment_method_completed.emit(fallback_method, success, confidence)
                            self._log_action(f"Fallback alignment successful: {fallback_method}", {
                                'confidence': confidence
                            })
                        else:
                            self.alignment_method_completed.emit(fallback_method, False, 0.0)
            else:
                self._log_action("Alignment service method unavailable, creating default result", {})
                alignment_result = {
                    'transform_matrix': np.eye(3),
                    'aligned_image': filtered_image,
                    'confidence': 0.1,
                    'method': 'fallback_identity'
                }
                success = True
                confidence = 0.1
        except Exception as e:
            error_msg = f"Alignment failed: {str(e)}"
            self._log_action(error_msg, {'method': alignment_method, 'error': str(e)})
            self.alignment_method_completed.emit(alignment_method, False, 0.0)
            alignment_result = {
                'transform_matrix': np.eye(3),
                'aligned_image': filtered_image,
                'confidence': 0.0,
                'method': 'error_fallback',
                'error': str(e)
            }
            success = False
        self.stage_progress.emit('alignment', 100)
        self._log_action("Automatic alignment completed", {
            'success': success,
            'confidence': confidence,
            'final_method': alignment_result.get('method', alignment_method) if alignment_result else 'unknown'
        })
        return alignment_result

    def _should_early_exit(self):
        if self.mode != 'automatic':
            return False
        threshold = self.config.get('early_exit_threshold', 0.3)
        if 'alignment' in self.stage_results:
            confidence = self.stage_results['alignment'].get('confidence', 0.0)
            if confidence < threshold:
                self._log_action("Early exit triggered - alignment confidence too low", {
                    'confidence': confidence,
                    'threshold': threshold
                })
                return True
        if 'scoring' in self.stage_results:
            best_score = self.stage_results['scoring'].get('best_score', 0.0)
            if best_score < threshold:
                self._log_action("Early exit triggered - scoring too low", {
                    'best_score': best_score,
                    'threshold': threshold
                })
                return True
        if 'filtering' in self.stage_results:
            filter_chain = self.stage_results['filtering'].get('filter_chain', [])
            failed_filters = [f for f in filter_chain if not f.get('success', False)]
            if len(failed_filters) > len(filter_chain) / 2:
                self._log_action("Early exit triggered - too many filter failures", {
                    'failed_filters': len(failed_filters),
                    'total_filters': len(filter_chain)
                })
                return True
        return False
