from PySide6.QtCore import QObject, Signal
from .pipeline_results import PipelineResults
from .pipeline_utils import *
from src.services.filters.image_processing_service import ImageProcessingService
from src.services.simple_alignment_service import AlignmentService
from src.services.simple_scoring_service import ScoringService
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple, List
import time
import os

class ProcessingPipelineBase(QObject):
    """
    Base pipeline: shared state, signals, result aggregation, and utility methods.
    Inherit for mode-specific logic.
    """
    # Progress signals
    stage_started = Signal(str)
    stage_progress = Signal(str, int)
    stage_completed = Signal(str, dict)
    pipeline_completed = Signal(dict)
    pipeline_error = Signal(str)
    pipeline_started = Signal(str, dict)
    # Detailed progress signals
    filter_trial_started = Signal(str, int, int)
    filter_trial_completed = Signal(str, dict)
    filter_sequence_progress = Signal(int, int, str)
    alignment_method_started = Signal(str)
    alignment_method_completed = Signal(str, bool, float)
    alignment_fallback = Signal(str, str)
    alignment_progress_detail = Signal(str, str, int)
    scoring_method_started = Signal(str)
    scoring_method_completed = Signal(str, float)
    scoring_progress_detail = Signal(int, int, str, float)
    # Real-time status signals
    status_message = Signal(str)
    processing_statistics = Signal(dict)
    intermediate_result_ready = Signal(str, object)

    def __init__(self):
        super().__init__()
        self.image_processor = ImageProcessingService()
        self.alignment_service = AlignmentService()
        self.scoring_service = ScoringService()
        self.sem_image = None
        self.gds_image = None
        self.mode = 'manual'
        self.is_initialized = False
        self.is_running = False
        self.current_stage = None
        self.stages = ['filtering', 'alignment', 'scoring']
        self.stage_results = {}
        self.pipeline_log = []
        self.pipeline_results = PipelineResults()
        self.final_results = {
            'filter_chain': [],
            'transform_matrix': None,
            'scoring_results': {},
            'intermediate_images': {},
            'metadata': {}
        }
        self.config = get_default_config()
        self.mode_configs = get_mode_configs()
        self._pipeline_start_time = None
        self._connect_service_signals()

    def _connect_service_signals(self):
        if hasattr(self.image_processor, 'filter_applied'):
            self.image_processor.filter_applied.connect(self._on_filter_applied)
        if hasattr(self.image_processor, 'processing_progress'):
            self.image_processor.processing_progress.connect(self._on_processing_progress)
        if hasattr(self.alignment_service, 'alignment_completed'):
            self.alignment_service.alignment_completed.connect(self._on_alignment_completed)
        if hasattr(self.alignment_service, 'alignment_progress'):
            self.alignment_service.alignment_progress.connect(self._on_alignment_progress)
        if hasattr(self.scoring_service, 'score_calculated'):
            self.scoring_service.score_calculated.connect(self._on_score_calculated)
        if hasattr(self.scoring_service, 'scoring_progress'):
            self.scoring_service.scoring_progress.connect(self._on_scoring_progress)

    def initialize(self, sem_image, gds_image, mode='manual'):
        try:
            self.sem_image = deepcopy(sem_image)
            self.gds_image = deepcopy(gds_image)
            self.mode = mode
            self.stage_results = {}
            self.pipeline_log = []
            self.final_results = {
                'filter_chain': [],
                'transform_matrix': None,
                'scoring_results': {},
                'intermediate_images': {},
                'metadata': {
                    'mode': mode,
                    'timestamp': None,
                    'input_shapes': {
                        'sem': getattr(sem_image, 'shape', None),
                        'gds': getattr(gds_image, 'shape', None)
                    }
                }
            }
            self.image_processor.load_image(sem_image)
            self.pipeline_results.set_original_images(sem_image, gds_image)
            self.is_initialized = True
            self.is_running = False
            self._log_action("Pipeline initialized", {
                'mode': mode,
                'sem_shape': getattr(sem_image, 'shape', None),
                'gds_shape': getattr(gds_image, 'shape', None)
            })
        except Exception as e:
            self.pipeline_error.emit(f"Pipeline initialization failed: {str(e)}")
            raise

    def run(self, custom_config=None):
        if not self.is_initialized:
            self.pipeline_error.emit("Pipeline not initialized. Call initialize() first.")
            return
        if self.is_running:
            self.pipeline_error.emit("Pipeline is already running.")
            return
        try:
            self.is_running = True
            self._pipeline_start_time = time.time()
            if custom_config:
                self.config.update(custom_config)
            self.pipeline_started.emit(self.mode, self.config.copy())
            self.status_message.emit(f"Starting {self.mode} processing pipeline...")
            self._log_action("Pipeline execution started", {'config': self.config.copy()})
            stage_input = self.sem_image
            stage_output = None
            for i, stage in enumerate(self.stages):
                self.current_stage = stage
                self.stage_started.emit(stage)
                self._emit_processing_statistics(stage)
                self._log_stage_input(stage, stage_input, i)
                if stage == 'filtering':
                    stage_output = self._execute_filtering_stage(stage_input)
                elif stage == 'alignment':
                    stage_output = self._execute_alignment_stage(stage_input)
                elif stage == 'scoring':
                    stage_output = self._execute_scoring_stage(stage_input)
                self._log_stage_output(stage, stage_output, i)
                verification = self._verify_stage_output_integrity(stage, stage_output)
                if not verification['output_valid']:
                    error_msg = f"Stage '{stage}' output verification failed: {verification['missing_components']}"
                    self.pipeline_error.emit(error_msg)
                    raise ValueError(error_msg)
                stage_params = self.stage_results.get(stage, {}).get('parameters_used', {})
                self._enhance_reproducibility_logging(stage, stage_params, stage_output)
                self._emit_intermediate_result(stage, stage_output)
                self._emit_processing_statistics(stage)
                stage_input = stage_output
                if self.mode == 'automatic' and hasattr(self, '_should_early_exit') and self._should_early_exit():
                    self._log_action("Early exit triggered", {'stage': stage, 'stage_index': i})
                    break
            self._aggregate_results()
            self.pipeline_completed.emit(self.final_results)
            self._log_action("Pipeline execution completed", {'success': True, 'final_output': type(stage_output).__name__})
        except Exception as e:
            error_msg = f"Pipeline execution failed at stage '{self.current_stage}': {str(e)}"
            self.pipeline_error.emit(error_msg)
            self._log_action("Pipeline execution failed", {'error': str(e), 'stage': self.current_stage})
        finally:
            self.is_running = False
            self.current_stage = None

    def load_config(self, path: str):
        """
        Load pipeline configuration from a file (JSON/YAML) and update pipeline config.
        Emits pipeline_error if loading or validation fails.
        """
        try:
            config = load_config_from_file(path)
            valid, missing = validate_config(config, required_keys=list(self.config.keys()))
            if not valid:
                self.pipeline_error.emit(f"Config missing required keys: {missing}")
                return False
            self.config.update(config)
            self._log_action("Config loaded from file", {'path': path, 'keys': list(config.keys())})
            return True
        except Exception as e:
            self.pipeline_error.emit(f"Failed to load config: {str(e)}")
            return False

    def save_config(self, path: str):
        """
        Save current pipeline configuration to a file (JSON/YAML).
        Emits pipeline_error if saving fails.
        """
        try:
            save_config_to_file(self.config, path)
            self._log_action("Config saved to file", {'path': path})
            return True
        except Exception as e:
            self.pipeline_error.emit(f"Failed to save config: {str(e)}")
            return False

    def validate_current_config(self) -> Tuple[bool, list]:
        """
        Validate current pipeline configuration. Returns (True, []) if valid, else (False, [missing_keys]).
        """
        return validate_config(self.config, required_keys=list(get_default_config().keys()))

    def export_results(self, out_dir: str) -> dict:
        """
        Export all results: JSON report, overlay, stage images, and transform matrix.
        Args:
            out_dir: Output directory for all exports
        Returns:
            Dict with export status and file paths
        """
        os.makedirs(out_dir, exist_ok=True)
        status = {}
        # Export JSON report
        json_path = os.path.join(out_dir, 'pipeline_results.json')
        status['json_report'] = self.pipeline_results.export_json_report(json_path)
        # Export overlay image
        overlay_path = os.path.join(out_dir, 'overlay.png')
        status['overlay'] = self.pipeline_results.export_overlay_image(overlay_path)
        # Export stage images
        status['stage_images'] = self.pipeline_results.export_stage_images(out_dir)
        # Export transform matrix
        matrix_path = os.path.join(out_dir, 'transform_matrix.csv')
        status['transform_matrix'] = self.pipeline_results.export_transform_matrix(matrix_path)
        self._log_action("Results exported", status)
        return status

    def save_pipeline_state(self, path: str) -> bool:
        """
        Save the entire pipeline state (results, log, config) to a JSON file for session persistence.
        Args:
            path: Path to save the state file
        Returns:
            True if successful, False otherwise
        """
        import json
        try:
            state = {
                'results': self.pipeline_results.get_results(),
                'log': self.pipeline_log,
                'config': self.config,
                'mode': self.mode,
                'stages': self.stages,
                'stage_results': self.stage_results
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)
            self._log_action("Pipeline state saved", {'path': path})
            return True
        except Exception as e:
            self.pipeline_error.emit(f"Failed to save pipeline state: {str(e)}")
            return False

    def load_pipeline_state(self, path: str) -> bool:
        """
        Load pipeline state (results, log, config) from a JSON file.
        Args:
            path: Path to load the state file
        Returns:
            True if successful, False otherwise
        """
        import json
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            # Restore state
            self.pipeline_results.results = state.get('results', {})
            self.pipeline_log = state.get('log', [])
            self.config = state.get('config', get_default_config())
            self.mode = state.get('mode', 'manual')
            self.stages = state.get('stages', ['filtering', 'alignment', 'scoring'])
            self.stage_results = state.get('stage_results', {})
            self._log_action("Pipeline state loaded", {'path': path})
            return True
        except Exception as e:
            self.pipeline_error.emit(f"Failed to load pipeline state: {str(e)}")
            return False

    def validate_pipeline(self) -> dict:
        """
        Validate the pipeline state, configuration, and results for completeness and consistency.
        Returns:
            dict: Validation report with status, issues, and warnings
        """
        report = {
            'config_valid': True,
            'missing_config_keys': [],
            'results_valid': True,
            'missing_results': [],
            'stage_results_valid': True,
            'missing_stage_results': [],
            'warnings': [],
            'status': 'ok'
        }
        # Validate config
        valid, missing = validate_config(self.config, required_keys=list(get_default_config().keys()))
        if not valid:
            report['config_valid'] = False
            report['missing_config_keys'] = missing
            report['status'] = 'error'
        # Validate results
        results = self.pipeline_results.get_results()
        required_result_keys = ['filter_chain', 'transform_matrix', 'scoring_results']
        missing_results = [k for k in required_result_keys if k not in results or results[k] is None]
        if missing_results:
            report['results_valid'] = False
            report['missing_results'] = missing_results
            report['status'] = 'error'
        # Validate stage results
        stage_results = results.get('stage_results', {})
        for stage in ['filtering', 'alignment', 'scoring']:
            if stage not in stage_results or not stage_results[stage]:
                report['stage_results_valid'] = False
                report['missing_stage_results'].append(stage)
                report['status'] = 'error'
        # Add warnings for early exit or incomplete pipeline
        if results.get('metadata', {}).get('early_exit', False):
            report['warnings'].append('Pipeline exited early before all stages completed.')
        if report['status'] == 'ok':
            report['status'] = 'valid'
        return report

    def report_validation_errors(self, report: dict):
        """
        Emit pipeline_error signals for any validation issues found in the report.
        Args:
            report: Validation report from validate_pipeline()
        """
        if not report.get('config_valid', True):
            self.pipeline_error.emit(f"Missing config keys: {report.get('missing_config_keys')}")
        if not report.get('results_valid', True):
            self.pipeline_error.emit(f"Missing result keys: {report.get('missing_results')}")
        if not report.get('stage_results_valid', True):
            self.pipeline_error.emit(f"Missing stage results: {report.get('missing_stage_results')}")
        for warning in report.get('warnings', []):
            self.status_message.emit(f"Validation warning: {warning}")
