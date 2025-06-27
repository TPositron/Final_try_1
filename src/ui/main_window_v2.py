import sys
import cv2
import numpy as np
import os
from pathlib import Path
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QMenuBar, QMenu, QFileDialog, QMessageBox, QDockWidget,
                               QApplication, QStatusBar, QSplitter, QPushButton)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QAction, QIcon
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse


from src.image_analysis.ui.components.image_viewer import ImageViewer
from src.image_analysis.ui.components.filter_panel import FilterPanel
from src.image_analysis.ui.components.alignment_panel import AlignmentPanel
from src.image_analysis.ui.components.score_panel import ScorePanel
from src.image_analysis.services.file_service import FileManager
from src.image_analysis.services.image_processing_service import ImageProcessingService
from src.image_analysis.services.alignment_service import AlignmentService
from src.image_analysis.core.processors.overlay import OverlayRenderer
from src.image_analysis.core.models.gds_model import GDSModel


# Structure definitions for GDS extraction
STRUCTURES = {
    1: {'name': 'Circpol_T2', 'initial_bounds': (688.55, 5736.55, 760.55, 5807.1), 'layers': [14]},
    2: {'name': 'IP935Left_11', 'initial_bounds': (693.99, 6406.40, 723.59, 6428.96), 'layers': [1, 2]},
    3: {'name': 'IP935Left_14', 'initial_bounds': (980.959, 6025.959, 1001.770, 6044.979), 'layers': [1]},
    4: {'name': 'QC855GC_CROSS_Bottom', 'initial_bounds': (3730.00, 4700.99, 3756.00, 4760.00), 'layers': [1, 2]},
    5: {'name': 'QC935_46', 'initial_bounds': (7195.558, 5046.99, 7203.99, 5055.33964), 'layers': [1]}
}

class MainWindow(QMainWindow):
    def __init__(self):
        print("MainWindow constructor called")
        super().__init__()
        self.setWindowTitle("Image Analysis - SEM/GDS Alignment Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        self.file_manager = FileManager()
        # Removed GDS image generation on startup
        # User will load GDS/SEM files via the UI
        
        self.image_processing_service = ImageProcessingService()
        self.alignment_service = AlignmentService()
        
        self.current_sem_image = None
        self.current_structures = {}  # Initialize as empty dict
        self.current_structure_name = None
        self.current_gds_overlay = None
        self.current_alignment_result = None
        self.overlay_renderer = None
        
        self.setup_ui()
        self.setup_menu()
        self.connect_signals()
        self.setup_status_bar()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        self.image_viewer = ImageViewer()
        splitter.addWidget(self.image_viewer)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Add Load SEM Image button at the top of the right panel
        self.load_sem_btn = QPushButton("Load SEM Image")
        self.load_sem_btn.clicked.connect(self.load_sem_image)
        right_layout.addWidget(self.load_sem_btn)

        self.filter_panel = FilterPanel()
        self.alignment_panel = AlignmentPanel()
        self.score_panel = ScorePanel()

        # Build structure list for dropdown: 'Structure 1: Circpol_T2', etc.
        self.structure_display_names = {}
        for idx, struct in STRUCTURES.items():
            display_name = f"Structure {idx}: {struct['name']}"
            self.structure_display_names[display_name] = idx
        self.alignment_panel.structure_combo.clear()
        self.alignment_panel.structure_combo.addItem("Select Structure", "")
        for display_name in self.structure_display_names:
            self.alignment_panel.structure_combo.addItem(display_name, display_name)

        right_layout.addWidget(self.filter_panel)
        right_layout.addWidget(self.alignment_panel)
        right_layout.addWidget(self.score_panel)

        right_panel.setLayout(right_layout)
        splitter.addWidget(right_panel)
        splitter.setSizes([1000, 400])

    def setup_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        
        load_sem_action = QAction("Load SEM Image", self)
        load_sem_action.triggered.connect(self.load_sem_image)
        file_menu.addAction(load_sem_action)
        
        load_gds_action = QAction("Load GDS File", self)
        load_gds_action.triggered.connect(self.load_gds_file)
        file_menu.addAction(load_gds_action)
        
        file_menu.addSeparator()
        
        save_result_action = QAction("Save Results", self)
        save_result_action.triggered.connect(self.save_results)
        file_menu.addAction(save_result_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu("View")
        
        reset_view_action = QAction("Reset View", self)
        reset_view_action.triggered.connect(self.image_viewer.reset_view)
        view_menu.addAction(reset_view_action)
        
        tools_menu = menubar.addMenu("Tools")
        
        reset_alignment_action = QAction("Reset Alignment", self)
        reset_alignment_action.triggered.connect(self.reset_alignment)
        tools_menu.addAction(reset_alignment_action)
        
        auto_align_action = QAction("Auto Align", self)
        auto_align_action.triggered.connect(self.auto_align)
        tools_menu.addAction(auto_align_action)
        
    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def connect_signals(self):
        # Connect alignment panel signals
        self.alignment_panel.structure_selected.connect(self.on_structure_selected)
        self.alignment_panel.alignment_changed.connect(self.on_alignment_changed)
        self.alignment_panel.reset_requested.connect(self.reset_alignment)
        
        # Connect filter panel signals
        self.filter_panel.filter_changed.connect(self.on_filter_changed)
        self.filter_panel.apply_clicked.connect(self.on_filter_applied)
        self.filter_panel.reset_clicked.connect(self.on_reset_filters)
    
    def load_sem_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load SEM Image", str(self.file_manager.get_sem_dir()),
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg)")

        if file_path:
            try:
                sem_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if sem_array is None:
                    raise ValueError(f"Failed to load image: {file_path}")
                h, w = sem_array.shape[:2]
                crop_h, crop_w = 666, 1024
                # Always crop from the bottom center
                start_y = max(0, h - crop_h)
                start_x = max(0, (w - crop_w) // 2)
                cropped = sem_array[start_y:start_y+crop_h, start_x:start_x+crop_w]
                print(f"Cropped image shape: {cropped.shape}, min: {cropped.min()}, max: {cropped.max()}")
                self.current_sem_image = cropped
                self.image_viewer.set_sem_image(cropped)
                self.status_bar.showMessage(f"Loaded SEM image: {Path(file_path).name} (cropped to 1024x666 from bottom center)")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load SEM image: {str(e)}")
                
    def load_gds_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load GDS File", str(self.file_manager.get_gds_dir()),
            "GDS Files (*.gds *.gds2)")
        if file_path:
            # Extract structures from GDS file using new workflow
            self.current_structures = self.file_manager.extract_structures_from_gds(Path(file_path).name, structures=STRUCTURES)
            self.alignment_panel.set_structure_data({info['name']: self.current_structures[info['name']] for info in STRUCTURES.values() if info['name'] in self.current_structures})
            self.status_bar.showMessage(f"Loaded GDS structures from {file_path}")

    def on_structure_selected(self, display_name):
        """Handle structure selection from alignment panel"""
        if not display_name or display_name not in self.structure_display_names:
            print("No valid structure selected.")
            return
        idx = self.structure_display_names[display_name]
        struct = STRUCTURES[idx]
        structure_name = struct['name']
        bounds = struct['initial_bounds']
        layers = struct['layers']
        gds_files = self.file_manager.list_gds_files()
        if not gds_files:
            QMessageBox.warning(self, "No GDS File", "Please load a GDS file first.")
            return
        gds_filename = gds_files[0]  # Use the first loaded GDS file
        try:
            # Extract binary image for the selected structure
            images = self.file_manager.extract_structures_from_gds(gds_filename, {idx: struct})
            gds_mask = images[structure_name]
            self.current_gds_overlay = gds_mask
            print(f"Extracted GDS mask for {structure_name}, shape: {gds_mask.shape}, dtype: {gds_mask.dtype}, unique: {np.unique(gds_mask)}")
            # Ensure mask is 0/255 uint8
            if gds_mask.max() <= 1:
                gds_mask = (gds_mask * 255).astype(np.uint8)
            else:
                gds_mask = gds_mask.astype(np.uint8)
            # Convert to RGB if needed
            if len(gds_mask.shape) == 2:
                gds_mask_rgb = cv2.cvtColor(gds_mask, cv2.COLOR_GRAY2RGB)
            else:
                gds_mask_rgb = gds_mask
            print(f"Displaying GDS mask, shape: {gds_mask_rgb.shape}, dtype: {gds_mask_rgb.dtype}")
            self.image_viewer.set_gds_overlay(gds_mask_rgb)
            self.status_bar.showMessage(f"Loaded structure: {display_name}")
            if self.current_sem_image is not None:
                self.update_alignment_display()
        except Exception as e:
            print(f"Error extracting or displaying GDS structure: {e}")
            QMessageBox.critical(self, "Error", f"Failed to extract GDS structure: {e}")
            self.current_structure_name = None

    def on_alignment_changed(self, parameters):
        if self.current_sem_image is None or self.current_structures is None or self.current_structure_name is None:
            return
        try:
            # Use the new structure data format
            result = self.alignment_service.apply_transformations(
                self.current_sem_image,
                self.current_structures,
                self.current_structure_name,
                x_offset=parameters.get('x_offset', 0),
                y_offset=parameters.get('y_offset', 0),
                rotation=parameters.get('rotation', 0.0),
                scale=parameters.get('scale', 1.0),
                transparency=parameters.get('transparency', 70)
            )
            self.current_alignment_result = result
            self.update_alignment_display()
        except Exception as e:
            QMessageBox.critical(self, "Alignment Error", str(e))

    def update_alignment_display(self):
        if self.current_sem_image is None or self.current_alignment_result is None:
            return
        overlay = self.current_alignment_result.get('overlay_preview')
        if overlay is not None:
            self.image_viewer.set_image(overlay)

    def on_filter_applied(self, filter_name, parameters):
        if self.current_sem_image is None:
            return
            
        try:
            self.image_processing_service.apply_filter(filter_name, parameters)
            filtered_image = self.image_processing_service.get_current_image()
            
            self.image_viewer.set_sem_image(filtered_image)
            self.status_bar.showMessage(f"Applied filter: {filter_name}")
            
            if self.current_gds_overlay is not None:
                self.update_alignment_display()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply filter: {str(e)}")
            
    def on_filter_preview(self, filter_name, parameters):
        if self.current_sem_image is None:
            return
            
        try:
            preview_array = self.image_processing_service.preview_filter(filter_name, parameters)
            self.image_viewer.set_preview_image(preview_array)
            
        except Exception as e:
            self.status_bar.showMessage(f"Preview error: {str(e)}")
            
    def on_reset_filters(self):
        if self.current_sem_image is None:
            return
            
        try:
            self.image_processing_service.reset_to_original()
            original_image = self.image_processing_service.get_current_image()
            
            self.image_viewer.set_sem_image(original_image)
            self.status_bar.showMessage("Reset to original image")
            
            if self.current_gds_overlay is not None:
                self.update_alignment_display()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reset filters: {str(e)}")
            
    def reset_alignment(self):
        self.alignment_panel.reset_parameters()
        
        if self.current_sem_image is not None and self.current_gds_overlay is not None:
            self.update_alignment_display()
            
        self.status_bar.showMessage("Reset alignment parameters")
        
    def auto_align(self):
        if self.current_sem_image is None or self.current_gds_overlay is None:
            QMessageBox.warning(self, "Warning", "Please load both SEM image and GDS file first")
            return
            
        try:
            self.status_bar.showMessage("Performing auto-alignment...")
            QApplication.processEvents()
            
            current_image = self.image_processing_service.get_current_image()
            
            search_result = self.alignment_service.batch_alignment_search(
                current_image, self.current_gds_overlay)
            
            if search_result['best_result'] is not None:
                best_params = search_result['best_parameters']
                self.alignment_panel.set_parameters(best_params)
                
                self.current_alignment_result = search_result['best_result']
                self.image_viewer.set_alignment_result(self.current_alignment_result)
                
                self.score_panel.update_scores({
                    'alignment_score': search_result['best_score'],
                    'total_tested': search_result['total_tested']
                })
                
                self.status_bar.showMessage(f"Auto-alignment complete. Best score: {search_result['best_score']:.4f}")
                
            else:
                self.status_bar.showMessage("Auto-alignment failed to find good parameters")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto-alignment failed: {str(e)}")
            self.status_bar.showMessage("Auto-alignment failed")
            
    def calculate_scores(self):
        if self.current_alignment_result is None:
            QMessageBox.warning(self, "Warning", "No alignment result available for scoring")
            return
            
        try:
            current_image = self.image_processing_service.get_current_image()
            
            if hasattr(current_image, 'to_array'):
                sem_array = current_image.to_array()
            else:
                sem_array = current_image
                
            gds_array = self.current_alignment_result['transformed_gds']
            # We already have cv2 imported at the top
            # Getting the score values
            try:
                ssim_score = ssim(sem_array, gds_array, data_range=1.0)
                mse_score = mse(sem_array, gds_array)

                # Calculate edge detection for additional comparison
                sem_edges = cv2.Canny((sem_array * 255).astype('uint8'), 50, 150)
                gds_edges = cv2.Canny((gds_array * 255).astype('uint8'), 50, 150)
            except Exception as e:
                print(f"Error calculating similarity scores: {e}")
                ssim_score = 0
                mse_score = 1
                sem_edges = np.zeros_like(sem_array, dtype='uint8')
                gds_edges = np.zeros_like(gds_array, dtype='uint8')

            edge_overlap = (sem_edges & gds_edges).sum()
            total_edges = sem_edges.sum() + gds_edges.sum()
            edge_ratio = edge_overlap / max(total_edges, 1)

            scores = {
                'ssim': ssim_score,
                'mse': mse_score,
                'edge_overlap_ratio': edge_ratio,
                'alignment_score': self.current_alignment_result['alignment_score']
            }

            self.score_panel.update_scores(scores)
            self.status_bar.showMessage("Scores calculated")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate scores: {str(e)}")
            
    def save_results(self):
        if self.current_alignment_result is None:
            QMessageBox.warning(self, "Warning", "No results to save")
            return
            
        try:
            timestamp = QTimer().remainingTime()  
            result_dir = self.file_manager.create_result_subdir(f"alignment_result_{timestamp}")
            
            saved_files = self.alignment_service.save_alignment_result(
                self.current_alignment_result, result_dir)
            
            if hasattr(self.score_panel, 'get_current_scores'):
                scores = self.score_panel.get_current_scores()
                if scores:
                    self.file_manager.save_scores(scores, "alignment_scores", 
                                                subdir=result_dir.name)
            
            self.status_bar.showMessage(f"Results saved to: {result_dir}")
            QMessageBox.information(self, "Success", f"Results saved to:\n{result_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")
            
    def on_filter_changed(self, filter_name, params):
        print(f"Filter changed: {filter_name}, params: {params}")
        # TODO: Implement filter change logic

    def on_structure_selected(self, structure_idx):
        try:
            if structure_idx in STRUCTURES:
                structure = STRUCTURES[structure_idx]
                print(f"Loading structure {structure_idx}: {structure['name']}")
                
                # Load the structure overlay from the file saved by extract_all_structures
                structure_name = structure['name']
                structure_overlay = self.file_manager.load_structure_overlay(structure_name)
                if structure_overlay is not None:
                    self.current_gds_overlay = structure_overlay
                    self.image_viewer.set_gds_overlay(self.current_gds_overlay)
                    self.alignment_panel.set_gds_overlay(self.current_gds_overlay)
                    
                    if self.current_sem_image is not None:
                        self.update_alignment_display()
                    
                    self.status_bar.showMessage(f"Loaded structure overlay: {structure_name}")
                else:
                    self.status_bar.showMessage(f"Failed to load structure overlay: {structure_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load structure: {str(e)}")
            self.status_bar.showMessage(f"Failed to load structure: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())