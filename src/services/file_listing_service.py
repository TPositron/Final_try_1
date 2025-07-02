"""Service for scanning and listing available SEM and GDS files."""

from pathlib import Path
from typing import List, Dict
from PySide6.QtCore import QObject, Signal

from src.core.utils import get_data_path, list_files_by_extension


class FileListingService(QObject):
    """Service for discovering and listing available data files."""
    
    # Signals
    files_updated = Signal(dict)  # Emitted when file list is updated
    
    def __init__(self):
        super().__init__()
        self._sem_files = []
        self._gds_files = []
    
    def scan_data_directory(self) -> Dict[str, List[Path]]:
        """
        Scan the Data directory for SEM and GDS files.
        
        Returns:
            Dictionary with 'sem' and 'gds' keys containing file lists
        """
        data_path = get_data_path()
        
        # Scan for SEM files (TIFF and PNG)
        sem_path = data_path / "SEM"
        self._sem_files = []
        if sem_path.exists():
            self._sem_files.extend(list_files_by_extension(sem_path, ".tif"))
            self._sem_files.extend(list_files_by_extension(sem_path, ".tiff"))
            self._sem_files.extend(list_files_by_extension(sem_path, ".png"))
        
        # Scan for GDS files
        gds_path = data_path / "GDS"
        self._gds_files = []
        if gds_path.exists():
            self._gds_files.extend(list_files_by_extension(gds_path, ".gds"))
        
        # Sort files by name
        self._sem_files.sort(key=lambda x: x.name)
        self._gds_files.sort(key=lambda x: x.name)
        
        files_dict = {
            'sem': self._sem_files,
            'gds': self._gds_files
        }
        
        self.files_updated.emit(files_dict)
        return files_dict
    
    def get_sem_files(self) -> List[Path]:
        """Get the list of available SEM files."""
        return self._sem_files.copy()
    
    def get_gds_files(self) -> List[Path]:
        """Get the list of available GDS files."""
        return self._gds_files.copy()
    
    def refresh(self) -> None:
        """Refresh the file lists by rescanning the data directory."""
        self.scan_data_directory()
    
    def get_file_info(self, file_path: Path) -> Dict:
        """
        Get information about a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file information
        """
        if not file_path.exists():
            return {}
        
        stat = file_path.stat()
        
        return {
            'name': file_path.name,
            'path': str(file_path),
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'extension': file_path.suffix.lower(),
            'type': self._get_file_type(file_path)
        }
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine the file type based on path and extension."""
        extension = file_path.suffix.lower()
        
        if extension == '.gds':
            return 'GDS Layout'
        elif extension in ['.tif', '.tiff']:
            return 'SEM Image (TIFF)'
        elif extension == '.png':
            return 'SEM Image (PNG)'
        else:
            return 'Unknown'
