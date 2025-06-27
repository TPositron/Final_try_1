#!/usr/bin/env python3
"""
Cross-platform launcher for Image Analysis GUI
"""
import sys
import subprocess
import os

def main():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to Python path
    sys.path.insert(0, script_dir)
    # Add src to sys.path for module resolution
    src_dir = os.path.join(script_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    # Ensure src/core is also added for deeper imports
    core_dir = os.path.join(src_dir, 'core')
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    
    # Change to the script directory
    os.chdir(script_dir)
    
    try:        # Import and run the main window directly
        from ui.main_window import MainWindow
        from PySide6.QtWidgets import QApplication
        
        print("Creating QApplication...")
        app = QApplication(sys.argv)
        print("Creating MainWindow...")
        window = MainWindow()
        print("Showing MainWindow...")
        window.show()
        print("Entering event loop...")
        return app.exec()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sys.exit(main())
