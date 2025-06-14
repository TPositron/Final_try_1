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
    
    # Change to the script directory
    os.chdir(script_dir)
    
    try:        # Import and run the main window directly
        from src.image_analysis.ui.main_window import MainWindow
        from PySide6.QtWidgets import QApplication
        
        print("Creating QApplication...")
        app = QApplication(sys.argv)
        print("Creating MainWindow...")
        window = MainWindow()
        print("Showing MainWindow...")
        window.show()
        print("Entering event loop...")
        return app.exec_()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    sys.exit(main())
