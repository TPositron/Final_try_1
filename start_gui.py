#!/usr/bin/env python
"""
Platform launcher for Image Analysis GUI
"""
import sys
import os

def main():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to Python path
    sys.path.insert(0, script_dir)
    # Add the src directory to Python path for package imports
    src_dir = os.path.join(script_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    # Change to the script directory
    os.chdir(script_dir)
    
    try:
        # Import and run the unified main window
        from src.ui.main_window import MainWindow
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
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
