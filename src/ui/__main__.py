#!/usr/bin/env python3
"""
Entry point for running the main window as a module.
"""
import sys
from PySide6.QtWidgets import QApplication
from .main_window_v2 import MainWindow

def main():
    print("Creating QApplication...")
    app = QApplication(sys.argv)
    print("Creating MainWindow...")
    window = MainWindow()
    print("Showing MainWindow...")
    window.show()
    print("Entering event loop...")
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
