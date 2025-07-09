"""
Help Dialog - Application Help and User Guide Display

This module provides a help dialog that displays user guidance and instructions
for using the SEM/GDS alignment application.

Main Class:
- HelpDialog: Qt dialog for displaying help content

Key Features:
- HTML-formatted help content display
- Instructions for all major application features
- User-friendly dialog interface
- Integration with main application help menu

Dependencies:
- Uses: PySide6.QtWidgets (QDialog, QVBoxLayout, QTextBrowser, QDialogButtonBox)
- Called by: ui/main_window.py (help menu action)

Help Content:
- SEM image loading instructions
- GDS file loading instructions
- Mode switcher usage
- Filter panel operations
- Alignment panel operations
- Score panel usage
- Processing pipeline instructions
- Settings and export guidance
"""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QDialogButtonBox
import os

HELP_TEXT = """
<h2>SEM/GDS Comparison Tool - Help</h2>
<ul>
<li><b>Open SEM Image:</b> Use the File menu or file selector to load a SEM image.</li>
<li><b>Open GDS File:</b> Use the File menu or file selector to load a GDS file.</li>
<li><b>Mode Switcher:</b> Choose Manual or Automatic mode for processing.</li>
<li><b>Filter Panel:</b> Select and adjust filters for SEM images.</li>
<li><b>Alignment Panel:</b> Adjust or run automatic alignment between SEM and GDS.</li>
<li><b>Score Panel:</b> View and export scoring results.</li>
<li><b>Start Processing:</b> Use the Tools menu or F5 to run the full pipeline.</li>
<li><b>Settings:</b> Change preferences and theme in the Settings menu.</li>
<li><b>Export:</b> Save results and images from the File menu.</li>
</ul>
<p>For more information, see the user guide in the documentation folder.</p>
"""

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help")
        layout = QVBoxLayout(self)
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(HELP_TEXT)
        layout.addWidget(self.text_browser)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
