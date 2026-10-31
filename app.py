"""
Root launcher for Readability and Text Complexity Analyzer.
Runs Readability_Text_Complexity_Analyzer/app.py directly.
"""
import runpy
import sys
from pathlib import Path

app_path = Path(__file__).resolve().parent / "Readability_Text_Complexity_Analyzer" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
