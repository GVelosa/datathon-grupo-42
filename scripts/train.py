"""DVC train stage: executa src/models/train.py com o projeto root no sys.path."""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

runpy.run_path("src/models/train.py", run_name="__main__")
