#!/usr/bin/env python3
"""
Standalone Dating Show Runner
Run with: python run_dating_show_standalone.py
"""

import sys
from pathlib import Path

# Add dating_show to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "dating_show"))

from dating_show.standalone_simulation import main

if __name__ == "__main__":
    exit(main())