#!/usr/bin/env python
"""Task4 demo command; implementation lives in demo/task4.py."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))

from task4 import main


if __name__ == "__main__":
    main()
