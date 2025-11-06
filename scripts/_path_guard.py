"""
tiny helper to ensure "import app" works when run from anywhere
"""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, ".."))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

