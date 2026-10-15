import sys
from pathlib import Path


ENGINE_BUILD_DIR = (
    Path(__file__).resolve().parents[2]
    / "engine"
    / "build"
)

if str(ENGINE_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_BUILD_DIR))


import alpha_engine_cpp


__all__ = ["alpha_engine_cpp"]