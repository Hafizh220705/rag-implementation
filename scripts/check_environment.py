from __future__ import annotations
import importlib.util
import platform
import sys
from pathlib import Path

def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"OS: {platform.platform()}")
    print(f"Project root: {Path(__file__).resolve().parents[1]}")
    for module in ("fitz", "requests", "sentence_transformers", "pytest"):
        available = importlib.util.find_spec(module) is not None
        print(f"{module}: {'installed' if available else 'NOT installed'}")
    py_ok = (3, 11) <= sys.version_info[:2] < (3, 13)
    print(f"Python supported: {py_ok}")
    return 0 if py_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
