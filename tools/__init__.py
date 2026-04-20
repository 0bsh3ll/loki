"""
LOKI Tools — Auto Discovery

On import, this package scans its own directory and imports every
.py file it finds. Each tool file uses @register_tool, so by the
time this __init__ finishes, all tools are in the registry.

Adding a new tool = adding one file here. Zero changes anywhere else.
"""

import importlib
import pkgutil
from pathlib import Path

package_dir = Path(__file__).parent

for module_info in pkgutil.iter_modules([str(package_dir)]):
    importlib.import_module(f".{module_info.name}", package=__name__)
