# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:08:10 2026

@author: pmarques

Auto-builds the material registry from every *.py file in this folder
that exposes a module-level `DATA` FrozenDict.

To add a new material, simply drop a new file (e.g. Copper.py) into this
folder with a `DATA = FrozenDict({...})` at the module level — it will be
picked up automatically on the next import.
"""

#%% Import packages
import importlib
import pkgutil
from flax.core import FrozenDict

#%% Auto-discover all submodules and collect their DATA attribute

MATERIALS: dict[str, FrozenDict] = {}

for _info in pkgutil.iter_modules(__path__):
    _mod = importlib.import_module(f"{__name__}.{_info.name}")
    if hasattr(_mod, "DATA") and isinstance(_mod.DATA, FrozenDict):
        MATERIALS[_info.name] = _mod.DATA