from importlib import import_module
import sys

_module = import_module("nodes")
sys.modules[__name__] = _module
