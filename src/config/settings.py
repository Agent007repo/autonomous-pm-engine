from importlib import import_module
import sys

_module = import_module("settings")
sys.modules[__name__] = _module
