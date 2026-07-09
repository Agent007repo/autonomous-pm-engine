from importlib import import_module
import sys

_module = import_module("state")
sys.modules[__name__] = _module
