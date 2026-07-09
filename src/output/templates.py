from importlib import import_module
import sys

_module = import_module("templates")
sys.modules[__name__] = _module
