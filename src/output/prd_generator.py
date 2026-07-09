from importlib import import_module
import sys

_module = import_module("prd_generator")
sys.modules[__name__] = _module
