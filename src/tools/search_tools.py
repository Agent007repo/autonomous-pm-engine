from importlib import import_module
import sys

_module = import_module("search_tools")
sys.modules[__name__] = _module
