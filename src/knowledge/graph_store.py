from importlib import import_module
import sys

_module = import_module("graph_store")
sys.modules[__name__] = _module
