from importlib import import_module
import sys

_module = import_module("vector_store")
sys.modules[__name__] = _module
