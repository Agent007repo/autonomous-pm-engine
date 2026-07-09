from importlib import import_module
import sys

_module = import_module("semantic_chunker")
sys.modules[__name__] = _module
