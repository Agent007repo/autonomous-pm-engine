from importlib import import_module
import sys

_module = import_module("document_loader")
sys.modules[__name__] = _module
