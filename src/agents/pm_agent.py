from importlib import import_module
import sys

_module = import_module("pm_agent")
sys.modules[__name__] = _module
