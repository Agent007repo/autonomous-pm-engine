from importlib import import_module
import sys

_module = import_module("data_analyst_agent")
sys.modules[__name__] = _module
