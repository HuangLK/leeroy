import os
import pkgutil
import inspect
import importlib

from .model_base import Model

__all__ = [
    'create_model',
    'add_cmdline_args'
]

_TASK2MODEL = {}
for mod_name in list(module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)])):
    module = importlib.import_module(f'.{mod_name}', package=__package__)
    for _, cls in inspect.getmembers(module, lambda c: inspect.isclass(c) and issubclass(c, Model) and c != Model):
        _TASK2MODEL[cls.task] = cls


def _get_model_cls(args):
    task_name = args.task.split('-')[0]
    if task_name not in _TASK2MODEL:
        raise ValueError(f'task={args.task} is unvalid.')
    return _TASK2MODEL[task_name]


def create_model(args) -> Model:
    model_cls = _get_model_cls(args)
    model = model_cls(args)
    return model


def add_cmdline_args(parser):
    args, _ = parser.parse_known_args()
    model_cls = _get_model_cls(args)
    model_cls.add_cmdline_args(parser)
