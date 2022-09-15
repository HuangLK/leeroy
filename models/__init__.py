from .model_base import Model
from .binary_classification import BinaryClassificationModel
from .multi_classification import MultiClassificationModel
from .multilabel_classification import MultiLabelClassificationModel
from .span_extraction import SpanExtractionModel

__all__ = [
    'create_model',
    'add_cmdline_args'
]

TASK2MODEL = {
    'binary_clf': BinaryClassificationModel,
    'mulclass_clf': MultiClassificationModel,
    'multilabel_clf': MultiLabelClassificationModel,
    'span': SpanExtractionModel,
}

def _get_model_cls(args):
    task_name = args.task.split('-')[0]
    if task_name not in TASK2MODEL:
        raise ValueError(f'task={args.task} is unvalid.')
    return TASK2MODEL[task_name]

def create_model(args) -> Model:
    model_cls = _get_model_cls(args)
    model = model_cls(args)
    return model


def add_cmdline_args(parser):
    args, _ = parser.parse_known_args()
    model_cls = _get_model_cls(args)
    model_cls.add_cmdline_args(parser)
