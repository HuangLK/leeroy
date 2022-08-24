from models.model_base import Model
from models.binary_classification import BinaryClassificationModel
from models.multiclass_classification import MultiClassClassificationModel
from models.multilabel_classification import MultiLabelClassificationModel

__all__ = [
    'create_model',
    'add_cmdline_args'
]

TASK2MODEL = {
    'binary_clf': BinaryClassificationModel,
    'mulclass_clf': MultiClassClassificationModel,
    'multilabel_clf': MultiLabelClassificationModel,
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
