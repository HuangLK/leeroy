"""Binary classification model.
"""

from torchmetrics import Accuracy, Precision, Recall, F1Score
from .multiclass_classification import MultiClassClassificationModel


class BinaryClassificationModel(MultiClassClassificationModel):
    """Model for the binary classification task.
    """
    @classmethod
    def add_cmdline_args(cls, parser):
        """Add command line arguments."""
        group = super().add_cmdline_args(parser)
        return group

    def __init__(self, args) -> None:
        super().__init__(args)
        self.num_classes = 2

    def _configure_metrics(self):
        # 默认二分类任务为正负例分类，计算准召只关注正样本（即label=1）
        metrics = {}
        for stage in ('val', 'test'):
            metrics[f'{stage}_precision'] = Precision(num_classes=1, average="macro", multiclass=False)
            metrics[f'{stage}_recall'] = Recall(num_classes=1, average="macro", multiclass=False)
            metrics[f'{stage}_f1'] = F1Score(num_classes=1, average="macro", multiclass=False)
            metrics[f'{stage}_accuracy'] = Accuracy()
        return metrics
