"""Binary classification model.
"""

from torchmetrics import Accuracy, Precision, Recall, F1Score
from .multiclass_classification import MultiClassClassificationModel

from models.multiclass_classification import MultiClassClassificationModel


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

    def configure_metrics(self) -> None:
        # 默认二分类任务为正负例分类，计算准召只关注正样本（即label=1）
        nclass = 1
        prec = Precision(num_classes=nclass, average="macro", multiclass=False)
        recall = Recall(num_classes=nclass, average="macro", multiclass=False)
        f1 = F1Score(num_classes=nclass, average="macro", multiclass=False)
        acc = Accuracy()
        self.metrics = {"precision": prec, "recall": recall, "f1": f1, "accuracy": acc}
