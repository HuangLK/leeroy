

import torch
from transformers import Adafactor, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup


def get_optimizer_scheduler(model, optimizer, scheduler, lr, weight_decay, warmup_steps, schedule_steps):
    """Configure optimizers and learning-rate schedulers.

    Args:
        model (_type_): The model instance.
        optimizer (str): The optimizer.
        scheduler (str): The learning rate scheduler.
        lr (float): The learning rate.
        weight_decay (float): The weight decay rate.
        warmup_steps (int): The number of warmup steps.
        schedule_steps (int): The number of learning rate schedule steps.

    Returns:
        dict: A optimizer configuration dictionary including the key 'optimizer' and 'lr_scheduler'.
    """
    without_decay_names = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in without_decay_names)],
            "weight_decay": weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in without_decay_names)],
            "weight_decay": 0
        }
    ]

    if optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(optimizer_parameters, lr=lr)
    elif optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=lr)
    elif optimizer.lower() == "adafactor":
        optimizer = Adafactor(
            optimizer_parameters,
            lr=lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        )
    if scheduler.lower() == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler.lower() == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=schedule_steps
        )
    elif scheduler.lower() == "noam":
        noam_scale = lambda epoch: (warmup_steps ** 0.5) * min((epoch + 1) ** -0.5,
                                                               (epoch + 1) * (warmup_steps ** -1.5))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_scale)

    scheduler_config = {
        "scheduler": scheduler,
        "interval": "step",
        "frequency": 1,
    }
    optimizers_schedulers = {
        "optimizer": optimizer,
        "lr_scheduler": scheduler_config
    }
    return optimizers_schedulers
