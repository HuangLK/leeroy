"""

Training.
"""
import sys
sys.path.extend([
    #'/home/huangliankai/code/lightning-transformers',
    '/home/huangliankai/code/kg_base/',
])

import argparse
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from transformers import AutoTokenizer

from models import TextClassificationModel
from readers import CSVDataModule
from logger_util import logger


def train(args):
    model = TextClassificationModel(args)
    dm = CSVDataModule(args)

    # setup tensorboard
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.save_path, name=args.task)
    tb_logger.log_hyperparams(args)

    # setup callbacks
    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(
        filename=args.task + "-val_end-{epoch}-{step}",
        save_top_k=-1,
        every_n_epochs=1
    ))
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))
    #early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    trainer = pl.Trainer(
        devices=-1,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=tb_logger,
        callbacks=callbacks,
        max_epochs=args.num_epochs,
        accumulate_grad_batches=args.accu_grad_steps,
        log_every_n_steps=args.log_steps,
        val_check_interval=args.valid_steps,
    )
    trainer.fit(model, datamodule=dm)

    # TODO move to test.py
    # if hasattr(args, "test_file"):
    #     trainer.test(datamodule=dm, ckpt_path='best', )

    logger.info('done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="The name of task.")
    parser.add_argument("--use_amp", default='false', type=str,
                        help="Whether use automatic mixed precision. Default: false.")
    parser.add_argument("--num_epochs", default=1, type=int,
                        help="The number of total training epochs. Default: 1.")
    parser.add_argument("--accu_grad_steps", default=1, type=int,
                        help="Accumulates gradients every x training forward steps. Default: 1.")
    parser.add_argument("--log_steps", default=100, type=int,
                        help="Log training / evaluation infomation to tensorboard every x training backward steps. "
                             "Default: 100.")
    parser.add_argument("--valid_steps", default=100, type=int,
                        help="Trigger validation loop every x training forward steps. Default: 100. "
                             "NOTE: "
                             "1. `valid_steps` must be less than the total training forward steps in an epoch, "
                                 "otherwise the validation loop will not be triggered;"
                             "2. `valid_steps` refers to the number of training forward steps, "
                                 "during which the model parameters are actually updated by "
                                 "`valid_steps` / `accu_grad_steps` steps.")
    parser.add_argument("--save_path", default="output", type=str,
                        help="The path to save model checkpoints and logs.")

    TextClassificationModel.add_cmdline_args(parser)
    CSVDataModule.add_cmdline_args(parser)
    args = parser.parse_args()
    args.use_amp = args.use_amp in ('true', 'True', '1')
    train(args)
