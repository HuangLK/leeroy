"""Train script.
"""

import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

import models
from readers import AutoDataModule


def train(args):
    pl.seed_everything(args.seed, workers=True)

    model = models.create_model(args)
    dm = AutoDataModule(args)
    models.Model.interact(model, dm)

    # setup tensorboard
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.save_path, name=args.task)
    tb_logger.log_hyperparams(args)

    # setup callbacks
    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(
        filename=args.task + "-{epoch}-{step}-min-val_loss",
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    ))

    callbacks.append(pl.callbacks.ModelCheckpoint(
        filename=args.task + "-{epoch}-{step}",
        save_top_k=-1,
        every_n_train_steps=args.valid_steps,
    ))
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))
    #early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    trainer = pl.Trainer(
        devices=-1,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=tb_logger,
        deterministic=True,
        callbacks=callbacks,
        max_epochs=args.num_epochs,
        accumulate_grad_batches=args.accu_grad_steps,
        log_every_n_steps=args.log_steps,
        val_check_interval=args.valid_steps,
        check_val_every_n_epoch=None,
        resume_from_checkpoint=None, # TODO
    )
    trainer.fit(model, datamodule=dm)

    sys.stdout.write('done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="The name of task.")
    parser.add_argument("--tips", type=str, help="Some tips of task.")
    parser.add_argument("--seed", default=42, type=int, help="Choose your lucky number.")
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

    models.add_cmdline_args(parser)
    AutoDataModule.add_cmdline_args(parser)
    args = parser.parse_args()
    args.use_amp = args.use_amp in ('true', 'True', '1')
    args.use_fast_tokenizer = args.use_fast_tokenizer in ('true', 'True', '1')
    train(args)
