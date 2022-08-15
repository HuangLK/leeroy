"""Infer script.
"""

import os
import sys
import argparse
import pytorch_lightning as pl

import models
from readers import CSVDataModule
from utils import csv_utils

def infer(args) -> str:
    model = models.create_model(args)
    dm = CSVDataModule(args)

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
    )
    preds = trainer.predict(model, datamodule=dm, ckpt_path=args.ckpt_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # TODO support other datamodules such as `json_datamodule`
    outfile = os.path.join(args.save_path, f'{os.path.basename(args.ckpt_path)}-pred.csv')
    rows = []
    for p_batch, batch in zip(preds, dm.predict_dataloader()):
        batch_rows = [{'pred': p.item()} for p in p_batch]
        for key, cols in batch.items():
            for idx, val in enumerate(cols):
                batch_rows[idx][key] = val
        rows.extend(batch_rows)
    csv_utils.dump_csv(outfile, rows, headers=('text_1', 'text_2', 'label', 'pred'))

    sys.stdout.write('done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="The name of task.")
    parser.add_argument("--ckpt_path", type=str, help="The path of checkpoint.")
    parser.add_argument("--save_path", default="infer_output", type=str,
                        help="The path to save infer result.")

    models.add_cmdline_args(parser)
    CSVDataModule.add_cmdline_args(parser)
    args = parser.parse_args()

    infer(args)
