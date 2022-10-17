"""Infer script.
"""

import os
import sys
import json
import argparse
import pytorch_lightning as pl
import models
from readers import AutoDataModule

def infer(args) -> str:
    model = models.create_model(args)
    dm = AutoDataModule(args)

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
    )
    outputs = trainer.predict(model, datamodule=dm)
    preds = [p for out in outputs for p in out['preds']]
    for pred in preds:
        print(pred)

    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)

    # outfile = args.ckpt_path.replace(".ckpt", "-pred.csv")
    # with open(outfile, 'w') as fout:
    #     for pred in preds:
    #         fout.write(json.dumps(pred, ensure_ascii=False))
    #         fout.write('\n')

    # sys.stdout.write(f'save result: {outfile}\n')
    # sys.stdout.write('done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="The name of task.")
    parser.add_argument("--ckpt_path", type=str, help="The path of checkpoint.")
    parser.add_argument("--save_path", default="infer_output", type=str,
                        help="The path to save infer result.")

    models.add_cmdline_args(parser)
    AutoDataModule.add_cmdline_args(parser)
    args = parser.parse_args()

    infer(args)
