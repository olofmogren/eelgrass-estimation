#!/bin/bash

source .venv/bin/activate

python train_model.py --learning-rate 1e-4 --batch-size 2 --deep-supervision True --invariance-loss-weight 0 --model-save-path checkpoints --results-csv-path results

