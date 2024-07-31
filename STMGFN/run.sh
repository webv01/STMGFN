#!/bin/bash
python experiments/train.py -c baselines/STMGFN/METR-LA.py --gpus '0'
python experiments/train.py -c baselines/STMGFN/PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/STMGFN/PEMS03.py --gpus '0'
python experiments/train.py -c baselines/STMGFN/PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STMGFN/PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STMGFN/PEMS08.py --gpus '0'
