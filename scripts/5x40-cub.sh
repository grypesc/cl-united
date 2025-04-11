#!/bin/bash
for SEED in 0
do
  for VAL in 10
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach camp_cov --seed $SEED --batch-size 256 --num-workers 8 --nepochs 200 --datasets cub200 --num-tasks 5 --nc-first-task 40 --lr 0.1 --weight-decay 1e-3 --adaptation-strategy full --S 24 --lamb 1 --shrink 0 --use-test-as-val --criterion ce --mahalanobis --normalize --multiplier 32 --distiller mlp --adapter mlp --pretrained-net --nnet resnet18 --use-224 --exp-name 5x40/v1
  done
done
