#!/bin/bash
for SEED in 1
do
  for VAL in 10
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach united2 --seed $SEED --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adaptation-strategy full --S 64 --lamb 10 --use-test-as-val --multiplier 32  --exp-name 10x10/
  done
done
