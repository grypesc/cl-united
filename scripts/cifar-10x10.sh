#!/bin/bash
for SEED in 1
do
  for VAL in 10
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach ensembler --seed $SEED --batch-size 128 --num-workers 4 --nepochs 300 --shrink 0.01 --dump --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.3 --weight-decay 5e-4 --distiller baseline --adapter baseline --S 64 --rotation --lamb 10 --use-test-as-val --multiplier 16  --exp-name 10x10/
  done
done
