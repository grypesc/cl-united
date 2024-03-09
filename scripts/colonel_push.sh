#!/bin/bash
for SEED in 0
do
  for MARGIN in 3
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach colonel_push --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --S 64 --N 300 --K 51 --alpha 30 --use-test-as-val --seed $SEED  --distiller mlp --nnet resnet18 --exp-name baseline/mlp_margin=${MARGIN}
  done
done
