#!/bin/bash
for SEED in 0
do
  for VAL in 10
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach camp_cov --seed $SEED --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adaptation-strategy full --S 64 --lamb 10 --shrink1 3 --shrink2 0 --use-test-as-val --criterion proxy-yolo --mahalanobis --distiller mlp --adapter mlp --nnet resnet18 --exp-name v1/cov_
  done
done
