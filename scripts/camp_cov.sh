#!/bin/bash
for SEED in 0
do
  for VAL in 10
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach camp_cov --seed $SEED --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 5 --nc-first-task 20 --lr 0.1 --weight-decay 1e-3 --adaptation-strategy full --S 64 --alpha 10 --shrink1 1 --shrink2 1 --use-test-as-val --criterion proxy-yolo --distiller linear --adapter linear --nnet resnet18 --exp-name v1/cov_
  done
done
