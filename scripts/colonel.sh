#!/bin/bash
for SEED in 0
do
  for ALPHA in 1 3 10 30
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach colonel_nca --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adapt --S 64 --N 300 --K 21 --alpha $ALPHA --use-test-as-val --seed $SEED  --distiller linear --nnet resnet32 --exp-name baseline/linear_alpha=${ALPHA}
  done
done
