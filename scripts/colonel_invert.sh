#!/bin/bash
for SEED in 0
do
  for VAL in 30
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach colonel_invert --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 1e-3 --weight-decay 0 --S 64 --N 1 --K 1 --alpha 30 --flow-depth 1 --flow-width 512 --use-test-as-val --seed $SEED  --nnet resnet18 --exp-name flow_v2/alpha=${VAL}
  done
done
