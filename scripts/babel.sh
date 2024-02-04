#!/bin/bash
for SEED in 0
do
  for ALPHA in 0.5
  do
    python src/main_incremental.py --approach babel --batch-size 128 --num-workers 4 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adapt --S 64 --N 5 --alpha $ALPHA --use-test-as-val --criterion ce --seed $SEED --distiller linear --nnet resnet32 --exp-name v1/linear_alpha=${ALPHA}
  done
done
