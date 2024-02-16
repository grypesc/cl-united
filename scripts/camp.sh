#!/bin/bash
for SEED in 0
do
  for ALPHA in 0.5 0.9
  do
    python src/main_incremental.py --approach camp --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 1e-5 --adapt --S 64 --alpha $ALPHA --use-test-as-val --criterion ce --seed $SEED --distiller linear --nnet resnet32 --exp-name 10x10/alpha=${ALPHA}
  done
done
