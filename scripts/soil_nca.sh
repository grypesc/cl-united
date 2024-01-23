#!/bin/bash
for SEED in 0
do
  for ALPHA in 0.5 0.0 0.1 0.9 0.99
  do
    python src/main_incremental.py --approach soil --batch-size 128 --num-workers 4 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.05 --weight-decay 1e-5 --adapt --S 64 --alpha $ALPHA --use-test-as-val --seed $SEED --nnet resnet32 --exp-name vanilla-nogames-nogimmicks/nca_resnet32_alpha=${ALPHA}
  done
done
