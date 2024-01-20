#!/bin/bash
for SEED in 0
do
  for ALPHA in 0.5 0.9 0.99
  do
    python src/main_incremental.py --approach shookone --batch-size 128 --num-workers 4 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 1e-5 --S 64 --alpha $ALPHA --smoothing 0.1 --use-test-as-val --distiller mlp --seed $SEED --nnet resnet32 --exp-name v3/mlp_alpha=${ALPHA}
  done
done
