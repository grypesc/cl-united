#!/bin/bash
for SEED in 0
do
  for ALPHA in 0
  do
    python src/main_incremental.py --approach painkiller --batch-size 128 --num-workers 4 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.05 --weight-decay 1e-5 --adapt --use-test-as-val --alpha $ALPHA --criterion ce --adapter mlp --seed $SEED --nnet resnet32 --exp-name joint_classifier/ce_mlp_alpha=${ALPHA}
  done
done
