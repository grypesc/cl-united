#!/bin/bash
for SEED in 0
do
  for ALPHA in 30
  do
    python src/main_incremental.py --approach camp_cov --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adaptation-strategy diag --S 64 --alpha $ALPHA --use-test-as-val --criterion proxy-yolo --seed $SEED --distiller linear --nnet resnet18 --exp-name v1/cov_alpha=${ALPHA}
  done
done
