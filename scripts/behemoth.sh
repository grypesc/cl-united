#!/bin/bash
for SEED in 0
do
  for N in 100 1000 10000
  do
    python src/main_incremental.py --approach behemoth --batch-size 128 --num-workers 4 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 1e-5 --adapt --S 64 --alpha 0.5 --use-test-as-val --criterion ce --seed $SEED --distiller linear --nnet resnet32 --exp-name behemothv1/ce_linear_alpha0.5_N=${N}
  done
done
