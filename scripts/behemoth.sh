#!/bin/bash
for SEED in 0
do
  for N in 1 10 100 1000 10000
  do
    python src/main_incremental.py --approach behemoth --batch-size 128 --num-workers 4 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adapt --S 64 --alpha 0.5 --use-test-as-val --criterion ce --seed $SEED --distiller linear --nnet resnet32 --exp-name v2/linear_alpha0.5_N=${N}
  done
done
