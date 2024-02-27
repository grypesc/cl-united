#!/bin/bash
for SEED in 0
do
  for BETA in 5 15
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach colonel --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adapt --S 64 --N 100 --alpha 10 --beta $BETA --use-test-as-val --seed $SEED  --K 21 --distiller linear --nnet resnet32 --exp-name colo/linear_beta=${BETA}
  done
done
