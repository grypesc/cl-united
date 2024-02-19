#!/bin/bash
for SEED in 0
do
  for K in 10
  do
    for N in 10
    do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach babel_mp --batch-size 128 --num-workers 0 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --S 64 --num-processes 10 --N $N --K $K --strategy constant --use-negative --full-initial --alpha 10 --use-test-as-val --seed $SEED --distiller linear --nnet resnet8 --exp-name v10/K=${K}
    done
  done
done
