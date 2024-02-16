#!/bin/bash
for SEED in 0
do
  for ALPHA in 10
  do
    for N in 10
    do
    python src/main_incremental.py --approach babel_mp --batch-size 128 --num-workers 0 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --S 64 --num-processes 10 --N $N --K 1 --strategy constant --alpha $ALPHA --use-test-as-val --seed $SEED --distiller linear --nnet resnet8 --exp-name v5-1vsAll/alpha=${ALPHA}
    done
  done
done
