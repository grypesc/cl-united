#!/bin/bash
for SEED in 0
do
  for ALPHA in 0.9 0.99 0.999 0.9999
  do
    python src/main_incremental.py --approach soil --batch-size 128 --num-workers 4 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 1e-5 --adapt --S 64 --alpha $ALPHA --use-test-as-val --criterion abc --seed $SEED --distiller linear --nnet resnet32 --exp-name abc/abc_linear_alpha=${ALPHA}
  done
done
