#!/bin/bash
for SEED in 0
do
  for S in 2 4 8 16 64
  do
    python src/main_incremental.py --approach soil --batch-size 128 --num-workers 4 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.05 --weight-decay 1e-5 --adapt --S $S --alpha 0.5 --use-test-as-val --criterion ce --seed $SEED --nnet resnet32 --exp-name loss/ce_resnet32_S=${S}_alpha=0.5
  done
done
