#!/bin/bash
for SEED in 0
do
  for S in 8 16 64
  do
    python src/main_incremental.py --approach painkiller --batch-size 128 --num-workers 4 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.05 --weight-decay 1e-5 --adapt --S $S --use-test-as-val --criterion ce --seed $SEED --nnet resnet32 --exp-name vanilla/ce_resnet32_S=${S}
  done
done
