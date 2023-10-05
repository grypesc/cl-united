#!/bin/bash
for SEED in 0
do
  for ALPHA in 0.5
  do
    CUDA_VISIBLE_DEVICES=0 python main_incremental.py --approach soil --batch-size 128 --num-workers 0 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --S 64 --lr 0.05 --weight-decay 0 --clipping 1 --alpha $ALPHA --use-test-as-val --network resnet32 --momentum 0.9 --seed $SEED --exp-name soil_S=64_$ALPHA
    CUDA_VISIBLE_DEVICES=0 python main_incremental.py --approach soil --batch-size 128 --num-workers 0 --nepochs 100 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --S 4 --lr 0.05 --weight-decay 0 --clipping 1 --alpha $ALPHA --use-test-as-val --network resnet32 --momentum 0.9 --seed $SEED --exp-name soil_S=4_$ALPHA
  done
done
