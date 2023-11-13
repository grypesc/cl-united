#!/bin/bash
for SEED in 0
do
  for NUM_EXPERTS in 1 3 5
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach berg --gmms 1 --max-experts $NUM_EXPERTS --use-multivariate --ft-selection-strategy robin  --nepochs 200 --tau 3 --batch-size 128 --num-workers 4 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.05 --weight-decay 5e-4 --clipping 1 --alpha 0.99 --use-test-as-val --network resnet32 --extra-aug fetril --momentum 0.9 --shared 0 --exp-name heatmap_$NUM_EXPERTS --seed $SEED
  done
done
