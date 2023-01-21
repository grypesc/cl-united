#!/bin/bash
for SEED in 0
do
  for NUM_EXPERTS in 5
  do
    CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach berg --gmms 1 --max-experts $NUM_EXPERTS --use-multivariate --ft-selection-strategy softmax  --nepochs 200 --tau 3 --batch-size 256 --num-workers 12 --datasets imagenet_subset_kaggle --num-tasks 6 --nc-first-task 50 --lr 0.05 --weight-decay 5e-4 --clipping 1 --alpha 0.99 --extra-aug fetril --use-test-as-val --network resnet18  --momentum 0.9 --exp-name imagenet_50+5x10_$NUM_EXPERTS --seed $SEED
  done
done
