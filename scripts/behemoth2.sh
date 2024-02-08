#!/bin/bash
for SEED in 0
do
  for N in 1 10 100 1000 10000
  do
    python src/main_incremental.py --approach behemoth2 --batch-size 128 --num-workers 4 --nepochs 200 --datasets cifar100_icarl --num-tasks 10 --nc-first-task 10 --lr 0.1 --weight-decay 5e-4 --adapt --S 64 --alpha 3 --N $N --K 1 --push-fun sigmoid --gamma 0.1 --momentum 0.9 --smoothing 0.0 --use-test-as-val --seed $SEED --distiller linear --nnet resnet32 --exp-name v1/alpha=3_N=${N}
  done
done
