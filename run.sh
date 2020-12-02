#!/bin/bash

model=resnet20
dataset=CIFAR100
for lt_factor in 1 10 50
do
python -u trainer.py  --arch=resnet20 --dataset CIFAR100 --lt_factor 1  --save-dir=checkpoints/save_$lt_factor |& tee -a log_$lt_factor
python -u trainer.py  --arch=resnet20 --dataset CIFAR100 --lt_factor 1  --save-dir=save_$lt_factor |& tee -a log_$lt_factor
done