#!/bin/bash
model=resnet20
dataset=CIFAR100
loss_fn=CrossEntropyLoss

for lt_factor in 100 10 1
do
    echo "python -u trainer.py  --arch=$model --dataset $dataset --lt_factor $lt_factor --loss_fn $loss_fn  --save-dir=checkpoints/${loss_fn}_${model}_${dataset}_${lt_factor} |& tee -a checkpoints/${loss_fn}_${model}_${dataset}_${lt_factor}/log_$lt_factor"
    python -u trainer.py  --arch=$model --dataset $dataset --lt_factor $lt_factor --loss_fn $loss_fn  --save-dir=checkpoints/${loss_fn}_${model}_${dataset}_${lt_factor} |& tee -a checkpoints/${loss_fn}_${model}_${dataset}_${lt_factor}/log.txt
done


model=resnet20
dataset=CIFAR100
loss_fn=CBCELoss

for lt_factor in 100 10 1
do
    echo "python -u trainer.py  --arch=$model --dataset $dataset --lt_factor $lt_factor --loss_fn $loss_fn  --save-dir=checkpoints/${loss_fn}_${model}_${dataset}_${lt_factor} |& tee -a checkpoints/${loss_fn}_${model}_${dataset}_${lt_factor}/log_$lt_factor"
    python -u trainer.py  --arch=$model --dataset $dataset --lt_factor $lt_factor --loss_fn $loss_fn  --save-dir=checkpoints/${loss_fn}_${model}_${dataset}_${lt_factor} |& tee -a checkpoints/${loss_fn}_${model}_${dataset}_${lt_factor}/log.txt
done