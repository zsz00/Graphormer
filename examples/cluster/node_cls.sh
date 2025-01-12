#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

ulimit -c unlimited

conda activate graphormer

CUDA_VISIBLE_DEVICES=3 fairseq-train \
--user-dir ../../graphormer \
--num-workers 1 \
--ddp-backend=legacy_ddp \
--user-data-dir=/home/zhangyong/codes/Graphormer/graphormer/data/customized_dataset  \
--dataset-name cluster_dataset_1 \
--dataset-source ogb \
--task node_prediction \
--criterion cross_entropy \
--arch graphormer_base \
--num-classes 15119 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 1000000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size 2 \
--fp16 \
--data-buffer-size 20 \
--save-dir ./ckpts



