#!/bin/bash
set -x
set -e
PS=localhost
WORKER1=localhost
WORKER2=localhost

DATASET_NAME=cifar10 # imagenet or cifar10
DATA_DIR=${HOME}/dataset/${DATASET_NAME}-data # dataset location
ROOT_WORKSPACE=${HOME}/dataset/results/cifar10/ # the location to store summary and logs
if [ ! -d "$ROOT_WORKSPACE" ]; then
  echo "${ROOT_WORKSPACE} does not exsit!"
  exit
fi

TRAIN_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_training_data/
EVAL_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_eval_data/
INFO_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_info/
if [ ! -d "${INFO_WORKSPACE}" ]; then
  echo "Creating ${INFO_WORKSPACE} ..."
  mkdir -p ${INFO_WORKSPACE}
fi
current_time=$(date)
current_time=`echo ${current_time} | sed 's/\ /-/g' | sed 's/://g'` #${current_time// /_}
#current_time=${current_time//:/-}
FOLDER_NAME=${DATASET_NAME}__${current_time}

export CUDA_VISIBLE_DEVICES=1
python ./inception/cifar10_distributed_train.py \
--optimizer adam \
--initial_learning_rate 0.0002 \
--batch_size 64 \
--num_epochs_per_decay 200 \
--max_steps 300000 \
--seed 123 \
--weight_decay 0.004 \
--net cifar10_alexnet \
--image_size 24 \
--data_dir=${DATA_DIR} \
--job_name='worker' \
--task_id=1 \
--ps_hosts="$PS:2222" \
--worker_hosts="${WORKER1}:2224,${WORKER2}:2226" \
--train_dir=/tmp/cifar10_distributed_train \
> ${INFO_WORKSPACE}/train_${FOLDER_NAME}_w1_info.txt 2>&1 &

export CUDA_VISIBLE_DEVICES=1
python ./inception/cifar10_distributed_train.py \
--optimizer adam \
--initial_learning_rate 0.0002 \
--batch_size 64 \
--num_epochs_per_decay 200 \
--max_steps 300000 \
--seed 123 \
--weight_decay 0.004 \
--net cifar10_alexnet \
--image_size 24 \
--data_dir=${DATA_DIR} \
--job_name='worker' \
--task_id=0 \
--ps_hosts="$PS:2222" \
--worker_hosts="${WORKER1}:2224,${WORKER2}:2226" \
--train_dir=/tmp/cifar10_distributed_train #\
#> ${INFO_WORKSPACE}/train_${FOLDER_NAME}_w2_info.txt 2>&1 &

export CUDA_VISIBLE_DEVICES=0
python ./inception/cifar10_distributed_train.py \
--job_name='ps' \
--task_id=0 \
--ps_hosts="$PS:2222" \
--worker_hosts="${WORKER1}:2224,${WORKER2}:2226" #\
#> ${INFO_WORKSPACE}/train_${FOLDER_NAME}_ps_info.txt 2>&1 &
