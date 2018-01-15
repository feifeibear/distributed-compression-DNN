#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 4 
#SBATCH -C knl,quad,cache
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -J dist_cifar
#SBATCH --output=dist_benchmark_cori.%j.log

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers

# load modules
module load tensorflow/intel-head
KMP_AFFINITY="granularity=fine,noverbose,nowarning,compact,1,0"
#module load tensorflow/1.4.0rc0

export OMP_NUM_THREADS=8 #66 #$SLURM_CPUS_PER_TASK
export NUM_INTER_THREADS=1 #2
export NUM_INTRA_THREADS=8 #33

# load virtualenv
#export WORKON_HOME=~/Envs
#source $WORKON_HOME/tf-daint/bin/activate

# set TensorFlow script parameters
DATASET_NAME=cifar10
export SCRIPT_DIR="/global/cscratch1/sd/yyang420/fjr/tensorflow/distributed-compression-DNN/terngrad/inception"
export TF_SCRIPT=$SCRIPT_DIR/cifar10_distributed_train.py
export TF_EVAL_SCRIPT=$SCRIPT_DIR/cifar10_eval.py
DATA_DIR=$SCRATCH/fjr/dataset/cifar10-data #dataset location

export PYTHONPATH=/global/cscratch1/sd/yyang420/fjr/tensorflow/distributed-compression-DNN/terngrad:$PYTHONPATH

ROOT_WORKSPACE=${SCRATCH}/fjr/dataset/results/cifar10 # the location to store summary and logs
TRAIN_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_train_log

export TF_FLAGS="
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
--train_dir=${TRAIN_WORKSPACE} 
"

EVAL_INTERVAL_SECS=10
EVAL_DEVICE="/cpu:0" # specify the device to eval. e.g. "/gpu:1", "/cpu:0"
RESTORE_AVG_VAR=True # use the moving average parameters to eval?
MAX_STEPS=300000
VAL_BATCH_SIZE=50 # set smaller to avoid OOM
VAL_TOWER=1 # -1 for cpu
EVAL_DIR=${ROOT_WORKSPACE}/${DATASET_NAME}_eval_log
export TF_EVAL_FLAGS="
--eval_interval_secs ${EVAL_INTERVAL_SECS} \
--device ${EVAL_DEVICE} \
--restore_avg_var ${RESTORE_AVG_VAR} \
--data_dir ${DATA_DIR} \
--subset "test" \
--net cifar10_alexnet \
--image_size 24 \
--batch_size ${VAL_BATCH_SIZE} \
--max_steps ${MAX_STEPS} \
--checkpoint_dir ${TRAIN_WORKSPACE} \
--tower ${VAL_TOWER} \
--eval_dir ${EVAL_DIR}
"



# set TensorFlow distributed parameters
export TF_NUM_PS=$1 # 1
export TF_NUM_WORKERS=$2 # $SLURM_JOB_NUM_NODES
# export TF_WORKER_PER_NODE=1
# export TF_PS_PER_NODE=1
export TF_PS_IN_WORKER=true

# run distributed TensorFlow
DIST_TF_LAUNCHER_DIR=/global/cscratch1/sd/yyang420/fjr/dist_cifar/res-$1-$2
# cd $DIST_TF_LAUNCHER_DIR
# current_time=$(date)
# current_time=`echo ${current_time} | sed 's/\ /-/g' | sed 's/://g'` #${current_time// /_}
mkdir -p $DIST_TF_LAUNCHER_DIR #${current_time}
cp ./run_dist_tf_cori.sh $DIST_TF_LAUNCHER_DIR #${current_time}
cd $DIST_TF_LAUNCHER_DIR
rm -rf .tfdist* ps.* worker.* eval.*
./run_dist_tf_cori.sh

# deactivate virtualenv
#deactivate
