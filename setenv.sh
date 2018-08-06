#!/bin/bash

export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_REPO="/home/kingstorm/repos/tensorflow"

export CUDA_ROOT="/usr/local/cuda"

export USR="/usr/local"

#export NSYNC="/home/kingstorm/anaconda3/envs/py35_tf_keras/lib/python3.5/site-packages/tensorflow/include/external/nsync/public"
#export NSYNC="/home/kingstorm/anaconda3/envs/tf12/lib/python3.5/site-packages/tensorflow/include/external/nsync/public"
export NSYNC=""

#export TF_PLF_DEFAULT="/home/kingstorm/anaconda3/envs/py35_tf_keras/lib/python3.5/site-packages/tensorflow/include/tensorflow/core/platform/default"
#export TF_PLF_DEFAULT="/home/kingstorm/anaconda3/envs/tf12/lib/python3.5/site-packages/tensorflow/include/tensorflow/core/platform/default"
export TF_PLF_DEFAULT=""
