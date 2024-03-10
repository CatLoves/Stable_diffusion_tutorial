#!/bin/bash

# set -xe

# # set envs value
# # gpu nums per machine
# GPUS=4
# #local python
# LOCAL_PYTHON="/training/env/weizixiong/nnset"

# if [ ! -d ${LOCAL_PYTHON} ];then
#     echo "local python dir not exists, pls check !!!"
#     exit -1
# fi

# ${LOCAL_PYTHON}/bin/python -m torch.distributed.launch \
#     --nnodes=${MLP_WORKER_NUM} \
#     --node_rank=${MLP_ROLE_INDEX} \
#     --master_addr=${MLP_WORKER_0_HOST} \
#     --nproc_per_node=$GPUS \
#     --master_port=${MLP_WORKER_0_PORT} \
#     tools/train.py configs/lane/seg_4pe_lane_parsing_switch_pipeline.py \
#     --launcher pytorch

pip install -v -e .
pip install lmdb
pip install tensorboard
# pip install numba
# pip install -U albumentations -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export OPENCV_LOG_LEVEL=OFF

train_sd()
{
    # export QAT_mode="horizon_eager_float"
    python -m torch.distributed.launch \
        --nnodes=${MLP_WORKER_NUM} \
        --node_rank=${MLP_ROLE_INDEX} \
        --master_addr=${MLP_WORKER_0_HOST} \
        --nproc_per_node=${MLP_WORKER_GPU} \
        --master_port=${MLP_WORKER_0_PORT} \
        tools/train_ddpm_cond.py \
        --launcher pytorch \
        --work-dir pilot_mono3d_mutil_taskes_small_dense_depth_auxiliary
}


train_sd
