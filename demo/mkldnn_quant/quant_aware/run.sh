#!/bin/bash
MODEL_DIR=$HOME/models/resnet50_quant_int8
#MODEL_DIR=$HOME/repo/Paddle/build/third_party/inference_demo/int8v2/resnet50/model
DATA_FILE=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin
#DATA_FILE=/data/datasets/ImageNet_py/val.bin
num_threads=10
with_accuracy_layer=false
use_profile=true
ITERATIONS=100
#cgdb --args ./build/inference \
./build/inference --logtostderr=1 \
    --infer_model=${MODEL_DIR} \
    --infer_data=${DATA_FILE} \
    --warmup_size=2 \
    --batch_size=1 \
    --num_threads=${num_threads} \
    --iterations=${ITERATIONS} \
    --with_accuracy_layer=${with_accuracy_layer} \
    --use_profile=${use_profile} \
