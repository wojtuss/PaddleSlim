#!/bin/bash

MODEL_DIR=/home/li/repo/Paddle/resnet50_quant_int8
#MODEL_DIR=/home/li/repo/Paddle/build/third_party/inference_demo/int8v2/resnet50/model
DATA_FILE=/home/li/.cache/paddle/dataset/int8/download/int8_full_val.bin
#DATA_FILE=/home/li/data/ILSVRC2012/data.bin
num_threads=10
with_accuracy_layer=false
profile=false
ITERATIONS=0
#cgdb --args ./build/inference \
./build/inference --logtostderr=1 \
    --infer_model=${MODEL_DIR} \
    --infer_data=${DATA_FILE} \
    --warmup_size=2 \
    --batch_size=50 \
    --num_threads=${num_threads} \
    --iterations=${ITERATIONS} \
    --with_accuracy_layer=${with_accuracy_layer} \
    --profile=${profile} \
