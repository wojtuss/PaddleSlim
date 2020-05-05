#!/bin/bash

MODEL_DIR=/home/li/models/ResNet50_4th_qat_int8
#DATA_FILE=/home/li/.cache/paddle/dataset/int8/download/int8_full_val.bin
DATA_FILE=/home/li/data/ILSVRC2012/data.bin
num_threads=1
with_accuracy_layer=false
profile=false
REPEAT=1
./build/inference \
    --infer_model=${MODEL_DIR} \
    --infer_data=${DATA_FILE} \
    --num_threads=${num_threads} \
    --iterations=2 \
    --with_accuracy_layer=${with_accuracy_layer} \
    --profile=${profile} \
