# Image classification models fixed-point quantization tutorial

## Overview

Quantization is an important method for model compression and inference performance improvement. PaddlePaddle supports two quantization strategies: `post` and `aware`. In the `post` strategy a trained model is quantized. In the `aware` strategy a model is trained for quantization and then quantized. This tutorial presents the use of the `aware` training quantization strategy to quantize an image classification model and accelerate it through DNNL optimizations. On Intel (R) Cascade Lake class CPU machines, 8-bits quantization, graph optimizations and DNNL acceleration yields performance of a quantized model up to 4 times better than of an original FP32 model. Currently, quantizable operators include `conv2d`, `depthwise_conv2d`, `mul`, `matmul`. DNNL optimizations consist mainly of operator fusing passes which simplify the model graph greatly, further improving the performance, including `batch_norm`, `relu`, `brelu`, `elementwise_add`, etc. After quantization and fuses, INT8 models performance will be greatly improved. For details about DNNL optimization users can refer to [SLIM QAT for DNNL INT8](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/QAT_mkldnn_int8_readme.md)

**Note**:

- PaddlePaddle in version 1.7.1 or higher is required.

- DNNL and MKL are required. The highest performance gain can be observed using CPU servers supporting AVX512 instructions.

- INT8 accuracy is best on CPU servers supporting AVX512 VNNI extension (e.g. CLX class Intel processors). A linux server supports AVX512 VNNI instructions if the output of the command lscpu contains the avx512_vnni entry in the Flags section. AVX512 VNNI support on Windows can be checked using the coreinfo tool.

## 1. Build Paddle, PaddleSlim and Inference library from source code
### 1.1 Build Paddle and Inference library
```
PADDLE_ROOT=/path/of/capi
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
git checkout 2.0-beta -b 2.0-beta
mkdir build
cd build
cmake -DFLUID_INFERENCE_INSTALL_DIR=$PADDLE_ROOT \
      -DCMAKE_INSTALL_PREFIX=./tmp \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_PYTHON=ON \
      -DWITH_MKL=ON \
      -DWITH_MKLDNN=ON \
      -DWITH_GPU=OFF  \
      -DON_INFER=ON \
      -DWITH_PROFILER=OFF \
      ..
 make -j$(nproc)
 make inference_lib_dist
```

### 1.2 Build PaddleSlim from source code
```
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd PaddleSlim
python setup.py install
```
### 1.3 Use paddle and slim in sample code
You can use paddle and paddleslim as follows:
```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```

## 2. Training and converting fp32 model to fp32 qat model
Users can download pre-trained models at [Download pre-trained models](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README.md)

We provide a script to insert `fake_quantize`/`fake_dequantize` ops, train a few iterations and then save the float32 QAT model. Run as follows
```
python train_image_classification.py --model=ResNet50 --pretrained_model=$PATH_TO_ResNet50_pretrained --data=imagenet --data_dir=$PATH_TO_ILSVRC2012/ --save_float32_qat_dir=$PATH_TO_float32_qat_dir
```
Available options in the above command and their descriptions are as follows:
- **model:** Model name. Default value: "ResNet50"
- **pretrained_model:** A path to pre-trained model. Default value: None
- **batch_size：** Number of training batch size. Default value is 128
- **num_epochs:** Number of training epoches. Default value is 1
- **config_file:** A path to training config file. Default value is `./config.yaml`
If the user needs to change the quantization strategy, modify `config.yaml`. We sugggest the following configuration to obtain the best accuracy.

```
config = {
         'weight_quantize_type': 'channel_wise_abs_max',
         'activation_quantize_type': 'moving_average_abs_max',
         'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d', 'elmentwise_add', 'pool2d']
     }
```
For better understanding of the strategies, please refer to [PaddleSlim quant_aware API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/#quant_aware)

**Note:**
- To modify the program for quantizaiton, some training options need to be closed. Running `` sync_batch_norm '' and quant-training with multicards will cause error, the reason is unknown, for now please make sure to close it as below.
```
build_strategy.fuse_all_reduce_ops = False
build_strategy.sync_batch_norm = False
```
- In `train_image_classification.py`, `paddleslim.quant.convert` is used to change the order of `fake_quantize`/`fake_dequantize` ops in Program. In addition, `paddleslim.quant.convert ` will also change the operator parameters such as `conv2d`,` depthwise_conv2d`, and `mul` to the values within the range of quantized `int8_t`, but the data type is still `float32`. This is the qat float32 model we need, the default saving location is `./quantization_models/`.

## 3. Convert fp32 qat model to DNNL INT8 model
The model saved after training in the previous step is the float32 qat model. We have to remove the `fake_quantize`/`fake_dequantize` ops, and fully convert it into INT8 model. Go to the Paddle directory and run

```
python python/paddle/fluid/contrib/slim/tests/save_qat_model.py --qat_model_path=$PATH_TO_float32_qat_dir --int8_model_save_path=$PATH_TO_SAVE_INT8_MODEL --quantized_ops="conv2d,pool2d"
```

## 4. Inference test

### 4.1 Data preprocessing
To run the inference test, the data needs to be converted to binary first. Run the following script without any pramaters allows you to transform the complete ILSVRC2012_val_data set to bianary file. Use `local` parameter to transform your own data. Go to Paddle directory and run:
```
python paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py --local --data_dir=$USER_DATASET_PATH --output_file=$PATH_TO_BINARY_DATA
```

Available options in the above command and their descriptions are as follows:
- No parameters set. The script will download the ILSVRC2012_img_val data from server and convert it into a binary file.
- local, once set, the script will process user data
- data_dir, set the user data directory, default value
- label_list, set the image path-image category list, similar to `val_list.txt`
- output_file, the name of the generated bin file
- data_dim, the length and width of the preprocessed image, the default is 224.

The user's own data set directory structure should be as follows:
```
imagenet_user
├── val
│ ├── ILSVRC2012_val_00000001.jpg
│ ├── ILSVRC2012_val_00000002.jpg
| | ── ...
└── val_list.txt
```
Among them, the content of val_list.txt should be as follows:
```
val/ILSVRC2012_val_00000001.jpg 0
val/ILSVRC2012_val_00000002.jpg 0
```

note:
- The reason for converting the dataset into a binary file is performance and independence of external C++ libraries. Image data requires preprocessing like resizing, cropping, etc. and it can be easily achieved using python Image module, both for training and inference. However, the performance of python tests is lower than of C++ tests, hence the decision to latter for the quantitative model inference tests are made. While effective image processing in C++ requires linking to external libraries like Open-CV, to avoid adding new dependencies to Paddle we use Python to preprocess the image data and put the result in a binary file. Then, the binary dataset is read into the C++ test. A python test sample_tester.py is there for reference, so users can observe its performance overhead compared to the C++ test sample_tester.cc.

### 4.2 Compile and run inference
####  Build the application
Run following commnd under the test directory:
```
mkdir build
cd build
cmake -DUSE_GPU=OFF -DPADDLE_ROOT=$PADDLE_ROOT -DUSE_PROFILER=OFF ..
make -j
```
#### Run the test
```
# Bind threads to cores
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
# Turbo Boost was set to OFF using the command
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# In the file run.sh, set `MODEL_DIR` to `path/to/int8/or/fp32/model`, it can be `$PATH_TO_float32_qat_dir` or `$PATH_TO_SAVE_INT8_MODEL`
# In the file run.sh, set `DATA_FILE` to `$PATH_TO_BINARY_DATA`
# For 1 thread performance:
./run.sh
# For 20 thread performance:
./run.sh -1 20
```

Available options in script `run.sh` and their descriptions are as follows:
- infer_model, the directory where the model is located, note that the model parameters must currently be saved separately into multiple files. Default value None.
- infer_data, the path where the test data file is located. Note that it must be a binary file converted by `full_ILSVRC2012_val_preprocess`.
- batch_size, predict batch size. The default value is 50.
- iterations, predict how many batches. The default is 0, which means predict all batches (image numbers / batch size) in infer_data
- num_threads, the number of CPU threads to be used. The default is one thread.
- with_accuracy_layer. The test is a general test for Image Classification inference. The model may or may not contain the accuracy layer. Default value is false.
- use_profile, provided by the Paddle inference library, set for performance analysis. The default value is false. If you want to set `use_profile` to `true`, users need to build Paddle with `-DWITH_PROFILER=ON` and build sample application with `-DUSE_PROFILER=ON` in advance.

## 5. Accuracy and Performance benchmark

This section contain QAT2 DNNL accuracy and performance benchmark results measured on two server
* Intel(R) Xeon(R) Gold 6271 (with AVX512 VNNI support),
* Intel(R) Xeon(R) Gold 6148.

#### 5.1 Accuracy

>**Intel(R) Xeon(R) Gold 6271**

|    Model     | FP32 Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.71%         |  -0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.11%         |  +0.21%   |       90.56%       |         90.62%         |  +0.06%   |
|  ResNet101   |       77.50%       |         77.64%         |  +0.14%   |       93.58%       |         93.58%         |   0.00%   |
|   ResNet50   |       76.63%       |         76.47%         |  -0.16%   |       93.10%       |         92.98%         |  -0.12%   |
|    VGG16     |       72.08%       |         71.73%         |  -0.35%   |       90.63%       |         89.71%         |  -0.92%   |
|    VGG19     |       72.57%       |         72.12%         |  -0.45%   |       90.84%       |         90.15%         |  -0.69%   |

>**Intel(R) Xeon(R) Gold 6148**

|    Model     | FP32 Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.85%         |   0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.08%         |   0.18%   |       90.56%       |         90.66%         |  +0.10%   |
|  ResNet101   |       77.50%       |         77.51%         |   0.01%   |       93.58%       |         93.50%         |  -0.08%   |
|   ResNet50   |       76.63%       |         76.55%         |  -0.08%   |       93.10%       |         92.96%         |  -0.14%   |
|    VGG16     |       72.08%       |         71.72%         |  -0.36%   |       90.63%       |         89.75%         |  -0.88%   |
|    VGG19     |       72.57%       |         72.08%         |  -0.49%   |       90.84%       |         90.11%         |  -0.73%   |

#### 5.2 Performance

Image classification models performance was measured using a single thread. The setting is included in the benchmark reproduction commands below.

>**Intel(R) Xeon(R) Gold 6271**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      74.36      |       210.68        |       2.83        |
| MobileNet-V2 |      89.59      |       186.55        |       2.08        |
|  ResNet101   |      7.21       |        26.41        |       3.67        |
|   ResNet50   |      13.23      |        48.89        |       3.70        |
|    VGG16     |      3.49       |        10.11        |       2.90        |
|    VGG19     |      2.84       |        8.69         |       3.06        |

>**Intel(R) Xeon(R) Gold 6148**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      75.23      |       111.15        |       1.48        |
| MobileNet-V2 |      86.65      |       127.21        |       1.47        |
|  ResNet101   |      6.61       |        10.60        |       1.60        |
|   ResNet50   |      12.42      |        19.74        |       1.59        |
|    VGG16     |      3.31       |        4.74         |       1.43        |
|    VGG19     |      2.68       |        3.91         |       1.46        |

Notes:

* Performance FP32 (images/s) values come from [INT8 DNNL post-training quantization](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/int8_mkldnn_quantization.md) document.
