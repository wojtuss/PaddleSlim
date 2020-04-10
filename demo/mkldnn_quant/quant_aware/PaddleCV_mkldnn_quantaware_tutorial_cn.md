# 图像分类模型定点量化教程

## 概述

量化是模型压缩的重要手段。在PaddlePaddle中，量化策略`post`为使用离线量化得到的模型，`aware`为在线量化训练得到的模型。本教程介绍了使用训练时量化策略(`aware`)，对图像分类模型进行量化和MKL-DNN优化加速。在Intel(R) Xeon(R) Gold 6271机器上，经过量化和MKL-DNN加速后，INT8模型在单线程上性能为原FP32模型的3~4倍，而精度仅有极小下降。目前，我们支持训练量化的op包括conv2d、depthwise_conv2d、mul, matmul；同时在MKL-DNN 优化阶段，我们会fuse很多其他op，包括batch_norm、relu、brelu，elementwise_add等，经过Op量化和op fuses，量化模型性能会大大提升。具体MKL-DNN优化可以参考[SLIM QAT for INT8 MKL-DNN](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/QAT_mkldnn_int8_readme.md)


注意：
1. 需要DNNL1.3库和MKL-ML库。
2. 只有使用AVX512系列CPU服务器才能获得性能提升。要达到本页面最下方的性能提升，运行机器须支持指令`avx512_vnni`,用户可以通过在命令行红输入`lscpu`查看本机支持指令。
3. 在支持`avx512_vnni`的CPU服务器上，INT8精度最高。

## 1. 从源代码编译安装Paddle, PaddleSlim 和预测库

#### 1.1 从源代码构建Paddle和预测库，请使用以下编译选项。
在用户的主目录，执行
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
      ..
 make -j$(nproc)
 make inference_lib_dist
```

#### 1.2 从源代码构建PaddleSlim
在用户的主目录，执行
```
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd PaddleSlim
python setup.py install
```

#### 1.3 在代码中使用
在用户自己的测试样例中，按以下方式导入Paddle和PaddleSlim:
```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```

## 2. 训练
首先，用户在此处链接下载我们已经预训练好的的模型[预训练模型下载](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README.md)。

我们提供了一个插入量化反量化op后训练，并且保存float32 QAT模型的脚本，你可以直接运行：
```
python train_image_classification.py --model=ResNet50 --pretrained_model=$PATH_TO_ResNet50_pretrained --data=imagenet --data_dir=$PATH_TO_ILSVRC2012/ --save_float32_qat_dir=$PATH_TO_float32_qat_dir
```
参数说明如下。
- **pretrained_model:** 传入预训练好的模型
- **max_iters:** 训练的总轮次。如果使用预训练模型，量化模型需要的训练轮次比正常训练小很多。
- **LeaningRate.base_lr:** 根据总`batch_size`调整`base_lr`，两者大小正相关，可以简单的按比例进行调整。
- **LearningRate.schedulers.PiecewiseDecay.milestones：** 请根据batch size的变化对其调整。
- **num_epochs:** 多训练几个epoch，精度理论上会更高。

在Program中插入量化和反量化OP阶段，如果用户需要更改量化策略，可以更改 `config.yaml` 配置。我们目前建议使用以下配置可以获得最佳精度。

```
config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d','elmentwise_add','pool2d']
    }
```
如果想了解各参数含义，可参考 [PaddleSlim quant_aware API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/#quant_aware)


**注意：**

- 因为量化要对Program做修改，所以一些会修改Program的训练策略需要关闭。``sync_batch_norm`` 和量化多卡训练同时使用时会出错，原因暂不知，请确保在代码中将其关闭。（示例代码中已关闭）
    ```
    build_strategy.fuse_all_reduce_ops = False
    build_strategy.sync_batch_norm = False
    ```

- 在``train_image_classification.py``中，``paddleslim.quant.convert`` 主要用于改变Program中量化op和反量化op的顺序。除此之外，``paddleslim.quant.convert`` 还会将`conv2d`、`depthwise_conv2d`、`mul`等算子参数变为量化后的`int8_t`范围内的值，但数据类型仍为`float32`。这就是我们需要的qat float32模型，位置默认为``./quantization_models/act_*/float``。

## 3. 转化fp32 qat模型为MKL-DNN优化后的INT8模型
上一步中训练后保存的模型是float32 qat模型。我们还需要移除量化，反量化op，fuse一些op，并且完全转化成 INT8 模型。运行下面的脚本

```
python ./save_qat_model.py --qat_model_path=$PATH_TO_float32_qat_dir --int8_model_save_path=$PATH_TO_SAVE_INT8_MODEL --quantized_ops="conv2d,pool2d"
```

## 4. 预测

### 4.1 数据预处理转化
在精度和性能预测中，需要先对数据进行二进制转化。运行脚本如下可转化完整ILSVRC2012 val数据集。使用可选参数转化用户自己的数据。
```
python ../tools/full_ILSVRC2012_val_preprocess.py --local --data_dir=$USER_DATASET_PATH --output_file=data.bin
```

可选参数：
- 不设置任何参数。脚本将下载 ILSVRC2012_img_val数据集，并转化为二进制文件。
- **local:** 设置便为true，表示用户将提供自己的数据
- **data_dir:** 设置用户自己的数据目录
- **label_list:** 设置图片路径-图片类别列表，类似于`val_list.txt`
- **output_file:** 生成的bin文件名
- **data_dim:** 预处理图片的长和宽，默认为224。不建议更改

用户自己的数据集目录结构应该如下
```
imagenet_user
├── val
│   ├── ILSVRC2012_val_00000001.jpg
│   ├── ILSVRC2012_val_00000002.jpg
|   |── ...
└── val_list.txt
```
其中，val_list.txt 内容应该如下：
```
val/ILSVRC2012_val_00000001.jpg 0
val/ILSVRC2012_val_00000002.jpg 0
```

注意：
- 为什么将数据集转化为二进制文件？因为paddle中的数据预处理（resize, crop等）都使用pythong.Image模块进行，训练出的模型也是基于Python预处理的图片，但是我们发现Python测试性能开销很大，导致预测性能下降。为了获得良好性能，在量化模型预测阶段，我们决定使用C++测试，而C++只支持Open-CV等库，Paddle不建议使用外部库，因此我们使用Python将图片预处理然后放入二进制文件，再在C++测试中读出。用户根据自己的需要，可以更改C++测试以使用open-cv库直接读数据并预处理，精度不会有太大下降。我们还提供了python测试`sample_tester.py`作为参考，与C++测试`sample_tester.cc`相比，用户可以看到Python测试更大的性能开销。

### 4.2 编译运行预测
#### 编译应用
在样例所在目录下，执
```
mkdir build
cd build
cmake -DUSE_GPU = OFF -DPADDLE_ROOT =$PADDLE_ROOT ..
make -j
```

#### 运行测试
```
# Bind threads to cores
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
# Turbo Boost was set to OFF using the command
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# In the file run.sh, set `MODEL_DIR` to `PATH_INT8_OR_FLOAT32_MODEL`
# In the file run.sh, set `DATA_FILE` to `PATH_TO_BINARY_DATA`
# For 1 thread performance:
./run.sh
# For 20 thread performance:
./run.sh -1 20
```

运行时需要配置以下参数：
- **infer_model:** 模型所在目录，注意模型参数当前必须是分开保存成多个文件的。无默认值。
- **infer_data:** 测试数据文件所在路径。注意需要是经`full_ILSVRC2012_val_preprocess`转化后的binary文件。
- **warmup_size:** warmup的步数。默认值为0，即没有warmup。
- **batch_size:** 预测batch size大小。默认值为50。
- **iterations:** 预测多少batches。默认为0，表示预测infer_data中所有batches (image numbers/batch size)
- **num_threads:** 预测使用CPU 线程数，默认为单核一个线程。
- **with_accuracy_layer:** 由于这个测试是Image Classification通用的测试，既可以测试float32模型也可以INT8模型，模型可以包含或者不包含label层，设置此参数更改。
- **use_profile:** 由Paddle预测库中提供，设置用来进行性能分析。默认值为false。

你可以直接修改`run.sh`中的MODEL_DIR和DATA_DIR后，即可执行`./run.sh`进行CPU预测。

## 5. QAT量化图像分类模型在 Xeon(R) 6271 和 Xeon(R) 6148 上的精度和性能

表格中的性能是在以下前提获得：
* 通过设置将thread指定给core

   ```
   export KMP_AFFINITY=granularity=fine,compact,1,0
   export KMP_BLOCKTIME=1
   ```

* 使用以下命令将Turbo Boost设置为OFF

   ```
   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
   ```

### 5.1 QAT量化模型精度

>**I. QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6271的精度**

|    Model     | FP32 Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.71%         |  -0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.11%         |  +0.21%   |       90.56%       |         90.62%         |  +0.06%   |
|  ResNet101   |       77.50%       |         77.64%         |  +0.14%   |       93.58%       |         93.58%         |   0.00%   |
|   ResNet50   |       76.63%       |         76.47%         |  -0.16%   |       93.10%       |         92.98%         |  -0.12%   |
|    VGG16     |       72.08%       |         71.73%         |  -0.35%   |       90.63%       |         89.71%         |  -0.92%   |
|    VGG19     |       72.57%       |         72.12%         |  -0.45%   |       90.84%       |         90.15%         |  -0.69%   |

>**II. QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6148 的精度**

|    Model     | FP32 Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.85%         |   0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.08%         |   0.18%   |       90.56%       |         90.66%         |  +0.10%   |
|  ResNet101   |       77.50%       |         77.51%         |   0.01%   |       93.58%       |         93.50%         |  -0.08%   |
|   ResNet50   |       76.63%       |         76.55%         |  -0.08%   |       93.10%       |         92.96%         |  -0.14%   |
|    VGG16     |       72.08%       |         71.72%         |  -0.36%   |       90.63%       |         89.75%         |  -0.88%   |
|    VGG19     |       72.57%       |         72.08%         |  -0.49%   |       90.84%       |         90.11%         |  -0.73%   |

### 5.2 QAT量化模型性能

>**III. QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6271的单核的性能**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      73.98      |       227.73        |       3.08        |
| MobileNet-V2 |      86.59      |       206.74        |       2.39        |
|  ResNet101   |      7.15       |        26.69        |       3.73        |
|   ResNet50   |      13.15      |        49.33        |       3.75        |
|    VGG16     |      3.34       |        10.15        |       3.04        |
|    VGG19     |      2.83       |        8.67         |       3.07        |


>**IV. QAT MKL-DNN 在 Intel(R) Xeon(R) Gold 6148的单核的性能**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      75.23      |       111.15        |       1.48        |
| MobileNet-V2 |      86.65      |       127.21        |       1.47        |
|  ResNet101   |      6.61       |        10.60        |       1.60        |
|   ResNet50   |      12.42      |        19.74        |       1.59        |
|    VGG16     |      3.31       |        4.74         |       1.43        |
|    VGG19     |      2.68       |        3.91         |       1.46        |

## FAQ

该示例使用PaddleSlim提供的[量化压缩API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/)对检测模型进行压缩。
在阅读该示例前，建议您先了解以下内容：

- [检测模型的常规训练方法](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)
