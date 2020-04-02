# 目标检测模型定点量化教程

>运行该示例前请安装Paddle1.6或更高版本和PaddleSlim

# 图像分类模型定点量化教程（采用训练中引入量化策略）

## 概述

量化是模型压缩的重要手段，可以缩小模型，并且保证精度仅有极小的下降。在PaddlePaddle中，量化策略`post`为使用离线量化得到的模型，`aware`为在线量化训练得到的模型。本教程介绍了使用训练时量化策略(`aware`)，结合MKL-DNN库，对图像分类模型进行量化和MKL-DNN加速。经过量化和MKL-DNN加速后，INT8模型在单线程上性能为原FP32模型的3倍，而精度仅下降了0.27%。MKL-DNN量化实现了以下优势。

在训练脚本中，在主要算子前插入量化op和反量化op，和因为op融合需要的quantize_dequantize_pass，通过训练微调这两类op。目前我们支持以下op前插入量化和反量化op：
```
conv, depthwise_conv2d, mul (anything else)
```
在转化成真实定点模型（INT8模型）阶段，根据MKL-DNN支持，将激活函数，batch normalization等op合入到conv op中。我们目前可以实现以下pattern的INT8 fuse如下。 op融合后不仅使用INT8 计算，而且融合了激活函数，无需另外开辟空间，大大提高了性能。
```
input1 → conv2d → output1 → batch_norm → output2 → relu → output3 →
```
转化为
```
... → input1 → conv2d → output3 → ...
```

注意：
1. 需要MKL-DNN和MKL。 只有使用AVX512系列CPU服务器才能获得性能提升。
2. 在支持AVX512 VNNI扩展的CPU服务器上，INT8精度最高。

## 1. 安装PaddleSlim

可按照[PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)中的步骤安装PaddleSlim。

PaddleSlim依赖Paddle1.7版本，请确认已正确安装Paddle，然后按以下方式导入Paddle和PaddleSlim:

```
import paddle
import paddle.fluid as fluid
import paddleslim as slim
import numpy as np
```

## 2. 训练

As the strategy changed, this part will be modified, together with the train.py scripts.
根据 [tools/train.py](https://github.com/PaddlePaddle/PaddleDetection/blob/master/tools/train.py) 编写压缩脚本train.py。脚本中量化的步骤如下。

### 2.1 预训练或者下载预训练好的模型
* 用户可以在此处链接下载我们已经预训练好的的模型。[预训练模型下载](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/image_classification/README.md)

### 2.2 插入量化和反量化OP
在Program中插入量化和反量化OP。`paddleslim.quant.quant_aware` 作用是在网络中的conv2d、depthwise_conv2d、mul等算子的各个输入前插入连续的量化op和反量化op，并改变相应反向算子的某些输入。并且在一些op之后加上quant_dequant op, 示例图如下：
<!-- <p align="center">
<img src="./images/TransformPass.png" height=400 width=520 hspace='10'/> <br />
<strong>图1：应用 paddleslim.quant.quant_aware 后的结果</strong>
</p> -->
对应到代码中的更改，首先需要更改 `config` 配置如下。PaddleCV模型所需要全部`config`设置已经列出。如果想了解各参数含义，可参考 [PaddleSlim quant_aware API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/#quant_aware)

```
config = {
        'weight_quantize_type': 'channel_wise_abs_max',
        'activation_quantize_type': 'moving_average_abs_max',
        'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d','elmentwise_add','pool2d']
    }
```

然后在`train.py`中调用quant_aware

```
quant_program  = quant_aware(train_prog, place, config, for_test=False)
val_quant_program = slim.quant.quant_aware(val_program, exe.place, for_test=True)
```

### 2.2 关闭一些训练策略

因为量化要对Program做修改，所以一些会修改Program的训练策略需要关闭。``sync_batch_norm`` 和量化多卡训练同时使用时会出错，原因暂不知，因此也需要将其关闭。
```
build_strategy.fuse_all_reduce_ops = False
build_strategy.sync_batch_norm = False
```

### 2.3 边训练边测试量化后的模型

* 用户可以使用train.py进行训练
调用train函数训练分类网络，train_program是在第2步：构建网络中定义的。
`train(train_program)`
调用test函数测试分类网络，val_program是在第2步：构建网络中定义的。
`test(val_program)`


### 2.4 转化模型为fp32 qat模型

``paddleslim.quant.convert`` 主要用于改变Program中量化op和反量化op的顺序，即将类似图1中的量化op和反量化op顺序改变为图2中的布局。除此之外，``paddleslim.quant.convert`` 还会将`conv2d`、`depthwise_conv2d`、`mul`等算子参数变为量化后的int8_t范围内的值(但数据类型仍为float32)，示例如图2：
<!--
<p align="center">
<img src="./images/FreezePass.png" height=400 width=420 hspace='10'/> <br />
<strong>图2：paddleslim.quant.convert 后的结果</strong>
</p> -->

所以在调用 ``paddleslim.quant.convert`` 之后，才得到最终的量化模型。此模型可使用PaddleLite进行加载预测，可参见教程[Paddle-Lite如何加载运行量化模型](https://github.com/PaddlePaddle/Paddle-Lite/wiki/model_quantization)。

### 2.1-2.4 训练示例

我们提供了一个图像分类再训练的脚本，你可以参考这个示例直接运行。

step1: 设置gpu卡（因为使用的是Pretrained models, 再训练时，可以略过这一步，使用CPU直接训练）
```
export CUDA_VISIBLE_DEVICES=0
```
step2: 开始训练

请在PaddleDetection根目录下运行。

```
TODO 策略更改，脚本将更新
```

>通过命令行覆设置max_iters选项，因为量化的训练轮次比正常训练小很多，所以需要修改此选项。
如果要调整训练卡数，可根据需要调整配置文件`yolov3_mobilenet_v1_voc.yml`中的以下参数：

- **max_iters:** 训练的总轮次。
- **LeaningRate.base_lr:** 根据多卡的总`batch_size`调整`base_lr`，两者大小正相关，可以简单的按比例进行调整。
- **LearningRate.schedulers.PiecewiseDecay.milestones：** 请根据batch size的变化对其调整。


通过`python slim/quantization/train.py --help`查看可配置参数。
通过`python ./tools/configure.py help ${option_name}`查看如何通过命令行覆盖配置文件中的参数。


## 3. 转化fp32 qat模型为MKL-DNN INT8 模型
运行下面的脚本，注意设置`quantized_ops`

```
cd /PATH/TO/PADDLE/build
python ../python/paddle/fluid/contrib/slim/tests/save_qat_model.py --qat_model_path=/PATH/TO/DOWNLOADED/QAT/MODEL --int8_model_save_path=/PATH/TO/SAVE/QAT/INT8/MODEL --quantized_ops="conv2d,pool2d"
```

## 4. 预测

### 4.1 数据转化
在精度和性能预测中，需要先对数据进行二进制转化。在我们的测试中，我们发现为了达到最佳性能，使用c++预测会远快于python脚本，因此我们需要将图片标签等做二进制转化，以方便获得最大性能。转化方法如下，以ImageNet Set 为例。

### 4.2 使用python脚本预测精度


运行命令示例:
```
python sample_tester.py --batch_size=50 --skip_batch_num=0 --infer_model=$HOME/models/ResNet50_4th_qat_int8/ --infer_data=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin
```

### 4.3 使用c++应用预测性能
你需要从Paddle源码编译Paddle推理库，请参考从源码编译文档。

或者从Paddle官网下载发布的预测库。您需要根据需要部署的服务器的硬件配置（是否支持avx、是否使用mkl、CUDA版本、cuDNN版本），来下载对应的版本。

你可以将准备好的预测库重命名为fluid_inference，放置在该测试项目下面，也可以在cmake时通过设置PADDLE_ROOT来指定Paddle预测库的位置。
```
mkdir build
cd build
cmake -DUSE_GPU=OFF -DPADDLE_ROOT=xx/xx ..
make
```

该测试程序运行时需要配置以下参数：

infer_model，模型所在目录，注意模型参数当前必须是分开保存成多个文件的。无默认值。
infer_data，测试数据文件所在路径。无默认值。
repeat，每个样本重复执行的次数。默认值为1.
warmup_size，warmup的步数。默认值为0，即没有warmup。
batch_size，预测batch size大小。默认值为50。
iterations，预测batch数目。表示默认预测infer_data所有batches（=image size/batch size）。
num_threads，预测使用CPU 线程数。
with_accuracy_layer, 由于这个测试是Image Classification通用的测试，既可以测试float32模型也可以INT8模型，模型可以包含或者不包含label层，设置此参数更改。
profile，由Paddle预测库中提供，设置用来进行性能分析。默认值为false。

该项目提供了一个运行脚本run.sh，修改了其中的MODEL_DIR和DATA_DIR后，即可执行./run.sh进行CPU预测。

## 5. 部分PaddleCV图像分类模型在ImageNet val全集上的精度和性能

### Image classification models benchmark results

>**I. QAT2 MKL-DNN 在 Intel(R) Xeon(R) Gold 6271的精度**

|     Model    | Fake QAT Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | Fake QAT Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|:----------------------:|:----------------------:|:---------:|
| MobileNet-V1 |         70.72%         |         70.78%         |   +0.06%  |         89.47%         |         89.39%         |   -0.08%  |
| MobileNet-V2 |         72.07%         |         72.17%         |   +0.10%  |         90.65%         |         90.63%         |   -0.02%  |
|   ResNet101  |         77.86%         |         77.59%         |   -0.27%  |         93.54%         |         93.54%         |   0.00%   |
|   ResNet50   |         76.62%         |         76.53%         |   -0.09%  |         93.01%         |         92.98%         |   -0.03%  |
|     VGG16    |         71.74%         |         71.75%         |   +0.01%  |         89.96%         |         89.73%         |   -0.23%  |
|     VGG19    |         72.30%         |         72.09%         |   -0.21%  |         90.19%         |         90.13%         |   -0.06%  |


>**II. QAT2 MKL-DNN 在 Intel(R) Xeon(R) Gold 6271的单核的性能**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      73.98      |       227.73        |       3.08        |
| MobileNet-V2 |      86.59      |       206.74        |       2.39        |
|  ResNet101   |      7.15       |        26.69        |       3.73        |
|   ResNet50   |      13.15      |        49.33        |       3.75        |
|    VGG16     |      3.34       |        10.15        |       3.04        |
|    VGG19     |      2.83       |        8.67         |       3.07        |


## FAQ

该示例使用PaddleSlim提供的[量化压缩API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/)对检测模型进行压缩。
在阅读该示例前，建议您先了解以下内容：

- [检测模型的常规训练方法](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)

已发布量化模型见[压缩模型库](../README.md)
