/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <paddle_inference_api.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#ifdef WITH_GPERFTOOLS
#include <paddle/fluid/platform/profiler.h>
#include <gperftools/profiler.h>
DECLARE_bool(profile);
#endif


DEFINE_string(infer_model, "", "model directory");
DEFINE_string(infer_data, "", "input data path");
DEFINE_int32(repeat, 1, "repeat");
DEFINE_int32(warmup_size, 0, "warm up samples");
DEFINE_int32(batch_size, 50, "batch size");
DEFINE_int32(iterations, 0, "number of batches to process, by default test whole set");
DEFINE_int32(num_threads, 1, "num_threads");
DEFINE_bool(with_accuracy_layer, true, "label is required in the input of the inference model");
DEFINE_bool(record_benchmark, false, "Record benchmark after profiling the model");
DEFINE_bool(use_profile, false, "Do profile or not");
static void SetConfig(paddle::AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->SwitchSpecifyInputNames();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_num_threads);
  cfg->EnableMKLDNN();
}

struct Timer {
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(startu -
          start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

// template <typename T>
// constexpr paddle::PaddleDType GetPaddleDType();

// template <>
// constexpr paddle::PaddleDType GetPaddleDType<int64_t>() {
//   return paddle::PaddleDType::INT64;
// }

// template <>
// constexpr paddle::PaddleDType GetPaddleDType<float>() {
//   return paddle::PaddleDType::FLOAT32;
// }

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file, size_t beginning_offset,
               std::vector<int> shape, std::string name)
      : file_(file), position_(beginning_offset), shape_(shape), name_(name) {
    numel_ = std::accumulate(shape_.begin(), shape_.end(), size_t{1},
                             std::multiplies<size_t>());
  }

  paddle::PaddleTensor NextBatch() {
    paddle::PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape_;
    // tensor.dtype = GetPaddleDType<T>();
    if (std::is_same<T, int64_t>::value){
      tensor.dtype = paddle::PaddleDType::INT64;
    }else if(std::is_same<T, float>::value){
      tensor.dtype = paddle::PaddleDType::FLOAT32;
    }else{
      throw std::runtime_error("Converting tensor type failed");
    }

    file_.seekg(position_);
    file_.read(static_cast<char *>(tensor.data.data()), numel_ * sizeof(T));
    position_ = file_.tellg();

    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");

    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position_;
  std::vector<int> shape_;
  std::string name_;
  size_t numel_;
};

std::shared_ptr<std::vector<paddle::PaddleTensor>> GetWarmupData(
    const std::vector<std::vector<paddle::PaddleTensor>> &test_data,
    int num_images = FLAGS_warmup_size) {
  int test_data_batch_size = test_data[0][0].shape[0];
  auto iterations = test_data.size();
  auto all_test_data_size = iterations * test_data_batch_size;
  // PADDLE_ENFORCE_LE(static_cast<size_t>(num_images), all_test_data_size,
  //                   paddle::platform::errors::InvalidArgument(
  //                       "The requested quantization warmup data size must be "
  //                       "lower or equal to the test data size. But received"
  //                       "warmup size is %d and test data size is %d. Please "
  //                       "use --warmup_batch_size parameter to set smaller "
  //                       "warmup batch size.",
  //                       num_images, all_test_data_size));

  paddle::PaddleTensor images;
  images.name = "image";
  images.shape = {num_images, 3, 224, 224};
  images.dtype = paddle::PaddleDType::FLOAT32;
  images.data.Resize(sizeof(float) * num_images * 3 * 224 * 224);

  paddle::PaddleTensor labels;
  labels.name = "label";
  labels.shape = {num_images, 1};
  labels.dtype = paddle::PaddleDType::INT64;
  labels.data.Resize(sizeof(int64_t) * num_images);

  for (int i = 0; i < num_images; i++) {
    auto batch = i / test_data_batch_size;
    auto element_in_batch = i % test_data_batch_size;
    std::copy_n(static_cast<float *>(test_data[batch][0].data.data()) +
                    element_in_batch * 3 * 224 * 224,
                3 * 224 * 224,
                static_cast<float *>(images.data.data()) + i * 3 * 224 * 224);

    std::copy_n(static_cast<int64_t *>(test_data[batch][1].data.data()) +
                    element_in_batch,
                1, static_cast<int64_t *>(labels.data.data()) + i);
  }

  auto warmup_data = std::make_shared<std::vector<paddle::PaddleTensor>>(2);
  (*warmup_data)[0] = std::move(images);
  (*warmup_data)[1] = std::move(labels);
  return warmup_data;
}

void SetInput(std::vector<std::vector<paddle::PaddleTensor>> *inputs,
              bool with_accuracy_layer = FLAGS_with_accuracy_layer,
              int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Couldn't open file: " + FLAGS_infer_data);
  }

  int64_t total_images{0};
  file.read(reinterpret_cast<char *>(&total_images), sizeof(total_images));
  LOG(INFO) << "Total images in file: " << total_images;

  std::vector<int> image_batch_shape{batch_size, 3, 224, 224};
  std::vector<int> label_batch_shape{batch_size, 1};
  auto images_offset_in_file = static_cast<size_t>(file.tellg());

  TensorReader<float> image_reader(file, images_offset_in_file,
                                   image_batch_shape, "image");

  auto iterations_max = total_images / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }

  auto labels_offset_in_file =
      images_offset_in_file + sizeof(float) * total_images * 3 * 224 * 224;

  TensorReader<int64_t> label_reader(file, labels_offset_in_file,
                                     label_batch_shape, "label");
  for (auto i = 0; i < iterations; i++) {
    auto images = image_reader.NextBatch();
    std::vector<paddle::PaddleTensor> tmp_vec;
    tmp_vec.push_back(std::move(images));
    if (with_accuracy_layer) {
      auto labels = label_reader.NextBatch();
      tmp_vec.push_back(std::move(labels));
    }
    inputs->push_back(std::move(tmp_vec));
  }
}

static void PrintTime(int batch_size, int repeat, int num_threads, int tid,
                      double batch_latency, int epoch = 1) {
  // PADDLE_ENFORCE_GT(batch_size, 0, "Non-positive batch size.");
  double sample_latency = batch_latency / batch_size;
  LOG(INFO) << "====== threads: " << num_threads << ", thread id: " << tid
            << " ======";
  LOG(INFO) << "====== batch size: " << batch_size << ", iterations: " << epoch
            << ", repetitions: " << repeat << " ======";
  LOG(INFO) << "====== batch latency: " << batch_latency
            << "ms, number of samples: " << batch_size * epoch
            << ", sample latency: " << sample_latency
            << "ms, fps: " << 1000.f / sample_latency << " ======";
}

std::unique_ptr<paddle::PaddlePredictor> CreatePredictor(
        const paddle::PaddlePredictor::Config *config, bool use_analysis = true) {
  const auto *analysis_config = reinterpret_cast<const paddle::AnalysisConfig *>(config);
  if (use_analysis) {
    return paddle::CreatePaddlePredictor<paddle::AnalysisConfig>(*analysis_config);
  }
  auto native_config = analysis_config->ToNativeConfig();
  return paddle::CreatePaddlePredictor<paddle::NativeConfig>(native_config);
}

static void PredictionWarmUp(paddle::PaddlePredictor *predictor,
                      const std::vector<std::vector<paddle::PaddleTensor>> &inputs,
                      std::vector<std::vector<paddle::PaddleTensor>> *outputs,
                      int num_threads, int tid) {
  int batch_size = FLAGS_batch_size;
  LOG(INFO) << "Running thread " << tid << ", warm up run...";
  outputs->resize(1);
  Timer warmup_timer;
  warmup_timer.tic();
  predictor->Run(inputs[0], &(*outputs)[0], batch_size);
  PrintTime(batch_size, 1, num_threads, tid, warmup_timer.toc(), 1);
  #ifdef WITH_GPERFTOOLS
  if (FLAGS_use_profile) {
    paddle::platform::ResetProfiler();
  }
  #endif
}

void PredictionRun(paddle::PaddlePredictor *predictor,
                   const std::vector<std::vector<paddle::PaddleTensor>> &inputs,
                   std::vector<std::vector<paddle::PaddleTensor>> *outputs,
                   int num_threads, int tid,
                   float *sample_latency = nullptr) {
  int num_times = FLAGS_repeat;
  int iterations = inputs.size();  // process the whole dataset ...
  if (FLAGS_iterations > 0 &&
      FLAGS_iterations < static_cast<int64_t>(inputs.size()))
    iterations =
        FLAGS_iterations;  // ... unless the number of iterations is set
  outputs->resize(iterations);
  LOG(INFO) << "Thread " << tid << ", number of threads " << num_threads
            << ", run " << num_times << " times...";
  Timer run_timer;
  double elapsed_time = 0;
#ifdef WITH_GPERFTOOLS
  ProfilerStart("paddle_inference.prof");
#endif
  int predicted_num = 0;
  
  for (int i = 0; i < iterations; i++) {
    run_timer.tic();
    for (int j = 0; j < num_times; j++) {
      predictor->Run(inputs[i], &(*outputs)[i], FLAGS_batch_size);
    }
    elapsed_time += run_timer.toc();

    predicted_num += FLAGS_batch_size;
    if (predicted_num % 100 == 0) {
      LOG(INFO) << predicted_num << " samples";
    }
  }

#ifdef WITH_GPERFTOOLS
  ProfilerStop();
#endif

  auto batch_latency = elapsed_time / (iterations * num_times);
  PrintTime(FLAGS_batch_size, num_times, num_threads, tid, batch_latency,
            iterations);

  if (sample_latency != nullptr)
    *sample_latency = batch_latency / FLAGS_batch_size;
}

int main(int argc, char *argv[]) {
  // InitFLAGS(argc, argv);
  google::InitGoogleLogging(*argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  paddle::AnalysisConfig cfg;
  SetConfig(&cfg);
  // read data from file and prepare batches with test data
  std::vector<std::vector<paddle::PaddleTensor>> input_slots_all;
  std::vector<std::vector<paddle::PaddleTensor>> outputs;
  SetInput(&input_slots_all);
  
  auto predictor = CreatePredictor(reinterpret_cast<paddle::PaddlePredictor::Config *>(&cfg), true);
  if(FLAGS_warmup_size){
    PredictionWarmUp(predictor.get(), input_slots_all, &outputs, 1, 0);
  }
  PredictionRun(predictor.get(), input_slots_all, &outputs, 1, 0);
}

