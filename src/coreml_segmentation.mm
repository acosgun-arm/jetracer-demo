#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "jetracer_sim/coreml_segmentation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace jetracer::sim {
namespace {

std::runtime_error coreml_error(const std::string& context, NSError* error) {
  const char* description =
      error == nil ? "unknown Core ML error" : error.localizedDescription.UTF8String;
  return std::runtime_error(context + ": " + description);
}

MLComputeUnits parse_compute_units(const std::string& value) {
  if (value == "all") return MLComputeUnitsAll;
  if (value == "cpu_and_gpu") return MLComputeUnitsCPUAndGPU;
  if (value == "cpu_only") return MLComputeUnitsCPUOnly;
  if (value == "cpu_and_neural_engine") {
    if (@available(macOS 13.0, *)) return MLComputeUnitsCPUAndNeuralEngine;
  }
  throw std::invalid_argument("unsupported Core ML compute units: " + value);
}

std::size_t dimension(MLMultiArray* array, NSUInteger index) {
  return static_cast<std::size_t>(array.shape[index].unsignedLongLongValue);
}

std::size_t stride(MLMultiArray* array, NSUInteger index) {
  return static_cast<std::size_t>(array.strides[index].unsignedLongLongValue);
}

float output_value(MLMultiArray* array, std::size_t index) {
  switch (array.dataType) {
    case MLMultiArrayDataTypeFloat32:
      return static_cast<const float*>(array.dataPointer)[index];
    case MLMultiArrayDataTypeDouble:
      return static_cast<float>(static_cast<const double*>(array.dataPointer)[index]);
    case MLMultiArrayDataTypeFloat16:
      return static_cast<float>(
          static_cast<const _Float16*>(array.dataPointer)[index]);
    default:
      throw std::runtime_error("Core ML logits must be float16, float32, or double");
  }
}

}  // namespace

struct CoreMLSegmentationSession::Impl {
  MLModel* model = nil;
  NSString* input_name = nil;
  NSString* output_name = nil;
  int input_width = 0;
  int input_height = 0;
  int output_width = 0;
  int output_height = 0;
  std::vector<int> source_road_class_ids;
  std::uint8_t road_class_id = 1;
  float input_scale = 1.0F;
  std::array<float, 3> mean_rgb{};
  std::array<float, 3> std_rgb{};
  std::mutex mutex;
};

CoreMLSegmentationSession::CoreMLSegmentationSession(
    const std::string& model_path, const std::string& input_name,
    const std::string& output_name, int input_width, int input_height,
    int output_width, int output_height,
    std::vector<int> source_road_class_ids, std::uint8_t road_class_id,
    float input_scale, std::vector<float> mean_rgb,
    std::vector<float> std_rgb, const std::string& compute_units)
    : impl_(std::make_unique<Impl>()) {
  if (model_path.empty() || input_name.empty() || output_name.empty()) {
    throw std::invalid_argument("Core ML model path and feature names are required");
  }
  if (input_width <= 0 || input_height <= 0 || output_width <= 0 ||
      output_height <= 0) {
    throw std::invalid_argument("Core ML tensor dimensions must be positive");
  }
  if (source_road_class_ids.empty() ||
      std::any_of(source_road_class_ids.begin(), source_road_class_ids.end(),
                  [](int value) { return value < 0; })) {
    throw std::invalid_argument("Core ML source road classes are invalid");
  }
  if (road_class_id == 0 || input_scale <= 0.0F || mean_rgb.size() != 3 ||
      std_rgb.size() != 3 ||
      std::any_of(std_rgb.begin(), std_rgb.end(),
                  [](float value) { return value <= 0.0F; })) {
    throw std::invalid_argument("Core ML preprocessing configuration is invalid");
  }

  impl_->input_width = input_width;
  impl_->input_height = input_height;
  impl_->output_width = output_width;
  impl_->output_height = output_height;
  impl_->source_road_class_ids = std::move(source_road_class_ids);
  impl_->road_class_id = road_class_id;
  impl_->input_scale = input_scale;
  std::copy(mean_rgb.begin(), mean_rgb.end(), impl_->mean_rgb.begin());
  std::copy(std_rgb.begin(), std_rgb.end(), impl_->std_rgb.begin());

  @autoreleasepool {
    NSString* path = [NSString stringWithUTF8String:model_path.c_str()];
    NSURL* url = [NSURL fileURLWithPath:path isDirectory:YES];
    MLModelConfiguration* configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = parse_compute_units(compute_units);
    NSError* error = nil;
    MLModel* model = [MLModel modelWithContentsOfURL:url
                                      configuration:configuration
                                              error:&error];
    if (model == nil) throw coreml_error("failed to load compiled Core ML model", error);
    impl_->model = model;
    impl_->input_name = [NSString stringWithUTF8String:input_name.c_str()];
    impl_->output_name = [NSString stringWithUTF8String:output_name.c_str()];
  }
}

CoreMLSegmentationSession::~CoreMLSegmentationSession() = default;
CoreMLSegmentationSession::CoreMLSegmentationSession(
    CoreMLSegmentationSession&&) noexcept = default;
CoreMLSegmentationSession& CoreMLSegmentationSession::operator=(
    CoreMLSegmentationSession&&) noexcept = default;

std::vector<std::uint8_t> CoreMLSegmentationSession::infer(
    const std::uint8_t* image_bgr, int image_width, int image_height) {
  if (image_bgr == nullptr || image_width <= 0 || image_height <= 0) {
    throw std::invalid_argument("Core ML input image is invalid");
  }
  std::scoped_lock lock(impl_->mutex);
  @autoreleasepool {
    NSError* error = nil;
    MLMultiArray* input = [[MLMultiArray alloc]
        initWithShape:@[ @1, @3, @(impl_->input_height), @(impl_->input_width) ]
             dataType:MLMultiArrayDataTypeFloat32
                error:&error];
    if (input == nil) throw coreml_error("failed to allocate Core ML input", error);
    auto* input_data = static_cast<float*>(input.dataPointer);
    const std::array<std::size_t, 4> input_strides{
        stride(input, 0), stride(input, 1), stride(input, 2), stride(input, 3)};
    for (int y = 0; y < impl_->input_height; ++y) {
      const int source_y = std::min(
          y * image_height / impl_->input_height, image_height - 1);
      for (int x = 0; x < impl_->input_width; ++x) {
        const int source_x = std::min(
            x * image_width / impl_->input_width, image_width - 1);
        const std::size_t pixel =
            (static_cast<std::size_t>(source_y) * image_width + source_x) * 3;
        for (std::size_t channel = 0; channel < 3; ++channel) {
          const float value =
              static_cast<float>(image_bgr[pixel + (2 - channel)]) *
              impl_->input_scale;
          const std::size_t tensor_index = channel * input_strides[1] +
              static_cast<std::size_t>(y) * input_strides[2] +
              static_cast<std::size_t>(x) * input_strides[3];
          input_data[tensor_index] =
              (value - impl_->mean_rgb[channel]) / impl_->std_rgb[channel];
        }
      }
    }

    MLDictionaryFeatureProvider* provider =
        [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{impl_->input_name : input}
                         error:&error];
    if (provider == nil) throw coreml_error("failed to create Core ML input", error);
    id<MLFeatureProvider> prediction =
        [impl_->model predictionFromFeatures:provider error:&error];
    if (prediction == nil) throw coreml_error("Core ML prediction failed", error);
    MLFeatureValue* feature = [prediction featureValueForName:impl_->output_name];
    MLMultiArray* logits = feature.multiArrayValue;
    if (logits == nil || logits.shape.count != 4 || dimension(logits, 0) != 1 ||
        dimension(logits, 2) != static_cast<std::size_t>(impl_->output_height) ||
        dimension(logits, 3) != static_cast<std::size_t>(impl_->output_width)) {
      throw std::runtime_error("Core ML output shape does not match configuration");
    }
    const std::size_t class_count = dimension(logits, 1);
    if (std::any_of(impl_->source_road_class_ids.begin(),
                    impl_->source_road_class_ids.end(),
                    [class_count](int value) {
                      return static_cast<std::size_t>(value) >= class_count;
                    })) {
      throw std::runtime_error("Core ML road class is outside the output tensor");
    }
    const std::array<std::size_t, 4> output_strides{
        stride(logits, 0), stride(logits, 1), stride(logits, 2), stride(logits, 3)};
    std::vector<std::uint8_t> labels(
        static_cast<std::size_t>(impl_->output_width) * impl_->output_height, 0);
    for (int y = 0; y < impl_->output_height; ++y) {
      for (int x = 0; x < impl_->output_width; ++x) {
        float maximum = -std::numeric_limits<float>::infinity();
        int maximum_class = 0;
        for (std::size_t class_id = 0; class_id < class_count; ++class_id) {
          const std::size_t index = class_id * output_strides[1] +
              static_cast<std::size_t>(y) * output_strides[2] +
              static_cast<std::size_t>(x) * output_strides[3];
          const float value = output_value(logits, index);
          if (value > maximum) {
            maximum = value;
            maximum_class = static_cast<int>(class_id);
          }
        }
        if (std::find(impl_->source_road_class_ids.begin(),
                      impl_->source_road_class_ids.end(), maximum_class) !=
            impl_->source_road_class_ids.end()) {
          labels[static_cast<std::size_t>(y) * impl_->output_width + x] =
              impl_->road_class_id;
        }
      }
    }
    return labels;
  }
}

int CoreMLSegmentationSession::output_width() const noexcept {
  return impl_->output_width;
}

int CoreMLSegmentationSession::output_height() const noexcept {
  return impl_->output_height;
}

}  // namespace jetracer::sim
