#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::runtime_error error(const std::string& context, NSError* detail = nil) {
  if (detail == nil) return std::runtime_error(context);
  return std::runtime_error(context + ": " + detail.localizedDescription.UTF8String);
}

int positive_integer(const char* value, const std::string& name) {
  const int parsed = std::stoi(value);
  if (parsed <= 0) throw error(name + " must be positive");
  return parsed;
}

MLComputeUnits compute_units(const std::string& value) {
  if (value == "all") return MLComputeUnitsAll;
  if (value == "cpu_and_gpu") return MLComputeUnitsCPUAndGPU;
  if (value == "cpu_only") return MLComputeUnitsCPUOnly;
  if (value == "cpu_and_neural_engine") {
    if (@available(macOS 13.0, *)) return MLComputeUnitsCPUAndNeuralEngine;
  }
  throw error("unsupported compute units: " + value);
}

void run(int argc, const char* argv[]) {
  if (argc != 12) {
    throw error(
        "usage: coreml_compile_smoke SOURCE TARGET INPUT OUTPUT INPUT_W "
        "INPUT_H OUTPUT_W OUTPUT_H COMPUTE_UNITS WARMUP ITERATIONS");
  }
  NSString* source_path = [NSString stringWithUTF8String:argv[1]];
  NSString* target_path = [NSString stringWithUTF8String:argv[2]];
  NSString* input_name = [NSString stringWithUTF8String:argv[3]];
  NSString* output_name = [NSString stringWithUTF8String:argv[4]];
  const int input_width = positive_integer(argv[5], "input width");
  const int input_height = positive_integer(argv[6], "input height");
  const int output_width = positive_integer(argv[7], "output width");
  const int output_height = positive_integer(argv[8], "output height");
  const MLComputeUnits units = compute_units(argv[9]);
  const int warmup = positive_integer(argv[10], "warmup iterations");
  const int iterations = positive_integer(argv[11], "iterations");
  NSFileManager* manager = NSFileManager.defaultManager;
  if (![manager fileExistsAtPath:source_path]) {
    throw error("source Core ML package does not exist");
  }
  if ([manager fileExistsAtPath:target_path]) {
    throw error("target compiled Core ML model already exists");
  }

  NSError* detail = nil;
  NSURL* source_url = [NSURL fileURLWithPath:source_path isDirectory:YES];
  NSURL* compiled_url = [MLModel compileModelAtURL:source_url error:&detail];
  if (compiled_url == nil) throw error("Core ML compilation failed", detail);
  NSURL* target_url = [NSURL fileURLWithPath:target_path isDirectory:YES];
  if (![manager copyItemAtURL:compiled_url toURL:target_url error:&detail]) {
    throw error("failed to copy compiled Core ML model", detail);
  }

  MLModelConfiguration* configuration = [[MLModelConfiguration alloc] init];
  configuration.computeUnits = units;
  MLModel* model = [MLModel modelWithContentsOfURL:target_url
                                     configuration:configuration
                                             error:&detail];
  if (model == nil) throw error("failed to load compiled Core ML model", detail);
  MLMultiArray* input = [[MLMultiArray alloc]
      initWithShape:@[ @1, @3, @(input_height), @(input_width) ]
           dataType:MLMultiArrayDataTypeFloat32
              error:&detail];
  if (input == nil) throw error("failed to allocate Core ML input", detail);
  auto* values = static_cast<float*>(input.dataPointer);
  for (NSInteger index = 0; index < input.count; ++index) {
    values[index] = static_cast<float>(index % 257) / 256.0F;
  }
  MLDictionaryFeatureProvider* provider =
      [[MLDictionaryFeatureProvider alloc]
          initWithDictionary:@{input_name : input}
                       error:&detail];
  if (provider == nil) throw error("failed to create Core ML input", detail);

  std::vector<double> latencies;
  NSArray<NSNumber*>* validated_shape = nil;
  for (int index = 0; index < warmup + iterations; ++index) {
    const CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    id<MLFeatureProvider> prediction =
        [model predictionFromFeatures:provider error:&detail];
    const double latency = CFAbsoluteTimeGetCurrent() - started;
    if (prediction == nil) throw error("Core ML prediction failed", detail);
    MLMultiArray* output =
        [prediction featureValueForName:output_name].multiArrayValue;
    if (output == nil || output.shape.count != 4 ||
        output.shape[0].intValue != 1 ||
        output.shape[2].intValue != output_height ||
        output.shape[3].intValue != output_width) {
      throw error("unexpected Core ML output shape");
    }
    validated_shape = output.shape;
    if (index >= warmup) latencies.push_back(latency);
  }
  const double total =
      std::accumulate(latencies.begin(), latencies.end(), 0.0);
  const double mean = total / static_cast<double>(latencies.size());
  std::sort(latencies.begin(), latencies.end());
  const double median = latencies[latencies.size() / 2];
  NSDictionary* report = @{
    @"status" : @"passed",
    @"iterations" : @(iterations),
    @"warmup_iterations" : @(warmup),
    @"mean_latency_s" : @(mean),
    @"median_latency_s" : @(median),
    @"measured_fps" : @(1.0 / mean),
    @"output_shape" : validated_shape,
  };
  NSData* json = [NSJSONSerialization dataWithJSONObject:report
                                                 options:NSJSONWritingSortedKeys
                                                   error:&detail];
  if (json == nil) throw error("failed to encode smoke-test report", detail);
  std::cout << [[[NSString alloc] initWithData:json
                                      encoding:NSUTF8StringEncoding] UTF8String]
            << '\n';
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    try {
      run(argc, argv);
      return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
      std::cerr << "coreml_compile_smoke: " << exception.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
