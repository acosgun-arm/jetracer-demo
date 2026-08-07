#pragma once

#include <cstdint>
#include <string>

struct GstreamerPreprocessBenchmarkConfig {
  std::string device;
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t fps;
  std::uint32_t output_width;
  std::uint32_t output_height;
  int flip_method;
  double duration_s;
  int startup_timeout_ms;
  int frame_timeout_ms;
};

int benchmark_gstreamer_preprocess(
    const GstreamerPreprocessBenchmarkConfig& config);
