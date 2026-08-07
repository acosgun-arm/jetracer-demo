#pragma once

#include <cstdint>
#include <string>

struct GstreamerColorLaneBenchmarkConfig {
  std::string profile_path;
  std::string device;
  std::uint32_t input_width;
  std::uint32_t input_height;
  std::uint32_t fps;
  int flip_method;
  double duration_s;
  int startup_timeout_ms;
  int frame_timeout_ms;
};

int benchmark_gstreamer_color_lane(
    const GstreamerColorLaneBenchmarkConfig& config);
