#pragma once

#include <cstdint>
#include <string>

struct CameraTransportBenchmarkConfig {
  std::string device;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t fps;
  std::string fourcc;
  double duration_s;
  std::uint32_t buffer_count;
  int startup_timeout_ms;
  int frame_timeout_ms;
};

int benchmark_camera_transport(const CameraTransportBenchmarkConfig& config);
