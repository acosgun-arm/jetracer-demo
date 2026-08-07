#include "color_lane_native.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

double percentile(const std::vector<double>& sorted, double probability) {
  if (sorted.empty()) return 0.0;
  const double position = probability * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

}  // namespace

int benchmark_color_lane_image(const std::string& profile_path,
                               const std::string& image_path,
                               int iterations, int warmup_iterations) {
  if (iterations <= 0 || warmup_iterations < 0) {
    std::cerr << "invalid color-lane image benchmark iterations\n";
    return EXIT_FAILURE;
  }
  try {
    ColorLaneNativeProcessor processor(
        load_color_lane_native_config(profile_path));
    const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("cannot load benchmark image");
    ColorLaneNativeResult result;
    for (int index = 0; index < warmup_iterations; ++index) {
      result = processor.process(image);
    }
    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(iterations));
    for (int index = 0; index < iterations; ++index) {
      const auto started = std::chrono::steady_clock::now();
      result = processor.process(image);
      const auto completed = std::chrono::steady_clock::now();
      latencies_ms.push_back(
          std::chrono::duration<double, std::milli>(completed - started).count());
    }
    std::sort(latencies_ms.begin(), latencies_ms.end());
    const double mean = std::accumulate(latencies_ms.begin(), latencies_ms.end(),
                                        0.0) /
                        latencies_ms.size();
    std::cout << std::fixed << std::setprecision(6)
              << "{\n"
              << "  \"mode\": \"native_color_lane_image_benchmark\",\n"
              << "  \"actuators_accessed\": false,\n"
              << "  \"profile_id\": \"" << processor.config().profile_id
              << "\",\n"
              << "  \"iterations\": " << iterations << ",\n"
              << "  \"warmup_iterations\": " << warmup_iterations << ",\n"
              << "  \"mean_latency_ms\": " << mean << ",\n"
              << "  \"p50_latency_ms\": "
              << percentile(latencies_ms, 0.50) << ",\n"
              << "  \"p95_latency_ms\": "
              << percentile(latencies_ms, 0.95) << ",\n"
              << "  \"p99_latency_ms\": "
              << percentile(latencies_ms, 0.99) << ",\n"
              << "  \"maximum_latency_ms\": " << latencies_ms.back() << ",\n"
              << "  \"throughput_fps\": " << 1000.0 / mean << ",\n"
              << "  \"confidence\": " << result.confidence << ",\n"
              << "  \"observed_rows\": " << result.observed_rows << ",\n"
              << "  \"path_points\": "
              << result.center_path_normalized.size() << "\n"
              << "}\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "color-lane image benchmark failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
