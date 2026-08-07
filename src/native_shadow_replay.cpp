#include "color_lane_native.hpp"
#include "native_shadow_controller.hpp"

#include <opencv2/videoio.hpp>

#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: jetracer-shadow-replay COLOR_PROFILE SHADOW_CONFIG "
                 "VIDEO OUTPUT_JSONL\n";
    return EXIT_FAILURE;
  }
  try {
    const ColorLaneNativeConfig lane_config =
        load_color_lane_native_config(argv[1]);
    ColorLaneNativeProcessor processor(lane_config);
    NativeShadowController controller(load_native_shadow_config(argv[2]),
                                      lane_config.processing_width,
                                      lane_config.processing_height);
    cv::VideoCapture video(argv[3]);
    if (!video.isOpened()) throw std::runtime_error("cannot open replay video");
    const double fps = video.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0) {
      throw std::runtime_error("replay video has invalid frame rate");
    }
    std::ofstream output(argv[4], std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open replay telemetry output");
    cv::Mat frame;
    std::uint64_t frames = 0, tracking = 0, stopped = 0;
    double confidence_sum = 0.0, speed_sum = 0.0;
    double max_abs_steering = 0.0;
    while (video.read(frame)) {
      const ColorLaneNativeResult lane = processor.process(frame);
      const NativeShadowCommand command =
          controller.update(lane, 1.0 / fps, frames / fps);
      output << native_shadow_command_json(command) << '\n';
      ++frames;
      if (command.reason == "tracking") ++tracking;
      if (command.requested_speed_mps == 0.0) ++stopped;
      confidence_sum += command.confidence;
      speed_sum += command.estimated_speed_mps;
      max_abs_steering =
          std::max(max_abs_steering, std::abs(command.steering_rad));
    }
    if (frames == 0) throw std::runtime_error("replay video has no frames");
    std::cout << "{\n"
              << "  \"mode\": \"native_shadow_replay\",\n"
              << "  \"actuator_mode\": \"disabled\",\n"
              << "  \"actuators_accessed\": false,\n"
              << "  \"frames\": " << frames << ",\n"
              << "  \"source_fps\": " << fps << ",\n"
              << "  \"tracking_fraction\": "
              << static_cast<double>(tracking) / frames << ",\n"
              << "  \"zero_speed_fraction\": "
              << static_cast<double>(stopped) / frames << ",\n"
              << "  \"mean_confidence\": " << confidence_sum / frames
              << ",\n"
              << "  \"mean_estimated_speed_mps\": " << speed_sum / frames
              << ",\n"
              << "  \"maximum_absolute_steering_rad\": "
              << max_abs_steering << ",\n"
              << "  \"telemetry_path\": \"" << argv[4] << "\"\n"
              << "}\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "shadow replay failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
