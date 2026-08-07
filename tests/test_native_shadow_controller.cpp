#include "native_shadow_controller.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
  if (argc != 2) return EXIT_FAILURE;
  try {
    NativeShadowConfig config = load_native_shadow_config(argv[1]);
    require(config.actuator_mode == "disabled", "actuator mode is not disabled");
    DisabledActuatorSink sink;
    NativeShadowCommand recorded;
    sink.record(recorded);
    require(sink.command_count() == 1, "disabled sink did not record command");
    require(!sink.hardware_accessed(), "disabled sink reported hardware access");

    NativeShadowController controller(config, 640, 360);
    ColorLaneNativeResult centered;
    centered.confidence = 1.0;
    for (int row = 340; row >= 110; row -= 10) {
      centered.center_path_normalized.emplace_back(0.5F, row / 360.0F);
    }
    NativeShadowCommand straight = controller.update(centered, 0.005, 0.0);
    require(straight.reason == "tracking", "centered path was not tracked");
    require(straight.projected_points >= 7, "too few projected path points");
    require(std::abs(straight.steering_rad) < 0.02,
            "centered path generated excessive steering");
    require(!straight.actuator_write_attempted, "controller attempted actuator write");

    ColorLaneNativeResult missing;
    NativeShadowCommand stop = controller.update(missing, 0.5, 0.5);
    require(stop.requested_speed_mps == 0.0, "road loss did not request stop");
    require(stop.reason == "road_not_found", "wrong road-loss reason");

    NativeShadowController offset_controller(config, 640, 360);
    ColorLaneNativeResult offset = centered;
    for (cv::Point2f& point : offset.center_path_normalized) point.x += 0.08F;
    NativeShadowCommand turn = offset_controller.update(offset, 0.02, 0.0);
    require(turn.steering_rad < 0.0,
            "rightward image path did not generate right steering");
    std::cout << "native shadow controller tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
