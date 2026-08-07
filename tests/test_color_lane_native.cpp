#include "color_lane_native.hpp"

#include <opencv2/core.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "expected color-lane profile path\n";
    return EXIT_FAILURE;
  }
  try {
    const ColorLaneNativeConfig config =
        load_color_lane_native_config(argv[1]);
    require(config.profile_id == "waveshare-sim-white", "wrong profile ID");
    require(config.processing_width == 640, "wrong processing width");
    require(config.processing_height == 360, "wrong processing height");
    ColorLaneNativeProcessor processor(config);
    cv::Mat image = cv::Mat::zeros(config.processing_height,
                                   config.processing_width, CV_8UC3);
    for (int row = 0; row < image.rows; ++row) {
      const int offset = static_cast<int>(20.0 * row * row /
                                          (image.rows * image.rows));
      image.row(row).colRange(100 + offset, 108 + offset).setTo(cv::Scalar::all(255));
      image.row(row).colRange(500 + offset, 508 + offset).setTo(cv::Scalar::all(255));
    }
    const ColorLaneNativeResult result = processor.process(image);
    require(result.labels.type() == CV_8UC1, "wrong output type");
    require(result.labels.at<std::uint8_t>(300, 320) == config.road_class_id,
            "road center is not classified");
    require(result.labels.at<std::uint8_t>(300, 20) == 0,
            "outside pixel is classified as road");
    require(result.confidence > 0.8, "unexpectedly low fit confidence");
    require(result.center_path_normalized.size() ==
                static_cast<std::size_t>(config.path_sample_count),
            "wrong path sample count");
    std::cout << "native color-lane tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
