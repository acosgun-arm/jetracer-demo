#include "color_lane_native.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: jetracer-color-lane-benchmark PROFILE IMAGE "
                 "ITERATIONS WARMUP_ITERATIONS\n";
    return EXIT_FAILURE;
  }
  try {
    return benchmark_color_lane_image(argv[1], argv[2], std::stoi(argv[3]),
                                      std::stoi(argv[4]));
  } catch (const std::exception& error) {
    std::cerr << "invalid benchmark argument: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
