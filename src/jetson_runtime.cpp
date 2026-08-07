#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "v4l2_capture.hpp"
#include "gstreamer_color_lane.hpp"
#include "gstreamer_preprocess.hpp"

namespace {

const char* cuda_error_name(cudaError_t result) {
  return result == cudaSuccess ? "none" : cudaGetErrorName(result);
}

int ioctl_retry(int fd, unsigned long request, void* argument) {
  int result = 0;
  do {
    result = ioctl(fd, request, argument);
  } while (result < 0 && errno == EINTR);
  return result;
}

std::string fourcc_string(std::uint32_t value) {
  std::string result(4, ' ');
  for (int index = 0; index < 4; ++index) {
    result[static_cast<std::size_t>(index)] =
        static_cast<char>((value >> (8 * index)) & 0xffU);
  }
  return result;
}

int probe_camera(const char* path) {
  const int fd = open(path, O_RDWR | O_NONBLOCK);
  if (fd < 0) {
    std::cerr << "cannot open " << path << ": " << std::strerror(errno) << "\n";
    return EXIT_FAILURE;
  }

  v4l2_capability capability{};
  if (ioctl_retry(fd, VIDIOC_QUERYCAP, &capability) < 0) {
    std::cerr << "VIDIOC_QUERYCAP failed: " << std::strerror(errno) << "\n";
    close(fd);
    return EXIT_FAILURE;
  }

  std::cout << "device=" << path << " card=" << capability.card
            << " driver=" << capability.driver << "\n";
  for (std::uint32_t format_index = 0;; ++format_index) {
    v4l2_fmtdesc format{};
    format.index = format_index;
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl_retry(fd, VIDIOC_ENUM_FMT, &format) < 0) break;
    std::cout << "format=" << fourcc_string(format.pixelformat) << "\n";

    for (std::uint32_t size_index = 0;; ++size_index) {
      v4l2_frmsizeenum size{};
      size.index = size_index;
      size.pixel_format = format.pixelformat;
      if (ioctl_retry(fd, VIDIOC_ENUM_FRAMESIZES, &size) < 0) break;
      if (size.type != V4L2_FRMSIZE_TYPE_DISCRETE) {
        std::cout << "  size=continuous_or_stepwise\n";
        break;
      }

      std::cout << "  size=" << size.discrete.width << "x"
                << size.discrete.height << " fps=";
      bool found_interval = false;
      for (std::uint32_t interval_index = 0;; ++interval_index) {
        v4l2_frmivalenum interval{};
        interval.index = interval_index;
        interval.pixel_format = format.pixelformat;
        interval.width = size.discrete.width;
        interval.height = size.discrete.height;
        if (ioctl_retry(fd, VIDIOC_ENUM_FRAMEINTERVALS, &interval) < 0) break;
        if (interval.type != V4L2_FRMIVAL_TYPE_DISCRETE) {
          std::cout << "continuous_or_stepwise";
          found_interval = true;
          break;
        }
        if (found_interval) std::cout << ",";
        const double fps =
            static_cast<double>(interval.discrete.denominator) /
            static_cast<double>(interval.discrete.numerator);
        std::cout << std::fixed << std::setprecision(3) << fps;
        found_interval = true;
      }
      if (!found_interval) std::cout << "unknown";
      std::cout << "\n";
    }
  }
  close(fd);
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  const bool self_test = argc == 2 && std::strcmp(argv[1], "--self-test") == 0;
  const bool camera_probe =
      argc == 3 && std::strcmp(argv[1], "--camera-probe") == 0;
  if (camera_probe) return probe_camera(argv[2]);
  const bool camera_benchmark =
      argc == 11 && std::strcmp(argv[1], "--camera-benchmark") == 0;
  if (camera_benchmark) {
    try {
      return benchmark_camera_transport(CameraTransportBenchmarkConfig{
          argv[2], static_cast<std::uint32_t>(std::stoul(argv[3])),
          static_cast<std::uint32_t>(std::stoul(argv[4])),
          static_cast<std::uint32_t>(std::stoul(argv[5])), argv[6],
          std::stod(argv[7]),
          static_cast<std::uint32_t>(std::stoul(argv[8])), std::stoi(argv[9]),
          std::stoi(argv[10])});
    } catch (const std::exception& error) {
      std::cerr << "invalid camera benchmark argument: " << error.what() << "\n";
      return EXIT_FAILURE;
    }
  }
  const bool preprocess_benchmark =
      argc == 12 && std::strcmp(argv[1], "--preprocess-benchmark") == 0;
  if (preprocess_benchmark) {
    try {
      return benchmark_gstreamer_preprocess(GstreamerPreprocessBenchmarkConfig{
          argv[2], static_cast<std::uint32_t>(std::stoul(argv[3])),
          static_cast<std::uint32_t>(std::stoul(argv[4])),
          static_cast<std::uint32_t>(std::stoul(argv[5])),
          static_cast<std::uint32_t>(std::stoul(argv[6])),
          static_cast<std::uint32_t>(std::stoul(argv[7])), std::stoi(argv[8]),
          std::stod(argv[9]), std::stoi(argv[10]), std::stoi(argv[11])});
    } catch (const std::exception& error) {
      std::cerr << "invalid preprocess benchmark argument: " << error.what()
                << "\n";
      return EXIT_FAILURE;
    }
  }
  const bool color_lane_camera_benchmark =
      argc == 11 &&
      std::strcmp(argv[1], "--color-lane-camera-benchmark") == 0;
  if (color_lane_camera_benchmark) {
    try {
      return benchmark_gstreamer_color_lane(GstreamerColorLaneBenchmarkConfig{
          argv[2], argv[3], static_cast<std::uint32_t>(std::stoul(argv[4])),
          static_cast<std::uint32_t>(std::stoul(argv[5])),
          static_cast<std::uint32_t>(std::stoul(argv[6])), std::stoi(argv[7]),
          std::stod(argv[8]), std::stoi(argv[9]), std::stoi(argv[10])});
    } catch (const std::exception& error) {
      std::cerr << "invalid color-lane camera benchmark argument: "
                << error.what() << "\n";
      return EXIT_FAILURE;
    }
  }
  if (argc > 2 || (argc == 2 && !self_test)) {
    std::cerr << "usage: jetracer-jetson-runtime [--self-test | "
                 "--camera-probe DEVICE | --camera-benchmark DEVICE WIDTH "
                 "HEIGHT FPS FOURCC DURATION_SECONDS BUFFER_COUNT "
                 "STARTUP_TIMEOUT_MS FRAME_TIMEOUT_MS | "
                 "--preprocess-benchmark DEVICE INPUT_WIDTH INPUT_HEIGHT FPS "
                 "OUTPUT_WIDTH OUTPUT_HEIGHT FLIP_METHOD DURATION_SECONDS "
                 "STARTUP_TIMEOUT_MS FRAME_TIMEOUT_MS | "
                 "--color-lane-camera-benchmark PROFILE "
                 "DEVICE INPUT_WIDTH INPUT_HEIGHT FPS FLIP_METHOD "
                 "DURATION_SECONDS STARTUP_TIMEOUT_MS FRAME_TIMEOUT_MS]\n";
    return EXIT_FAILURE;
  }

  int runtime_version = 0;
  const cudaError_t version_result = cudaRuntimeGetVersion(&runtime_version);
  int device_count = 0;
  const cudaError_t device_result = cudaGetDeviceCount(&device_count);

  std::cout << "{\n"
            << "  \"runtime\": \"jetracer-jetson\",\n"
            << "  \"mode\": \"headless_safe_probe\",\n"
            << "  \"actuators_accessed\": false,\n"
            << "  \"tensorrt_version\": \"" << NV_TENSORRT_MAJOR << "."
            << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH << "\",\n"
            << "  \"cuda_runtime_version\": " << runtime_version << ",\n"
            << "  \"cuda_version_error\": \""
            << cuda_error_name(version_result) << "\",\n"
            << "  \"cuda_device_count\": " << device_count << ",\n"
            << "  \"cuda_device_error\": \""
            << cuda_error_name(device_result) << "\",\n"
            << "  \"ready\": "
            << (version_result == cudaSuccess && device_result == cudaSuccess &&
                        device_count > 0
                    ? "true"
                    : "false")
            << "\n}\n";

  if (self_test && (version_result != cudaSuccess ||
                    device_result != cudaSuccess || device_count < 1)) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
