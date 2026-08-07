#include "v4l2_capture.hpp"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

struct MappedBuffer {
  void* address{MAP_FAILED};
  std::size_t length{0};
};

int ioctl_retry(int fd, unsigned long request, void* argument) {
  int result = 0;
  do {
    result = ioctl(fd, request, argument);
  } while (result < 0 && errno == EINTR);
  return result;
}

std::uint32_t fourcc_value(const std::string& value) {
  return v4l2_fourcc(value[0], value[1], value[2], value[3]);
}

std::string fourcc_string(std::uint32_t value) {
  std::string result(4, ' ');
  for (int index = 0; index < 4; ++index) {
    result[static_cast<std::size_t>(index)] =
        static_cast<char>((value >> (8 * index)) & 0xffU);
  }
  return result;
}

void release_buffers(int fd, std::vector<MappedBuffer>& buffers,
                     bool streaming) {
  if (streaming) {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl_retry(fd, VIDIOC_STREAMOFF, &type);
  }
  for (const MappedBuffer& buffer : buffers) {
    if (buffer.address != MAP_FAILED) munmap(buffer.address, buffer.length);
  }
  if (fd >= 0) close(fd);
}

int fail(int fd, std::vector<MappedBuffer>& buffers, bool streaming,
         const char* operation) {
  const int saved_errno = errno;
  release_buffers(fd, buffers, streaming);
  std::cerr << operation << " failed: " << std::strerror(saved_errno) << "\n";
  return EXIT_FAILURE;
}

}  // namespace

int benchmark_camera_transport(const CameraTransportBenchmarkConfig& config) {
  if (config.device.empty() || config.width == 0 || config.height == 0 ||
      config.fps == 0 || config.fourcc.size() != 4 ||
      !std::isfinite(config.duration_s) || config.duration_s <= 0.0 ||
      config.buffer_count < 2 || config.startup_timeout_ms <= 0 ||
      config.frame_timeout_ms <= 0) {
    std::cerr << "invalid camera transport benchmark configuration\n";
    return EXIT_FAILURE;
  }

  const int fd = open(config.device.c_str(), O_RDWR | O_NONBLOCK);
  std::vector<MappedBuffer> buffers;
  bool streaming = false;
  if (fd < 0) return fail(fd, buffers, streaming, "open camera");

  v4l2_capability capability{};
  if (ioctl_retry(fd, VIDIOC_QUERYCAP, &capability) < 0) {
    return fail(fd, buffers, streaming, "VIDIOC_QUERYCAP");
  }
  const std::uint32_t capabilities =
      capability.capabilities & V4L2_CAP_DEVICE_CAPS
          ? capability.device_caps
          : capability.capabilities;
  if (!(capabilities & V4L2_CAP_VIDEO_CAPTURE) ||
      !(capabilities & V4L2_CAP_STREAMING)) {
    errno = ENOTSUP;
    return fail(fd, buffers, streaming, "camera streaming capability");
  }

  v4l2_format format{};
  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  format.fmt.pix.width = config.width;
  format.fmt.pix.height = config.height;
  format.fmt.pix.pixelformat = fourcc_value(config.fourcc);
  format.fmt.pix.field = V4L2_FIELD_ANY;
  if (ioctl_retry(fd, VIDIOC_S_FMT, &format) < 0) {
    return fail(fd, buffers, streaming, "VIDIOC_S_FMT");
  }

  v4l2_streamparm parameters{};
  parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  parameters.parm.capture.timeperframe.numerator = 1;
  parameters.parm.capture.timeperframe.denominator = config.fps;
  if (ioctl_retry(fd, VIDIOC_S_PARM, &parameters) < 0) {
    return fail(fd, buffers, streaming, "VIDIOC_S_PARM");
  }
  if (ioctl_retry(fd, VIDIOC_G_PARM, &parameters) < 0) {
    return fail(fd, buffers, streaming, "VIDIOC_G_PARM");
  }

  v4l2_requestbuffers request{};
  request.count = config.buffer_count;
  request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  request.memory = V4L2_MEMORY_MMAP;
  if (ioctl_retry(fd, VIDIOC_REQBUFS, &request) < 0 || request.count < 2) {
    return fail(fd, buffers, streaming, "VIDIOC_REQBUFS");
  }
  buffers.resize(request.count);
  for (std::uint32_t index = 0; index < request.count; ++index) {
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    if (ioctl_retry(fd, VIDIOC_QUERYBUF, &buffer) < 0) {
      return fail(fd, buffers, streaming, "VIDIOC_QUERYBUF");
    }
    buffers[index].length = buffer.length;
    buffers[index].address =
        mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
             static_cast<off_t>(buffer.m.offset));
    if (buffers[index].address == MAP_FAILED) {
      return fail(fd, buffers, streaming, "mmap camera buffer");
    }
    if (ioctl_retry(fd, VIDIOC_QBUF, &buffer) < 0) {
      return fail(fd, buffers, streaming, "initial VIDIOC_QBUF");
    }
  }

  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  const auto stream_started_at = std::chrono::steady_clock::now();
  if (ioctl_retry(fd, VIDIOC_STREAMON, &type) < 0) {
    return fail(fd, buffers, streaming, "VIDIOC_STREAMON");
  }
  streaming = true;

  std::uint64_t frames = 0;
  std::uint64_t sequence_gaps = 0;
  std::uint64_t bytes = 0;
  std::uint32_t previous_sequence = 0;
  bool have_previous_sequence = false;
  double interval_mean_s = 0.0;
  double interval_m2_s2 = 0.0;
  double maximum_interval_s = 0.0;
  std::chrono::steady_clock::time_point first_received_at{};
  std::chrono::steady_clock::time_point previous_received_at{};
  std::chrono::steady_clock::time_point last_received_at{};

  while (true) {
    pollfd descriptor{};
    descriptor.fd = fd;
    descriptor.events = POLLIN;
    const int timeout_ms =
        frames == 0 ? config.startup_timeout_ms : config.frame_timeout_ms;
    const int poll_result = poll(&descriptor, 1, timeout_ms);
    if (poll_result < 0) {
      if (errno == EINTR) continue;
      return fail(fd, buffers, streaming, "poll camera");
    }
    if (poll_result == 0) {
      errno = ETIMEDOUT;
      return fail(fd, buffers, streaming, "poll camera");
    }

    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (ioctl_retry(fd, VIDIOC_DQBUF, &buffer) < 0) {
      if (errno == EAGAIN) continue;
      return fail(fd, buffers, streaming, "VIDIOC_DQBUF");
    }
    const auto received_at = std::chrono::steady_clock::now();
    if (frames == 0) {
      first_received_at = received_at;
    } else {
      const double interval_s =
          std::chrono::duration<double>(received_at - previous_received_at)
              .count();
      const double delta = interval_s - interval_mean_s;
      interval_mean_s += delta / static_cast<double>(frames);
      interval_m2_s2 += delta * (interval_s - interval_mean_s);
      maximum_interval_s = std::max(maximum_interval_s, interval_s);
    }
    if (have_previous_sequence && buffer.sequence > previous_sequence + 1U) {
      sequence_gaps += buffer.sequence - previous_sequence - 1U;
    }
    previous_sequence = buffer.sequence;
    have_previous_sequence = true;
    previous_received_at = received_at;
    last_received_at = received_at;
    ++frames;
    bytes += buffer.bytesused;

    if (ioctl_retry(fd, VIDIOC_QBUF, &buffer) < 0) {
      return fail(fd, buffers, streaming, "VIDIOC_QBUF");
    }
    if (std::chrono::duration<double>(received_at - first_received_at).count() >=
        config.duration_s) {
      break;
    }
  }

  release_buffers(fd, buffers, streaming);
  const double elapsed_s =
      std::chrono::duration<double>(last_received_at - first_received_at).count();
  const double delivered_fps =
      frames > 1 && elapsed_s > 0.0 ? (frames - 1) / elapsed_s : 0.0;
  const double startup_s =
      std::chrono::duration<double>(first_received_at - stream_started_at).count();
  const double interval_std_s = frames > 2
                                    ? std::sqrt(interval_m2_s2 / (frames - 2))
                                    : 0.0;
  const double negotiated_fps =
      parameters.parm.capture.timeperframe.numerator == 0
          ? 0.0
          : static_cast<double>(
                parameters.parm.capture.timeperframe.denominator) /
                parameters.parm.capture.timeperframe.numerator;

  std::cout << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"mode\": \"v4l2_transport_benchmark\",\n"
            << "  \"actuators_accessed\": false,\n"
            << "  \"device\": \"" << config.device << "\",\n"
            << "  \"format\": \"" << fourcc_string(format.fmt.pix.pixelformat)
            << "\",\n"
            << "  \"width\": " << format.fmt.pix.width << ",\n"
            << "  \"height\": " << format.fmt.pix.height << ",\n"
            << "  \"negotiated_fps\": " << negotiated_fps << ",\n"
            << "  \"buffer_count\": " << request.count << ",\n"
            << "  \"duration_s\": " << elapsed_s << ",\n"
            << "  \"startup_s\": " << startup_s << ",\n"
            << "  \"frames\": " << frames << ",\n"
            << "  \"sequence_gaps\": " << sequence_gaps << ",\n"
            << "  \"delivered_fps\": " << delivered_fps << ",\n"
            << "  \"mean_interval_ms\": " << interval_mean_s * 1000.0 << ",\n"
            << "  \"interval_std_ms\": " << interval_std_s * 1000.0 << ",\n"
            << "  \"maximum_interval_ms\": " << maximum_interval_s * 1000.0
            << ",\n"
            << "  \"transport_mib_per_s\": "
            << (elapsed_s > 0.0 ? bytes / elapsed_s / (1024.0 * 1024.0) : 0.0)
            << "\n}\n";
  return EXIT_SUCCESS;
}
