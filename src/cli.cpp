#include "jetracer_sim/simulator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifdef __APPLE__
#include <ApplicationServices/ApplicationServices.h>
#endif

using namespace jetracer::sim;

namespace {

CameraProfile profile_named(const std::string& name) {
  if (name == "elp") return CameraProfile::elp_112();
  if (name == "stress") return CameraProfile::stress_720p_200();
  if (name == "imx219") return CameraProfile::imx219_160_provisional();
  throw std::invalid_argument("unknown profile '" + name +
                              "' (expected elp, stress, or imx219)");
}

double percentile(std::vector<double> values, double quantile) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::round((values.size() - 1) * quantile));
  return values[index];
}

double move_towards(double value, double target, double maximum_change) {
  if (value < target) return std::min(target, value + maximum_change);
  return std::max(target, value - maximum_change);
}

#ifdef __APPLE__
bool mac_key_down(CGKeyCode key_code) {
  return CGEventSourceKeyState(kCGEventSourceStateCombinedSessionState,
                               key_code);
}
#endif

std::uint64_t sparse_hash(const cv::Mat& image) {
  std::uint64_t value = 1469598103934665603ULL;
  const int step_x =
      std::max(1, image.cols / defaults::kCliSparseHashColumns);
  const int step_y =
      std::max(1, image.rows / defaults::kCliSparseHashRows);
  for (int y = 0; y < image.rows; y += step_y) {
    const auto* row = image.ptr<std::uint8_t>(y);
    for (int x = 0; x < image.cols; x += step_x) {
      value ^= row[x];
      value *= 1099511628211ULL;
    }
  }
  return value;
}

void write_yolo(const std::filesystem::path& path, const Frame& frame) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write " + path.string());
  output << std::fixed << std::setprecision(8);
  for (const Detection& detection : frame.detections) {
    const double x0 = detection.bbox_xyxy[0];
    const double y0 = detection.bbox_xyxy[1];
    const double x1 = detection.bbox_xyxy[2];
    const double y1 = detection.bbox_xyxy[3];
    output << detection.class_id << ' '
           << ((x0 + x1) * 0.5 / frame.camera.width) << ' '
           << ((y0 + y1) * 0.5 / frame.camera.height) << ' '
           << ((x1 - x0) / frame.camera.width) << ' '
           << ((y1 - y0) / frame.camera.height) << '\n';
  }
}

void usage() {
  std::cerr
      << "Usage:\n"
      << "  jetracer-sim generate <scene.json> [seed]\n"
      << "  jetracer-sim render <scene.json> <output-prefix> [profile]\n"
      << "  jetracer-sim benchmark [profile] [frames] [--paced]\n"
      << "  jetracer-sim drive [profile] [scene.json] --allow-native-gui\n"
      << "Profiles: stress, elp, imx219 (default: "
      << defaults::kCliDefaultProfile << ")\n";
}

void draw_drive_overlay(cv::Mat& image, const Frame& frame,
                        const VehicleCommand& command, bool show_labels,
                        bool paused) {
  if (show_labels) {
    cv::Mat labels(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    labels.setTo(cv::Scalar(70, 120, 70),
                 frame.semantic ==
                     static_cast<int>(SemanticClass::DrivableSurface));
    labels.setTo(cv::Scalar(40, 210, 240),
                 frame.semantic == static_cast<int>(SemanticClass::LaneMarking));
    labels.setTo(cv::Scalar(20, 20, 240),
                 frame.semantic == static_cast<int>(SemanticClass::StopSign));
    labels.setTo(cv::Scalar(220, 80, 180),
                 frame.semantic == static_cast<int>(SemanticClass::Obstacle));
    cv::addWeighted(image, 0.72, labels, 0.28, 0.0, image);
  }
  for (const Detection& detection : frame.detections) {
    const cv::Scalar colour =
        detection.class_id == static_cast<int>(SemanticClass::StopSign)
            ? cv::Scalar(30, 30, 240)
            : cv::Scalar(220, 120, 40);
    cv::rectangle(image,
                  cv::Rect(cv::Point(detection.bbox_xyxy[0],
                                     detection.bbox_xyxy[1]),
                           cv::Point(detection.bbox_xyxy[2],
                                     detection.bbox_xyxy[3])),
                  colour, 2, cv::LINE_AA);
  }

  cv::rectangle(image, cv::Rect(0, 0, image.cols, 78), cv::Scalar(20, 20, 20),
                cv::FILLED);
  std::ostringstream status;
  status << std::fixed << std::setprecision(2) << "speed "
         << frame.vehicle.speed_mps << " / " << command.target_speed_mps
         << " m/s    steering " << command.steering_rad << " rad    sim "
         << frame.simulation_time_s << " s";
  cv::putText(image, status.str(), cv::Point(14, 28), cv::FONT_HERSHEY_SIMPLEX,
              0.65, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
  const std::string help =
      "Hold W/S speed  A/D steer  C centre  SPACE stop  P pause  L labels  R reset  Q quit";
  cv::putText(image, help, cv::Point(14, 58), cv::FONT_HERSHEY_SIMPLEX, 0.50,
              cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
  if (paused) {
    cv::putText(image, "PAUSED", cv::Point(image.cols - 125, 29),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(40, 210, 255), 2,
                cv::LINE_AA);
  }
}

int drive(const CameraProfile& camera, Scene scene) {
  constexpr std::string_view window_name = "JetRacer simulated camera";
  Simulator simulator(std::move(scene), camera);
  VehicleCommand vehicle_command{};
  bool paused = false;
  bool show_labels = false;
  cv::namedWindow(std::string(window_name), cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
  const double aspect = static_cast<double>(camera.width) / camera.height;
  const int window_width =
      std::min(camera.width, defaults::kCliMaximumWindowWidthPixels);
  cv::resizeWindow(std::string(window_name), window_width,
                   static_cast<int>(window_width / aspect));

  FramePtr frame = simulator.render_now();
  auto next_frame_time = std::chrono::steady_clock::now();
#ifdef __APPLE__
  bool previous_pause = false;
  bool previous_labels = false;
  bool previous_reset = false;
#endif
  while (cv::getWindowProperty(std::string(window_name), cv::WND_PROP_VISIBLE) >=
         1.0) {
    if (!paused) {
      FrameBatch frames =
          simulator.advance(vehicle_command, camera.frame_period_s());
      if (!frames.empty()) frame = frames.back();
    }
    cv::Mat display = frame_to_bgr(*frame);
    draw_drive_overlay(display, *frame, vehicle_command, show_labels, paused);
    cv::imshow(std::string(window_name), display);

    const int key = cv::waitKeyEx(1);
#ifdef __APPLE__
    // macOS virtual key codes. Polling gives true press/release state, unlike
    // HighGUI's character-repeat-only API.
    constexpr CGKeyCode key_a = 0;
    constexpr CGKeyCode key_s = 1;
    constexpr CGKeyCode key_d = 2;
    constexpr CGKeyCode key_c = 8;
    constexpr CGKeyCode key_q = 12;
    constexpr CGKeyCode key_w = 13;
    constexpr CGKeyCode key_r = 15;
    constexpr CGKeyCode key_p = 35;
    constexpr CGKeyCode key_l = 37;
    constexpr CGKeyCode key_space = 49;
    constexpr CGKeyCode key_escape = 53;
    constexpr CGKeyCode key_left = 123;
    constexpr CGKeyCode key_right = 124;
    constexpr CGKeyCode key_down = 125;
    constexpr CGKeyCode key_up = 126;

    const bool accelerate = mac_key_down(key_w) || mac_key_down(key_up);
    const bool brake = mac_key_down(key_s) || mac_key_down(key_down);
    const bool steer_left = mac_key_down(key_a) || mac_key_down(key_left);
    const bool steer_right = mac_key_down(key_d) || mac_key_down(key_right);
    const bool pause_down = mac_key_down(key_p);
    const bool labels_down = mac_key_down(key_l);
    const bool reset_down = mac_key_down(key_r);
    const double control_dt = camera.frame_period_s();

    if (accelerate != brake) {
      const double rate =
          accelerate ? defaults::kCliAccelerationRateMps2
                     : -defaults::kCliBrakingRateMps2;
      vehicle_command.target_speed_mps = std::clamp(
          vehicle_command.target_speed_mps + rate * control_dt, 0.0,
          defaults::kCliMaximumSpeedMps);
    }
    if (steer_left != steer_right) {
      const double direction = steer_left ? 1.0 : -1.0;
      vehicle_command.steering_rad = std::clamp(
          vehicle_command.steering_rad +
              direction * defaults::kCliSteeringRateRadS * control_dt,
          -simulator.scene().vehicle.max_steering_rad,
          simulator.scene().vehicle.max_steering_rad);
    } else {
      vehicle_command.steering_rad =
          move_towards(vehicle_command.steering_rad, 0.0,
                       defaults::kCliSteeringRecenterRateRadS * control_dt);
    }
    if (mac_key_down(key_c)) vehicle_command.steering_rad = 0.0;
    if (mac_key_down(key_space)) vehicle_command.target_speed_mps = 0.0;
    if (mac_key_down(key_q) || mac_key_down(key_escape) || key == 'q' ||
        key == 'Q' || key == 27) {
      break;
    }
    if (pause_down && !previous_pause) paused = !paused;
    if (labels_down && !previous_labels) show_labels = !show_labels;
    if (reset_down && !previous_reset) {
      simulator.reset();
      vehicle_command = {};
      frame = simulator.render_now();
      next_frame_time = std::chrono::steady_clock::now();
    }
    previous_pause = pause_down;
    previous_labels = labels_down;
    previous_reset = reset_down;
#else
    if (key == 'q' || key == 'Q' || key == 27) break;
    if (key == 'w' || key == 'W' || key == 63232 || key == 2490368) {
      vehicle_command.target_speed_mps =
          std::min(defaults::kCliMaximumSpeedMps,
                   vehicle_command.target_speed_mps +
                       defaults::kCliKeySpeedStepMps);
    } else if (key == 's' || key == 'S' || key == 63233 || key == 2621440) {
      vehicle_command.target_speed_mps =
          std::max(0.0, vehicle_command.target_speed_mps -
                            defaults::kCliKeySpeedStepMps);
    } else if (key == 'a' || key == 'A' || key == 63234 || key == 2424832) {
      vehicle_command.steering_rad =
          std::min(simulator.scene().vehicle.max_steering_rad,
                   vehicle_command.steering_rad +
                       defaults::kCliKeySteeringStepRad);
    } else if (key == 'd' || key == 'D' || key == 63235 || key == 2555904) {
      vehicle_command.steering_rad =
          std::max(-simulator.scene().vehicle.max_steering_rad,
                   vehicle_command.steering_rad -
                       defaults::kCliKeySteeringStepRad);
    } else if (key == 'c' || key == 'C') {
      vehicle_command.steering_rad = 0.0;
    } else if (key == ' ') {
      vehicle_command.target_speed_mps = 0.0;
    } else if (key == 'p' || key == 'P') {
      paused = !paused;
    } else if (key == 'l' || key == 'L') {
      show_labels = !show_labels;
    } else if (key == 'r' || key == 'R') {
      simulator.reset();
      vehicle_command = {};
      frame = simulator.render_now();
      next_frame_time = std::chrono::steady_clock::now();
    }
#endif

    next_frame_time +=
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(camera.frame_period_s()));
    const auto now = std::chrono::steady_clock::now();
    if (next_frame_time > now) {
      std::this_thread::sleep_until(next_frame_time);
    } else if (now - next_frame_time >
               std::chrono::milliseconds(defaults::kCliScheduleLagResetMs)) {
      next_frame_time = now;
    }
  }
  cv::destroyWindow(std::string(window_name));
  return 0;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string command = argv[1];
  if (command == "generate") {
    if (argc < 3) {
      usage();
      return 2;
    }
    SceneConfig config;
    if (argc >= 4) config.seed = std::stoull(argv[3]);
    const Scene scene = Scene::generate(config);
    scene.save(argv[2]);
    std::cout << "wrote " << argv[2] << " with " << scene.centerline.size()
              << " centreline samples and " << scene.objects.size()
              << " objects\n";
    return 0;
  }
  if (command == "render") {
    if (argc < 4) {
      usage();
      return 2;
    }
    Scene scene = Scene::load(argv[2]);
    const CameraProfile camera = profile_named(
        argc >= 5 ? argv[4] : std::string(defaults::kCliDefaultProfile));
    Simulator simulator(std::move(scene), camera);
    const FramePtr frame = simulator.render_now();
    const std::filesystem::path prefix(argv[3]);
    cv::imwrite(prefix.string() + "-rgb.png", frame_to_bgr(*frame));
    cv::imwrite(prefix.string() + "-semantic.png", frame->semantic);
    cv::Mat instance_16;
    frame->instance.convertTo(instance_16, CV_16UC1);
    cv::imwrite(prefix.string() + "-instance.png", instance_16);
    write_yolo(prefix.string() + ".txt", *frame);
    std::cout << "rendered frame " << frame->frame_id << " using "
              << camera.id << "\n";
    return 0;
  }
  if (command == "benchmark") {
    const CameraProfile camera = profile_named(
        argc >= 3 ? argv[2] : std::string(defaults::kCliDefaultProfile));
    const int requested_frames =
        argc >= 4 ? std::stoi(argv[3]) : defaults::kCliBenchmarkFrames;
    if (requested_frames <= 0) {
      throw std::invalid_argument("benchmark frame count must be positive");
    }
    const bool paced = argc >= 5 && std::string(argv[4]) == "--paced";
    SceneConfig config;
    Scene scene = Scene::generate(config);
    Simulator simulator(std::move(scene), camera);
    VehicleCommand vehicle_command{defaults::kCliBenchmarkSpeedMps,
                                   defaults::kCliBenchmarkSteeringRad};
    constexpr int warmup_frames = defaults::kCliBenchmarkWarmupFrames;
    double cold_render_ms = 0.0;
    for (int index = 0; index < warmup_frames; ++index) {
      const auto start = std::chrono::steady_clock::now();
      const FrameBatch batch =
          simulator.advance(vehicle_command, camera.frame_period_s());
      const auto end = std::chrono::steady_clock::now();
      if (index == 0 && !batch.empty()) {
        cold_render_ms =
            std::chrono::duration<double, std::milli>(end - start).count() /
            batch.size();
      }
    }
    simulator.reset();
    std::vector<double> render_times_ms;
    render_times_ms.reserve(requested_frames);
    std::unordered_set<std::uint64_t> hashes;
    int frames = 0;
    int missed_deadlines = 0;
    double max_deadline_lateness_ms = 0.0;
    const auto benchmark_start = std::chrono::steady_clock::now();
    auto next_deadline = benchmark_start;
    while (frames < requested_frames) {
      const auto start = std::chrono::steady_clock::now();
      FrameBatch batch = simulator.advance(vehicle_command, camera.frame_period_s());
      const auto end = std::chrono::steady_clock::now();
      if (!batch.empty()) {
        render_times_ms.push_back(
            std::chrono::duration<double, std::milli>(end - start).count() /
            batch.size());
        for (const FramePtr& frame : batch) hashes.insert(sparse_hash(frame->y_plane));
        frames += static_cast<int>(batch.size());
      }
      if (paced) {
        next_deadline += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(camera.frame_period_s()));
        if (end > next_deadline) {
          ++missed_deadlines;
          max_deadline_lateness_ms = std::max(
              max_deadline_lateness_ms,
              std::chrono::duration<double, std::milli>(end - next_deadline)
                  .count());
        } else {
          std::this_thread::sleep_until(next_deadline);
        }
      }
    }
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - benchmark_start)
                               .count();
    std::cout << std::fixed << std::setprecision(3)
              << "profile=" << camera.id << '\n'
              << "warmup_frames=" << warmup_frames << '\n'
              << "cold_render_ms=" << cold_render_ms << '\n'
              << "frames=" << frames << '\n'
              << "unique_sparse_hashes=" << hashes.size() << '\n'
              << "missed_deadlines=" << missed_deadlines << '\n'
              << "max_deadline_lateness_ms=" << max_deadline_lateness_ms << '\n'
              << "throughput_fps=" << frames / elapsed << '\n'
              << "render_ms_p50=" << percentile(render_times_ms, 0.50) << '\n'
              << "render_ms_p95=" << percentile(render_times_ms, 0.95) << '\n'
              << "render_ms_p99=" << percentile(render_times_ms, 0.99) << '\n'
              << "render_ms_max="
              << *std::max_element(render_times_ms.begin(),
                                   render_times_ms.end())
              << '\n';
    return 0;
  }
  if (command == "drive") {
    bool allow_native_gui = false;
    std::vector<std::string> positional;
    for (int index = 2; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--allow-native-gui") {
        allow_native_gui = true;
      } else {
        positional.push_back(argument);
      }
    }
    if (!allow_native_gui) {
      throw std::invalid_argument(
          "native OpenCV windows are disabled by default because macOS may "
          "abort while locked; pass --allow-native-gui only in an unlocked "
          "interactive session");
    }
    if (positional.size() > 2) {
      throw std::invalid_argument("drive accepts at most a profile and scene");
    }
    const CameraProfile camera = profile_named(
        positional.empty() ? std::string(defaults::kCliDefaultProfile)
                           : positional[0]);
    Scene scene;
    if (positional.size() >= 2) {
      scene = Scene::load(positional[1]);
    } else {
      SceneConfig config;
      config.seed = defaults::kCliDriveSceneSeed;
      scene = Scene::generate(config);
    }
    return drive(camera, std::move(scene));
  }
  usage();
  return 2;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
