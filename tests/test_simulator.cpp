#include "jetracer_sim/simulator.hpp"
#include "jetracer_sim/filesystem_compat.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace jetracer::sim;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

void test_profiles() {
  const CameraProfile elp = CameraProfile::elp_112();
  check(elp.width == 1920 && elp.height == 1200, "ELP dimensions");
  check(std::abs(elp.fps() - 120.00048) < 1e-5, "ELP measured cadence");
  const CameraProfile stress = CameraProfile::stress_720p_200();
  check(stress.width == 1280 && stress.height == 720, "stress dimensions");
  check(std::abs(stress.fps() - 200.0) < 1e-9, "stress cadence");
}

void test_bicycle() {
  VehicleConfig config;
  check(std::abs(config.body_length_m() - 0.2566875) < 1e-12,
        "JetRacer body length");
  check(std::abs(config.body_width_m - 0.14) < 1e-12,
        "JetRacer body width");
  check(config.minimum_turn_radius_m() > config.wheelbase_m,
        "finite bicycle turning radius");
  config.motor_time_constant_s = 0.0;
  config.steering_time_constant_s = 0.0;
  VehicleState state;
  step_bicycle(config, state, VehicleCommand{1.0, 0.0}, 1.0);
  check(std::abs(state.pose.x - 1.0) < 1e-9, "straight bicycle distance");
  check(std::abs(state.pose.y) < 1e-9, "straight bicycle lateral position");
  step_bicycle(config, state, VehicleCommand{1.0, 0.2}, 0.5);
  check(state.pose.y > 0.0, "positive steering turns left");
  check(state.pose.yaw > 0.0, "positive steering changes yaw");
}

void test_scene_replay() {
  SceneConfig config;
  config.seed = 42;
  Scene first = Scene::generate(config);
  Scene second = Scene::generate(config);
  check(first.centerline.size() == second.centerline.size(),
        "deterministic centreline size");
  check(std::abs(first.centerline[17].x - second.centerline[17].x) < 1e-12,
        "deterministic centreline coordinates");
  check(first.objects.size() == second.objects.size(), "deterministic objects");

  const auto path = jetracer_filesystem::temp_directory_path() /
                    "jetracer-sim-scene-test.json";
  first.save(path.string());
  Scene loaded = Scene::load(path.string());
  check(loaded.seed == first.seed, "scene seed round trip");
  check(loaded.centerline.size() == first.centerline.size(),
        "scene centreline round trip");
  check(loaded.objects.size() == first.objects.size(),
        "scene object round trip");
  check(std::abs(loaded.vehicle.body_width_m - first.vehicle.body_width_m) < 1e-12,
        "vehicle footprint round trip");
  check(loaded.camera.mount_provisional == first.camera.mount_provisional,
        "camera mount status round trip");
  std::error_code ignored;
  jetracer_filesystem::remove(path, ignored);
}

void test_render_and_cadence() {
  SceneConfig config;
  config.seed = 7;
  config.obstacle_count = 2;
  config.stop_sign_count = 1;
  Scene scene = Scene::generate(config);
  CameraProfile camera = CameraProfile::stress_720p_200();
  camera.width = 320;
  camera.height = 180;
  camera.apply_nominal_intrinsics();
  Simulator simulator(scene, camera);
  const FramePtr frame = simulator.render_now();
  check(frame->y_plane.rows == 180 && frame->y_plane.cols == 320,
        "NV12 Y plane dimensions");
  check(frame->uv_plane.rows == 90 && frame->uv_plane.cols == 320,
        "NV12 UV plane dimensions");
  check(frame->semantic.type() == CV_8UC1, "semantic type");
  check(frame->instance.type() == CV_32SC1, "instance type");
  check(frame_to_bgr(*frame).type() == CV_8UC3, "NV12 round-trip type");

  int emitted = 0;
  double last_timestamp = 0.0;
  for (int index = 0; index < 200; ++index) {
    const FrameBatch batch =
        simulator.advance(VehicleCommand{0.5, 0.0}, 0.005);
    emitted += static_cast<int>(batch.size());
    if (!batch.empty()) {
      check(batch.front()->simulation_time_s > last_timestamp,
            "camera timestamps are monotonic");
      last_timestamp = batch.front()->simulation_time_s;
    }
  }
  check(emitted == 200, "200 Hz camera cadence over one second");
  check(std::abs(last_timestamp - 1.0) < 1e-12,
        "rational camera schedule ends at one second");

  VehicleState relocated;
  relocated.pose = {1.0, -2.0, 0.25};
  relocated.speed_mps = 0.0;
  relocated.steering_rad = 4.0;
  simulator.set_vehicle_state(relocated);
  check(std::abs(simulator.vehicle_state().pose.x - 1.0) < 1e-12,
        "vehicle relocation updates pose");
  check(std::abs(simulator.vehicle_state().steering_rad -
                 scene.vehicle.max_steering_rad) < 1e-12,
        "vehicle relocation clamps steering");

  Simulator batched_simulator(scene, camera);
  const FrameBatch batched =
      batched_simulator.advance(VehicleCommand{0.5, 0.0}, 0.012);
  check(batched.size() == 2, "camera emits multiple due frames in one update");
  check(std::abs(batched[0]->simulation_time_s - 0.005) < 1e-12,
        "first batched timestamp");
  check(std::abs(batched[1]->simulation_time_s - 0.010) < 1e-12,
        "second batched timestamp");
}

void test_cylinder_rendering() {
  SceneConfig config;
  config.seed = 17;
  config.obstacle_count = 0;
  config.stop_sign_count = 0;
  Scene scene = Scene::generate(config);
  SceneObject cylinder;
  cylinder.instance_id = 1;
  cylinder.type = ObjectType::Cylinder;
  cylinder.semantic_class = SemanticClass::Obstacle;
  cylinder.position = scene.centerline.at(12);
  cylinder.width_m = 0.06;
  cylinder.depth_m = 0.06;
  cylinder.height_m = 0.20;
  cylinder.radial_segments = 24;
  cylinder.bgr = {220, 30, 220};
  scene.objects = {cylinder};
  scene.validate();

  CameraProfile camera = CameraProfile::stress_720p_200();
  camera.width = 320;
  camera.height = 180;
  camera.apply_nominal_intrinsics();
  Simulator simulator(scene, camera);
  const FramePtr frame = simulator.render_now();
  check(cv::countNonZero(frame->instance == 1) > 0,
        "cylinder produces instance pixels");
  check(cv::countNonZero(frame->semantic ==
                         static_cast<int>(SemanticClass::Obstacle)) > 0,
        "cylinder produces obstacle semantics");
}

}  // namespace

int main() {
  try {
    test_profiles();
    test_bicycle();
    test_scene_replay();
    test_render_and_cadence();
    test_cylinder_rendering();
  } catch (const std::exception& error) {
    std::cerr << "unexpected exception: " << error.what() << '\n';
    return 2;
  }
  if (failures == 0) {
    std::cout << "all simulator tests passed\n";
  }
  return failures == 0 ? 0 : 1;
}
