#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "jetracer_sim/native_defaults.hpp"

namespace jetracer::sim {

constexpr int kSceneSchemaVersion = defaults::kNativeSchemaVersion;

enum class LensModel { BrownConrady, FisheyeEquidistant };
enum class PixelFormat { Nv12VideoRange };
enum class ShutterType { Global, Rolling };
enum class SemanticClass : std::uint8_t {
  Background = 0,
  DrivableSurface = 1,
  LaneMarking = 2,
  StopSign = 3,
  Obstacle = 4,
};
enum class ObjectType { Box, StopSign };

struct Point2 {
  double x{0.0};
  double y{0.0};
};

struct Pose2D {
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

struct VehicleCommand {
  double target_speed_mps{0.0};
  double steering_rad{0.0};
};

struct VehicleConfig {
  // Pose2D is the rear-axle midpoint. Dimensions define the full footprint.
  double wheelbase_m{defaults::kVehicleWheelbaseM};
  double body_width_m{defaults::kVehicleBodyWidthM};
  double front_overhang_m{defaults::kVehicleFrontOverhangM};
  double rear_overhang_m{defaults::kVehicleRearOverhangM};
  double max_steering_rad{defaults::kVehicleMaxSteeringRad};
  double steering_time_constant_s{defaults::kVehicleSteeringTimeConstantS};
  double motor_time_constant_s{defaults::kVehicleMotorTimeConstantS};

  [[nodiscard]] double body_length_m() const;
  [[nodiscard]] double minimum_turn_radius_m() const;
};

struct VehicleState {
  Pose2D pose{};
  double speed_mps{0.0};
  double steering_rad{0.0};
};

struct CameraProfile {
  std::string id{defaults::kCameraProfilesStressId};
  int width{defaults::kCameraProfilesStressWidth};
  int height{defaults::kCameraProfilesStressHeight};
  std::int64_t fps_numerator{defaults::kCameraProfilesStressFpsNumerator};
  std::int64_t fps_denominator{defaults::kCameraProfilesStressFpsDenominator};
  PixelFormat pixel_format{PixelFormat::Nv12VideoRange};
  LensModel lens_model{
      defaults::kCameraProfilesStressLensModel == "fisheye_equidistant"
          ? LensModel::FisheyeEquidistant
          : LensModel::BrownConrady};
  ShutterType shutter{defaults::kCameraProfilesStressShutter == "global"
                          ? ShutterType::Global
                          : ShutterType::Rolling};
  double nominal_hfov_rad{
      defaults::kCameraProfilesStressNominalHfovDegrees *
      std::numbers::pi_v<double> / 180.0};
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
  std::array<double, 5> distortion{defaults::kCameraProfilesStressDistortion};
  double mount_x_m{defaults::kCameraProfilesStressMountXM};
  double mount_y_m{defaults::kCameraProfilesStressMountYM};
  double mount_z_m{defaults::kCameraProfilesStressMountZM};
  double mount_roll_rad{defaults::kCameraProfilesStressMountRollRad};
  double mount_pitch_down_rad{defaults::kCameraProfilesStressMountPitchDownRad};
  double mount_yaw_rad{defaults::kCameraProfilesStressMountYawRad};
  double exposure_s{defaults::kCameraProfilesStressExposureS};
  double rolling_readout_s{defaults::kCameraProfilesStressRollingReadoutS};
  bool provisional{defaults::kCameraProfilesStressProvisional};

  [[nodiscard]] double fps() const;
  [[nodiscard]] double frame_period_s() const;
  void apply_nominal_intrinsics();
  void apply_opencv_calibration(const std::string& path);
  void validate() const;

  static CameraProfile elp_112();
  static CameraProfile stress_720p_200();
  static CameraProfile imx219_160_provisional();
};

struct SceneObject {
  std::uint32_t instance_id{0};
  ObjectType type{ObjectType::Box};
  SemanticClass semantic_class{SemanticClass::Obstacle};
  Point2 position{};
  double base_z_m{0.0};
  double yaw_rad{0.0};
  double width_m{defaults::kDefaultObjectWidthM};
  double depth_m{defaults::kDefaultObjectDepthM};
  double height_m{defaults::kDefaultObjectHeightM};
  std::array<std::uint8_t, 3> bgr{
      static_cast<std::uint8_t>(defaults::kDefaultObjectBgr[0]),
      static_cast<std::uint8_t>(defaults::kDefaultObjectBgr[1]),
      static_cast<std::uint8_t>(defaults::kDefaultObjectBgr[2])};
};

struct SceneConfig {
  std::uint64_t seed{defaults::kSceneGenerationSeed};
  int control_points{defaults::kSceneGenerationControlPoints};
  int samples_per_segment{defaults::kSceneGenerationSamplesPerSegment};
  double base_radius_m{defaults::kSceneGenerationBaseRadiusM};
  double radius_jitter_m{defaults::kSceneGenerationRadiusJitterM};
  double road_width_m{defaults::kSceneGenerationRoadWidthM};
  double atlas_pixels_per_metre{
      defaults::kSceneGenerationAtlasPixelsPerMetre};
  int obstacle_count{defaults::kSceneGenerationObstacleCount};
  int stop_sign_count{defaults::kSceneGenerationStopSignCount};
  std::string background_texture_path{};
  std::string road_texture_path{};
};

struct Scene {
  int schema_version{kSceneSchemaVersion};
  std::uint64_t seed{defaults::kSceneGenerationSeed};
  double road_width_m{defaults::kSceneGenerationRoadWidthM};
  double atlas_pixels_per_metre{
      defaults::kSceneGenerationAtlasPixelsPerMetre};
  VehicleConfig vehicle{};
  VehicleState start{};
  CameraProfile camera{CameraProfile::stress_720p_200()};
  std::string background_texture_path{};
  std::string road_texture_path{};
  std::vector<Point2> centerline{};
  std::vector<SceneObject> objects{};

  static Scene generate(const SceneConfig& config);
  static Scene load(const std::string& path);
  void save(const std::string& path) const;
  void validate() const;
};

struct Detection {
  int class_id{0};
  std::uint32_t instance_id{0};
  // Half-open pixel coordinates: [x_min, y_min, x_max, y_max).
  std::array<int, 4> bbox_xyxy{};
  double visibility{0.0};
  double range_m{0.0};
  double relative_yaw_rad{0.0};
};

struct Frame {
  std::uint64_t frame_id{0};
  double simulation_time_s{0.0};
  double exposure_start_s{0.0};
  double exposure_end_s{0.0};
  VehicleState vehicle{};
  CameraProfile camera{};
  cv::Mat y_plane{};       // H x W, uint8
  cv::Mat uv_plane{};      // H/2 x W, interleaved U/V uint8
  cv::Mat semantic{};      // H x W, uint8 SemanticClass
  cv::Mat instance{};      // H x W, CV_32SC1 (non-negative instance ID)
  std::vector<Detection> detections{};
};

using FramePtr = std::shared_ptr<Frame>;
using FrameBatch = std::vector<FramePtr>;

void step_bicycle(const VehicleConfig& config, VehicleState& state,
                  const VehicleCommand& command, double dt_s);
cv::Mat frame_to_bgr(const Frame& frame);

class Simulator {
 public:
  Simulator(Scene scene, CameraProfile camera);
  ~Simulator();
  Simulator(Simulator&&) noexcept;
  Simulator& operator=(Simulator&&) noexcept;
  Simulator(const Simulator&) = delete;
  Simulator& operator=(const Simulator&) = delete;

  void reset();
  void reset(Scene scene, CameraProfile camera);
  void set_vehicle_state(VehicleState state);
  [[nodiscard]] FrameBatch advance(const VehicleCommand& command,
                                   double dt_s =
                                       static_cast<double>(
                                           defaults::kCameraProfilesStressFpsDenominator) /
                                       defaults::kCameraProfilesStressFpsNumerator);
  [[nodiscard]] FramePtr render_now();
  [[nodiscard]] const Scene& scene() const;
  [[nodiscard]] const CameraProfile& camera() const;
  [[nodiscard]] const VehicleState& vehicle_state() const;
  [[nodiscard]] double simulation_time_s() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jetracer::sim
