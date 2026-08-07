#pragma once

#include "color_lane_native.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct NativeShadowConfig {
  std::string profile_id;
  std::string actuator_mode;
  std::string calibration_id;
  bool calibration_validated = false;
  std::string lens_model;
  int source_width = 0;
  int source_height = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  std::array<double, 5> distortion{};
  double mount_x_m = 0.0;
  double mount_y_m = 0.0;
  double mount_z_m = 0.0;
  double mount_roll_rad = 0.0;
  double mount_pitch_down_rad = 0.0;
  double mount_yaw_rad = 0.0;
  double wheelbase_m = 0.0;
  double maximum_steering_rad = 0.0;
  double base_lookahead_m = 0.0;
  double speed_lookahead_s = 0.0;
  double minimum_lookahead_m = 0.0;
  double maximum_lookahead_m = 0.0;
  double pure_pursuit_gain = 0.0;
  double lateral_error_gain = 0.0;
  double lateral_speed_softening_mps = 0.0;
  double lost_steering_hold_s = 0.0;
  double steering_smoothing_time_s = 0.0;
  double maximum_steering_rate_rad_s = 0.0;
  double curvature_estimation_distance_m = 0.0;
  int minimum_curvature_points = 0;
  double curvature_lookahead_gain_m2 = 0.0;
  double lateral_error_lookahead_gain = 0.0;
  double cruise_speed_mps = 0.0;
  double maximum_speed_mps = 0.0;
  double minimum_tracking_confidence = 0.0;
  double full_speed_confidence = 0.0;
  double maximum_acceleration_mps2 = 0.0;
  double maximum_deceleration_mps2 = 0.0;
};

NativeShadowConfig load_native_shadow_config(const std::string& path);

struct NativeShadowCommand {
  std::uint64_t sequence = 0;
  double timestamp_s = 0.0;
  double requested_speed_mps = 0.0;
  double estimated_speed_mps = 0.0;
  double steering_rad = 0.0;
  double raw_steering_rad = 0.0;
  double confidence = 0.0;
  double requested_lookahead_m = 0.0;
  double actual_lookahead_m = 0.0;
  double target_forward_m = 0.0;
  double target_lateral_m = 0.0;
  int projected_points = 0;
  std::string reason;
  bool actuator_write_attempted = false;
};

class DisabledActuatorSink {
 public:
  void record(const NativeShadowCommand& command);
  std::uint64_t command_count() const { return command_count_; }
  bool hardware_accessed() const { return false; }

 private:
  std::uint64_t command_count_ = 0;
};

class NativeShadowController {
 public:
  NativeShadowController(NativeShadowConfig config, int processing_width,
                         int processing_height);
  NativeShadowCommand update(const ColorLaneNativeResult& lane, double dt_s,
                             double timestamp_s);
  void reset();

 private:
  struct PathPoint {
    cv::Point2d pixel;
    cv::Point2d vehicle;
    double distance_m = 0.0;
  };

  std::vector<PathPoint> project_path(
      const std::vector<cv::Point2f>& normalized) const;
  bool project_ground(double normalized_x, double normalized_y,
                      PathPoint* output) const;
  double filtered_steering(double raw, double dt_s);

  NativeShadowConfig config_;
  int processing_width_ = 0;
  int processing_height_ = 0;
  double steering_rad_ = 0.0;
  double estimated_speed_mps_ = 0.0;
  double lost_time_s_ = 0.0;
  std::uint64_t sequence_ = 0;
  DisabledActuatorSink actuator_;
};

std::string native_shadow_command_json(const NativeShadowCommand& command);
