#include "native_shadow_controller.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

double number(const cv::FileNode& parent, const char* name) {
  const cv::FileNode value = parent[name];
  if (value.empty() || (!value.isReal() && !value.isInt())) {
    throw std::runtime_error(std::string("missing numeric shadow setting: ") + name);
  }
  return static_cast<double>(value);
}

int integer(const cv::FileNode& parent, const char* name) {
  const cv::FileNode value = parent[name];
  if (value.empty() || !value.isInt()) {
    throw std::runtime_error(std::string("missing integer shadow setting: ") + name);
  }
  return static_cast<int>(value);
}

std::string text_value(const cv::FileNode& parent, const char* name) {
  std::string value;
  parent[name] >> value;
  if (value.empty()) {
    throw std::runtime_error(std::string("missing text shadow setting: ") + name);
  }
  return value;
}

cv::Matx33d rotation_x(double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  return {1, 0, 0, 0, c, -s, 0, s, c};
}
cv::Matx33d rotation_y(double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  return {c, 0, s, 0, 1, 0, -s, 0, c};
}
cv::Matx33d rotation_z(double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  return {c, -s, 0, s, c, 0, 0, 0, 1};
}

void validate(const NativeShadowConfig& c) {
  if (c.profile_id.empty() || c.actuator_mode != "disabled" ||
      c.calibration_id.empty() || !c.calibration_validated) {
    throw std::runtime_error(
        "shadow controller requires validated calibration and disabled actuators");
  }
  if (c.lens_model != "fisheye_equidistant" &&
      c.lens_model != "brown_conrady") {
    throw std::runtime_error("unsupported shadow camera lens model");
  }
  if (c.source_width <= 0 || c.source_height <= 0 || c.fx <= 0.0 ||
      c.fy <= 0.0 || c.mount_z_m <= 0.0 || c.wheelbase_m <= 0.0 ||
      c.maximum_steering_rad <= 0.0 || c.minimum_lookahead_m <= 0.0 ||
      c.maximum_lookahead_m < c.minimum_lookahead_m ||
      c.maximum_steering_rate_rad_s <= 0.0 ||
      c.minimum_curvature_points < 3 || c.cruise_speed_mps < 0.0 ||
      c.maximum_speed_mps < c.cruise_speed_mps ||
      c.minimum_tracking_confidence < 0.0 ||
      c.full_speed_confidence <= c.minimum_tracking_confidence ||
      c.full_speed_confidence > 1.0 || c.maximum_acceleration_mps2 <= 0.0 ||
      c.maximum_deceleration_mps2 <= 0.0) {
    throw std::runtime_error("invalid native shadow controller configuration");
  }
}

double clamp(double value, double low, double high) {
  return std::max(low, std::min(value, high));
}

}  // namespace

NativeShadowConfig load_native_shadow_config(const std::string& path) {
  cv::FileStorage storage(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  if (!storage.isOpened()) throw std::runtime_error("cannot open shadow config: " + path);
  if (integer(storage.root(), "schema_version") != 1) {
    throw std::runtime_error("unsupported shadow config schema");
  }
  NativeShadowConfig c;
  c.profile_id = text_value(storage.root(), "profile_id");
  c.actuator_mode = text_value(storage.root(), "actuator_mode");
  const cv::FileNode camera = storage["camera"];
  c.calibration_id = text_value(camera, "calibration_id");
  c.calibration_validated = integer(camera, "calibration_validated") != 0;
  c.lens_model = text_value(camera, "lens_model");
  c.source_width = integer(camera, "source_width");
  c.source_height = integer(camera, "source_height");
  c.fx = number(camera, "fx"); c.fy = number(camera, "fy");
  c.cx = number(camera, "cx"); c.cy = number(camera, "cy");
  const cv::FileNode distortion = camera["distortion"];
  if (!distortion.isSeq() || distortion.size() != 5) {
    throw std::runtime_error("shadow distortion must contain five values");
  }
  for (int i = 0; i < 5; ++i) c.distortion[i] = static_cast<double>(distortion[i]);
  c.mount_x_m = number(camera, "mount_x_m");
  c.mount_y_m = number(camera, "mount_y_m");
  c.mount_z_m = number(camera, "mount_z_m");
  c.mount_roll_rad = number(camera, "mount_roll_rad");
  c.mount_pitch_down_rad = number(camera, "mount_pitch_down_rad");
  c.mount_yaw_rad = number(camera, "mount_yaw_rad");
  const cv::FileNode vehicle = storage["vehicle"];
  c.wheelbase_m = number(vehicle, "wheelbase_m");
  c.maximum_steering_rad = number(vehicle, "maximum_steering_rad");
  const cv::FileNode controller = storage["controller"];
  if (text_value(controller, "kind") != "adaptive_pure_pursuit") {
    throw std::runtime_error("shadow controller kind must be adaptive_pure_pursuit");
  }
#define READ_CONTROLLER(field) c.field = number(controller, #field)
  READ_CONTROLLER(base_lookahead_m); READ_CONTROLLER(speed_lookahead_s);
  READ_CONTROLLER(minimum_lookahead_m); READ_CONTROLLER(maximum_lookahead_m);
  READ_CONTROLLER(pure_pursuit_gain); READ_CONTROLLER(lateral_error_gain);
  READ_CONTROLLER(lateral_speed_softening_mps); READ_CONTROLLER(lost_steering_hold_s);
  READ_CONTROLLER(steering_smoothing_time_s); READ_CONTROLLER(maximum_steering_rate_rad_s);
  READ_CONTROLLER(curvature_estimation_distance_m);
  c.minimum_curvature_points = integer(controller, "minimum_curvature_points");
  READ_CONTROLLER(curvature_lookahead_gain_m2);
  READ_CONTROLLER(lateral_error_lookahead_gain);
#undef READ_CONTROLLER
  const cv::FileNode speed = storage["speed"];
#define READ_SPEED(field) c.field = number(speed, #field)
  READ_SPEED(cruise_speed_mps); READ_SPEED(maximum_speed_mps);
  READ_SPEED(minimum_tracking_confidence); READ_SPEED(full_speed_confidence);
  READ_SPEED(maximum_acceleration_mps2); READ_SPEED(maximum_deceleration_mps2);
#undef READ_SPEED
  validate(c);
  return c;
}

void DisabledActuatorSink::record(const NativeShadowCommand&) { ++command_count_; }

NativeShadowController::NativeShadowController(NativeShadowConfig config,
                                               int processing_width,
                                               int processing_height)
    : config_(std::move(config)), processing_width_(processing_width),
      processing_height_(processing_height) {
  validate(config_);
  if (processing_width_ <= 0 || processing_height_ <= 0) {
    throw std::runtime_error("invalid shadow processing dimensions");
  }
}

void NativeShadowController::reset() {
  steering_rad_ = 0.0; estimated_speed_mps_ = 0.0; lost_time_s_ = 0.0;
  sequence_ = 0;
}

bool NativeShadowController::project_ground(double nx, double ny,
                                            PathPoint* output) const {
  const double px = nx * config_.source_width;
  const double py = ny * config_.source_height;
  const double xd = (px - config_.cx) / config_.fx;
  const double yd = (py - config_.cy) / config_.fy;
  cv::Vec3d ray;
  if (config_.lens_model == "fisheye_equidistant") {
    const double rd = std::hypot(xd, yd);
    if (rd < 1e-12) {
      ray = {0.0, 0.0, 1.0};
    } else {
      double theta = rd;
      for (int iteration = 0; iteration < 8; ++iteration) {
        const double t2 = theta * theta, t4 = t2 * t2;
        const double t6 = t4 * t2, t8 = t4 * t4;
        const double value = theta * (1.0 + config_.distortion[0] * t2 +
          config_.distortion[1] * t4 + config_.distortion[2] * t6 +
          config_.distortion[3] * t8) - rd;
        const double derivative = 1.0 + 3.0 * config_.distortion[0] * t2 +
          5.0 * config_.distortion[1] * t4 + 7.0 * config_.distortion[2] * t6 +
          9.0 * config_.distortion[3] * t8;
        theta -= value / std::max(derivative, 1e-12);
      }
      const double scale = std::sin(theta) / rd;
      ray = {xd * scale, yd * scale, std::cos(theta)};
    }
  } else {
    double x = xd, y = yd;
    for (int iteration = 0; iteration < 8; ++iteration) {
      const double r2 = x*x + y*y;
      const double radial = 1.0 + config_.distortion[0]*r2 +
        config_.distortion[1]*r2*r2 + config_.distortion[4]*r2*r2*r2;
      const double dx = 2.0*config_.distortion[2]*x*y +
        config_.distortion[3]*(r2 + 2.0*x*x);
      const double dy = config_.distortion[2]*(r2 + 2.0*y*y) +
        2.0*config_.distortion[3]*x*y;
      x = (xd-dx)/std::max(radial, 1e-12);
      y = (yd-dy)/std::max(radial, 1e-12);
    }
    ray = cv::normalize(cv::Vec3d(x, y, 1.0));
  }
  const cv::Matx33d camera_to_vehicle(0, 0, 1, -1, 0, 0, 0, -1, 0);
  const cv::Matx33d mount = rotation_z(config_.mount_yaw_rad) *
      rotation_y(config_.mount_pitch_down_rad) * rotation_x(config_.mount_roll_rad);
  const cv::Vec3d vehicle_ray = mount * camera_to_vehicle * ray;
  if (vehicle_ray[2] >= -1e-9) return false;
  const double scale = -config_.mount_z_m / vehicle_ray[2];
  const double forward = config_.mount_x_m + vehicle_ray[0] * scale;
  if (forward <= 0.0) return false;
  const double lateral = config_.mount_y_m + vehicle_ray[1] * scale;
  output->pixel = {nx * processing_width_, ny * processing_height_};
  output->vehicle = {forward, lateral};
  output->distance_m = std::hypot(forward, lateral);
  return true;
}

std::vector<NativeShadowController::PathPoint> NativeShadowController::project_path(
    const std::vector<cv::Point2f>& normalized) const {
  std::vector<PathPoint> path;
  path.reserve(normalized.size());
  for (const cv::Point2f& point : normalized) {
    PathPoint projected;
    if (project_ground(point.x, point.y, &projected)) path.push_back(projected);
  }
  return path;
}

double NativeShadowController::filtered_steering(double raw, double dt_s) {
  double smoothed = raw;
  if (config_.steering_smoothing_time_s > 0.0) {
    const double alpha = 1.0 - std::exp(-dt_s/config_.steering_smoothing_time_s);
    smoothed = steering_rad_ + alpha*(raw-steering_rad_);
  }
  const double maximum_change = config_.maximum_steering_rate_rad_s * dt_s;
  steering_rad_ += clamp(smoothed-steering_rad_, -maximum_change, maximum_change);
  steering_rad_ = clamp(steering_rad_, -config_.maximum_steering_rad,
                        config_.maximum_steering_rad);
  return steering_rad_;
}

NativeShadowCommand NativeShadowController::update(const ColorLaneNativeResult& lane,
                                                   double dt_s,
                                                   double timestamp_s) {
  if (!std::isfinite(dt_s) || dt_s <= 0.0) throw std::runtime_error("shadow dt must be positive");
  NativeShadowCommand command;
  command.sequence = sequence_++; command.timestamp_s = timestamp_s;
  command.confidence = lane.confidence;
  const std::vector<PathPoint> path = project_path(lane.center_path_normalized);
  command.projected_points = static_cast<int>(path.size());
  const bool tracking = !path.empty() && lane.confidence >= config_.minimum_tracking_confidence;
  if (!tracking) {
    lost_time_s_ += dt_s;
    command.reason = path.empty() ? "road_not_found" : "low_confidence";
    command.raw_steering_rad = 0.0;
    const double fallback = lost_time_s_ <= config_.lost_steering_hold_s ? steering_rad_ : 0.0;
    command.steering_rad = filtered_steering(fallback, dt_s);
  } else {
    lost_time_s_ = 0.0;
    command.reason = "tracking";
    double nominal = clamp(config_.base_lookahead_m +
        estimated_speed_mps_*config_.speed_lookahead_s,
        config_.minimum_lookahead_m, config_.maximum_lookahead_m);
    std::vector<cv::Point2d> fit;
    for (const PathPoint& point : path) {
      if (point.vehicle.x > 0.0 &&
          point.vehicle.x <= config_.curvature_estimation_distance_m) {
        fit.push_back(point.vehicle);
      }
    }
    double lateral_error = 0.0, curvature = 0.0;
    if (fit.size() >= static_cast<std::size_t>(config_.minimum_curvature_points)) {
      cv::Mat design(static_cast<int>(fit.size()), 3, CV_64F);
      cv::Mat values(static_cast<int>(fit.size()), 1, CV_64F);
      for (int i=0; i<static_cast<int>(fit.size()); ++i) {
        design.at<double>(i,0)=fit[i].x*fit[i].x;
        design.at<double>(i,1)=fit[i].x; design.at<double>(i,2)=1.0;
        values.at<double>(i)=fit[i].y;
      }
      cv::Mat coefficients;
      if (cv::solve(design, values, coefficients, cv::DECOMP_QR)) {
        lateral_error=coefficients.at<double>(2);
        const double slope=coefficients.at<double>(1);
        curvature=2.0*coefficients.at<double>(0)/std::pow(1.0+slope*slope,1.5);
      }
    }
    command.requested_lookahead_m = clamp(nominal -
        config_.curvature_lookahead_gain_m2*std::abs(curvature) -
        config_.lateral_error_lookahead_gain*std::abs(lateral_error),
        config_.minimum_lookahead_m, config_.maximum_lookahead_m);
    const PathPoint& target = *std::min_element(path.begin(), path.end(),
      [&](const PathPoint& a,const PathPoint& b){return std::abs(a.distance_m-command.requested_lookahead_m)<std::abs(b.distance_m-command.requested_lookahead_m);});
    const PathPoint& near = *std::min_element(path.begin(), path.end(),
      [&](const PathPoint& a,const PathPoint& b){return std::abs(a.distance_m-config_.minimum_lookahead_m)<std::abs(b.distance_m-config_.minimum_lookahead_m);});
    command.actual_lookahead_m=target.distance_m;
    command.target_forward_m=target.vehicle.x; command.target_lateral_m=target.vehicle.y;
    const double squared=std::max(target.vehicle.dot(target.vehicle),1e-9);
    const double curvature_steering=std::atan(config_.pure_pursuit_gain*2.0*
      config_.wheelbase_m*target.vehicle.y/squared);
    const double lateral_steering=std::atan(config_.lateral_error_gain*near.vehicle.y/
      (std::abs(estimated_speed_mps_)+config_.lateral_speed_softening_mps));
    command.raw_steering_rad=clamp(curvature_steering+lateral_steering,
      -config_.maximum_steering_rad,config_.maximum_steering_rad);
    command.steering_rad=filtered_steering(command.raw_steering_rad,dt_s);
  }
  if (tracking) {
    const double scale=clamp((lane.confidence-config_.minimum_tracking_confidence)/
      (config_.full_speed_confidence-config_.minimum_tracking_confidence),0.0,1.0);
    command.requested_speed_mps=config_.cruise_speed_mps*scale;
  }
  command.requested_speed_mps=std::min(command.requested_speed_mps,config_.maximum_speed_mps);
  const double rate=command.requested_speed_mps>=estimated_speed_mps_ ?
    config_.maximum_acceleration_mps2 : config_.maximum_deceleration_mps2;
  const double delta=clamp(command.requested_speed_mps-estimated_speed_mps_,-rate*dt_s,rate*dt_s);
  estimated_speed_mps_=clamp(estimated_speed_mps_+delta,0.0,config_.maximum_speed_mps);
  command.estimated_speed_mps=estimated_speed_mps_;
  actuator_.record(command);
  return command;
}

std::string native_shadow_command_json(const NativeShadowCommand& c) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << "{\"sequence\":" << c.sequence
      << ",\"timestamp_s\":" << c.timestamp_s
      << ",\"requested_speed_mps\":" << c.requested_speed_mps
      << ",\"estimated_speed_mps\":" << c.estimated_speed_mps
      << ",\"steering_rad\":" << c.steering_rad
      << ",\"raw_steering_rad\":" << c.raw_steering_rad
      << ",\"confidence\":" << c.confidence
      << ",\"requested_lookahead_m\":" << c.requested_lookahead_m
      << ",\"actual_lookahead_m\":" << c.actual_lookahead_m
      << ",\"target_forward_m\":" << c.target_forward_m
      << ",\"target_lateral_m\":" << c.target_lateral_m
      << ",\"projected_points\":" << c.projected_points
      << ",\"reason\":\"" << c.reason
      << "\",\"actuator_write_attempted\":false}";
  return out.str();
}
