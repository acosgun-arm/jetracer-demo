#include "jetracer_sim/simulator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace jetracer::sim {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;
constexpr double kEpsilon = 1e-9;

double clamp_value(double value, double low, double high) {
  return std::max(low, std::min(high, value));
}

cv::Scalar configured_scalar(const std::array<int, 3>& values) {
  return {static_cast<double>(values[0]), static_cast<double>(values[1]),
          static_cast<double>(values[2])};
}

cv::Scalar configured_scalar(const std::array<int, 4>& values) {
  return {static_cast<double>(values[0]), static_cast<double>(values[1]),
          static_cast<double>(values[2]), static_cast<double>(values[3])};
}

std::array<std::uint8_t, 3> configured_object_bgr(
    const std::array<int, 3>& values) {
  return {
      cv::saturate_cast<std::uint8_t>(values[0]),
      cv::saturate_cast<std::uint8_t>(values[1]),
      cv::saturate_cast<std::uint8_t>(values[2]),
  };
}

double wrap_angle(double value) {
  while (value > kPi) value -= 2.0 * kPi;
  while (value < -kPi) value += 2.0 * kPi;
  return value;
}

cv::Matx33d rotation_x(double angle) {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {1, 0, 0, 0, c, -s, 0, s, c};
}

cv::Matx33d rotation_y(double angle) {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {c, 0, s, 0, 1, 0, -s, 0, c};
}

cv::Matx33d rotation_z(double angle) {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  return {c, -s, 0, s, c, 0, 0, 0, 1};
}

Point2 catmull_rom(const Point2& p0, const Point2& p1, const Point2& p2,
                   const Point2& p3, double t) {
  const double t2 = t * t;
  const double t3 = t2 * t;
  return {
      0.5 * ((2.0 * p1.x) + (-p0.x + p2.x) * t +
             (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2 +
             (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3),
      0.5 * ((2.0 * p1.y) + (-p0.y + p2.y) * t +
             (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2 +
             (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3),
  };
}

double point_distance(const Point2& a, const Point2& b) {
  return std::hypot(a.x - b.x, a.y - b.y);
}

std::string lens_to_string(LensModel model) {
  return model == LensModel::FisheyeEquidistant ? "fisheye_equidistant"
                                                 : "brown_conrady";
}

LensModel lens_from_string(const std::string& value) {
  if (value == "fisheye_equidistant") return LensModel::FisheyeEquidistant;
  if (value == "brown_conrady") return LensModel::BrownConrady;
  throw std::runtime_error("unknown lens model: " + value);
}

std::string shutter_to_string(ShutterType shutter) {
  return shutter == ShutterType::Global ? "global" : "rolling";
}

ShutterType shutter_from_string(const std::string& value) {
  if (value == "global") return ShutterType::Global;
  if (value == "rolling") return ShutterType::Rolling;
  throw std::runtime_error("unknown shutter type: " + value);
}

struct ConfiguredCameraProfile {
  std::string_view id;
  int width;
  int height;
  std::int64_t fps_numerator;
  std::int64_t fps_denominator;
  double nominal_hfov_degrees;
  std::string_view lens_model;
  std::string_view shutter;
  std::array<double, 5> distortion;
  double mount_x_m;
  double mount_y_m;
  double mount_z_m;
  double mount_roll_rad;
  double mount_pitch_down_rad;
  double mount_yaw_rad;
  double exposure_s;
  double rolling_readout_s;
  bool provisional;
};

constexpr ConfiguredCameraProfile kStressCameraDefaults{
    defaults::kCameraProfilesStressId,
    defaults::kCameraProfilesStressWidth,
    defaults::kCameraProfilesStressHeight,
    defaults::kCameraProfilesStressFpsNumerator,
    defaults::kCameraProfilesStressFpsDenominator,
    defaults::kCameraProfilesStressNominalHfovDegrees,
    defaults::kCameraProfilesStressLensModel,
    defaults::kCameraProfilesStressShutter,
    defaults::kCameraProfilesStressDistortion,
    defaults::kCameraProfilesStressMountXM,
    defaults::kCameraProfilesStressMountYM,
    defaults::kCameraProfilesStressMountZM,
    defaults::kCameraProfilesStressMountRollRad,
    defaults::kCameraProfilesStressMountPitchDownRad,
    defaults::kCameraProfilesStressMountYawRad,
    defaults::kCameraProfilesStressExposureS,
    defaults::kCameraProfilesStressRollingReadoutS,
    defaults::kCameraProfilesStressProvisional,
};

constexpr ConfiguredCameraProfile kElpCameraDefaults{
    defaults::kCameraProfilesElpId,
    defaults::kCameraProfilesElpWidth,
    defaults::kCameraProfilesElpHeight,
    defaults::kCameraProfilesElpFpsNumerator,
    defaults::kCameraProfilesElpFpsDenominator,
    defaults::kCameraProfilesElpNominalHfovDegrees,
    defaults::kCameraProfilesElpLensModel,
    defaults::kCameraProfilesElpShutter,
    defaults::kCameraProfilesElpDistortion,
    defaults::kCameraProfilesElpMountXM,
    defaults::kCameraProfilesElpMountYM,
    defaults::kCameraProfilesElpMountZM,
    defaults::kCameraProfilesElpMountRollRad,
    defaults::kCameraProfilesElpMountPitchDownRad,
    defaults::kCameraProfilesElpMountYawRad,
    defaults::kCameraProfilesElpExposureS,
    defaults::kCameraProfilesElpRollingReadoutS,
    defaults::kCameraProfilesElpProvisional,
};

constexpr ConfiguredCameraProfile kImx219CameraDefaults{
    defaults::kCameraProfilesImx219Id,
    defaults::kCameraProfilesImx219Width,
    defaults::kCameraProfilesImx219Height,
    defaults::kCameraProfilesImx219FpsNumerator,
    defaults::kCameraProfilesImx219FpsDenominator,
    defaults::kCameraProfilesImx219NominalHfovDegrees,
    defaults::kCameraProfilesImx219LensModel,
    defaults::kCameraProfilesImx219Shutter,
    defaults::kCameraProfilesImx219Distortion,
    defaults::kCameraProfilesImx219MountXM,
    defaults::kCameraProfilesImx219MountYM,
    defaults::kCameraProfilesImx219MountZM,
    defaults::kCameraProfilesImx219MountRollRad,
    defaults::kCameraProfilesImx219MountPitchDownRad,
    defaults::kCameraProfilesImx219MountYawRad,
    defaults::kCameraProfilesImx219ExposureS,
    defaults::kCameraProfilesImx219RollingReadoutS,
    defaults::kCameraProfilesImx219Provisional,
};

CameraProfile make_camera_profile(const ConfiguredCameraProfile& configured) {
  CameraProfile profile;
  profile.id = configured.id;
  profile.width = configured.width;
  profile.height = configured.height;
  profile.fps_numerator = configured.fps_numerator;
  profile.fps_denominator = configured.fps_denominator;
  profile.nominal_hfov_rad = configured.nominal_hfov_degrees * kPi / 180.0;
  profile.lens_model = lens_from_string(std::string(configured.lens_model));
  profile.shutter = shutter_from_string(std::string(configured.shutter));
  profile.distortion = configured.distortion;
  profile.mount_x_m = configured.mount_x_m;
  profile.mount_y_m = configured.mount_y_m;
  profile.mount_z_m = configured.mount_z_m;
  profile.mount_roll_rad = configured.mount_roll_rad;
  profile.mount_pitch_down_rad = configured.mount_pitch_down_rad;
  profile.mount_yaw_rad = configured.mount_yaw_rad;
  profile.exposure_s = configured.exposure_s;
  profile.rolling_readout_s = configured.rolling_readout_s;
  profile.provisional = configured.provisional;
  profile.apply_nominal_intrinsics();
  return profile;
}

CameraProfile make_camera_profile(std::string_view alias) {
  if (alias == "stress") return make_camera_profile(kStressCameraDefaults);
  if (alias == "elp") return make_camera_profile(kElpCameraDefaults);
  if (alias == "imx219") return make_camera_profile(kImx219CameraDefaults);
  throw std::invalid_argument("unknown configured camera profile: " +
                              std::string(alias));
}

std::string object_to_string(ObjectType type) {
  return type == ObjectType::StopSign ? "stop_sign" : "box";
}

ObjectType object_from_string(const std::string& value) {
  if (value == "stop_sign") return ObjectType::StopSign;
  if (value == "box") return ObjectType::Box;
  throw std::runtime_error("unknown object type: " + value);
}

void write_camera(cv::FileStorage& fs, const CameraProfile& camera) {
  fs << "camera"
     << "{";
  fs << "id" << camera.id << "width" << camera.width << "height"
     << camera.height << "fps_numerator"
     << std::to_string(camera.fps_numerator) << "fps_denominator"
     << std::to_string(camera.fps_denominator) << "pixel_format"
     << "nv12_video_range" << "lens_model" << lens_to_string(camera.lens_model)
     << "shutter" << shutter_to_string(camera.shutter) << "nominal_hfov_rad"
     << camera.nominal_hfov_rad << "fx" << camera.fx << "fy" << camera.fy
     << "cx" << camera.cx << "cy" << camera.cy;
  fs << "distortion"
     << "[";
  for (double value : camera.distortion) fs << value;
  fs << "]";
  fs << "mount"
     << "{"
     << "x_m" << camera.mount_x_m << "y_m" << camera.mount_y_m << "z_m"
     << camera.mount_z_m << "roll_rad" << camera.mount_roll_rad
     << "pitch_down_rad" << camera.mount_pitch_down_rad << "yaw_rad"
     << camera.mount_yaw_rad << "}";
  fs << "exposure_s" << camera.exposure_s << "rolling_readout_s"
     << camera.rolling_readout_s << "provisional" << camera.provisional << "}";
}

CameraProfile read_camera(const cv::FileNode& node) {
  if (node.empty()) throw std::runtime_error("scene is missing camera profile");
  CameraProfile camera;
  node["id"] >> camera.id;
  node["width"] >> camera.width;
  node["height"] >> camera.height;
  std::string numerator;
  std::string denominator;
  node["fps_numerator"] >> numerator;
  node["fps_denominator"] >> denominator;
  camera.fps_numerator = std::stoll(numerator);
  camera.fps_denominator = std::stoll(denominator);
  std::string pixel_format;
  node["pixel_format"] >> pixel_format;
  if (pixel_format != "nv12_video_range") {
    throw std::runtime_error("unsupported camera pixel format: " + pixel_format);
  }
  camera.pixel_format = PixelFormat::Nv12VideoRange;
  std::string lens;
  std::string shutter;
  node["lens_model"] >> lens;
  node["shutter"] >> shutter;
  camera.lens_model = lens_from_string(lens);
  camera.shutter = shutter_from_string(shutter);
  node["nominal_hfov_rad"] >> camera.nominal_hfov_rad;
  node["fx"] >> camera.fx;
  node["fy"] >> camera.fy;
  node["cx"] >> camera.cx;
  node["cy"] >> camera.cy;
  int index = 0;
  for (const auto& value : node["distortion"]) {
    if (index < static_cast<int>(camera.distortion.size())) {
      camera.distortion[index++] = static_cast<double>(value);
    }
  }
  const cv::FileNode mount = node["mount"];
  mount["x_m"] >> camera.mount_x_m;
  mount["y_m"] >> camera.mount_y_m;
  mount["z_m"] >> camera.mount_z_m;
  mount["roll_rad"] >> camera.mount_roll_rad;
  mount["pitch_down_rad"] >> camera.mount_pitch_down_rad;
  mount["yaw_rad"] >> camera.mount_yaw_rad;
  node["exposure_s"] >> camera.exposure_s;
  node["rolling_readout_s"] >> camera.rolling_readout_s;
  int provisional = 0;
  node["provisional"] >> provisional;
  camera.provisional = provisional != 0;
  camera.validate();
  return camera;
}

struct ProjectedVertex {
  double x{0.0};
  double y{0.0};
  double depth{0.0};
  double u{0.0};
  double v{0.0};
  bool valid{false};
};

struct CameraTransform {
  cv::Vec3d origin{};
  cv::Matx33d camera_to_world{};
};

CameraTransform camera_transform(const VehicleState& state,
                                 const CameraProfile& camera) {
  const cv::Matx33d vehicle_to_world = rotation_z(state.pose.yaw);
  const cv::Matx33d mount_rotation =
      rotation_z(camera.mount_yaw_rad) * rotation_y(camera.mount_pitch_down_rad) *
      rotation_x(camera.mount_roll_rad);
  // Camera coordinates are +x right, +y down, +z forward. Vehicle coordinates
  // are +x forward, +y left, +z up.
  const cv::Matx33d camera_to_vehicle(0, 0, 1, -1, 0, 0, 0, -1, 0);
  const cv::Vec3d mount(camera.mount_x_m, camera.mount_y_m, camera.mount_z_m);
  return {
      cv::Vec3d(state.pose.x, state.pose.y, 0.0) + vehicle_to_world * mount,
      vehicle_to_world * mount_rotation * camera_to_vehicle,
  };
}

cv::Vec3f inverse_project_ray(const CameraProfile& camera, int px, int py) {
  const double xd = (static_cast<double>(px) - camera.cx) / camera.fx;
  const double yd = (static_cast<double>(py) - camera.cy) / camera.fy;
  double x = xd;
  double y = yd;
  if (camera.lens_model == LensModel::BrownConrady) {
    const double k1 = camera.distortion[0];
    const double k2 = camera.distortion[1];
    const double p1 = camera.distortion[2];
    const double p2 = camera.distortion[3];
    const double k3 = camera.distortion[4];
    for (int iteration = 0;
         iteration < defaults::kRendererInverseProjectionIterations;
         ++iteration) {
      const double r2 = x * x + y * y;
      const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
      const double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
      const double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
      x = (xd - dx) / std::max(radial, kEpsilon);
      y = (yd - dy) / std::max(radial, kEpsilon);
    }
    cv::Vec3d ray(x, y, 1.0);
    ray /= cv::norm(ray);
    return cv::Vec3f(ray);
  }

  const double radius_distorted = std::hypot(xd, yd);
  if (radius_distorted < kEpsilon) return {0.0F, 0.0F, 1.0F};
  double theta = radius_distorted;
  for (int iteration = 0;
       iteration < defaults::kRendererInverseProjectionIterations;
       ++iteration) {
    const double t2 = theta * theta;
    const double t4 = t2 * t2;
    const double t6 = t4 * t2;
    const double t8 = t4 * t4;
    const double value =
        theta * (1.0 + camera.distortion[0] * t2 +
                 camera.distortion[1] * t4 + camera.distortion[2] * t6 +
                 camera.distortion[3] * t8) -
        radius_distorted;
    const double derivative =
        1.0 + 3.0 * camera.distortion[0] * t2 +
        5.0 * camera.distortion[1] * t4 +
        7.0 * camera.distortion[2] * t6 +
        9.0 * camera.distortion[3] * t8;
    theta -= value / std::max(derivative, kEpsilon);
  }
  const double scale = std::sin(theta) / radius_distorted;
  return {static_cast<float>(xd * scale), static_cast<float>(yd * scale),
          static_cast<float>(std::cos(theta))};
}

ProjectedVertex project_point(const cv::Vec3d& point, const CameraProfile& camera,
                              const CameraTransform& transform, double u = 0.0,
                              double v = 0.0) {
  const cv::Vec3d relative = point - transform.origin;
  const cv::Vec3d local = transform.camera_to_world.t() * relative;
  if (local[2] <= defaults::kRendererNearClipM) return {};
  const double depth = cv::norm(relative);
  double image_x = 0.0;
  double image_y = 0.0;
  if (camera.lens_model == LensModel::BrownConrady) {
    const double x = local[0] / local[2];
    const double y = local[1] / local[2];
    const double r2 = x * x + y * y;
    const double radial = 1.0 + camera.distortion[0] * r2 +
                          camera.distortion[1] * r2 * r2 +
                          camera.distortion[4] * r2 * r2 * r2;
    const double xd = x * radial + 2.0 * camera.distortion[2] * x * y +
                      camera.distortion[3] * (r2 + 2.0 * x * x);
    const double yd = y * radial + camera.distortion[2] * (r2 + 2.0 * y * y) +
                      2.0 * camera.distortion[3] * x * y;
    image_x = camera.fx * xd + camera.cx;
    image_y = camera.fy * yd + camera.cy;
  } else {
    const double radial = std::hypot(local[0], local[1]);
    const double theta = std::atan2(radial, local[2]);
    const double t2 = theta * theta;
    const double theta_distorted =
        theta * (1.0 + camera.distortion[0] * t2 +
                 camera.distortion[1] * t2 * t2 +
                 camera.distortion[2] * t2 * t2 * t2 +
                 camera.distortion[3] * t2 * t2 * t2 * t2);
    const double scale = radial < kEpsilon ? 1.0 : theta_distorted / radial;
    image_x = camera.fx * local[0] * scale + camera.cx;
    image_y = camera.fy * local[1] * scale + camera.cy;
  }
  return {image_x, image_y, depth, u, v, true};
}

double edge_function(double ax, double ay, double bx, double by, double px,
                     double py) {
  return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

struct Atlas {
  cv::Mat bgr;
  cv::Mat semantic;
  cv::Mat instance;
  double min_x{0.0};
  double max_y{0.0};
  double pixels_per_metre{defaults::kSceneGenerationAtlasPixelsPerMetre};
};

cv::Point atlas_point(const Point2& point, const Atlas& atlas) {
  return {static_cast<int>(std::lround((point.x - atlas.min_x) * atlas.pixels_per_metre)),
          static_cast<int>(std::lround((atlas.max_y - point.y) * atlas.pixels_per_metre))};
}

Atlas build_atlas(const Scene& scene) {
  double min_x = std::numeric_limits<double>::max();
  double min_y = min_x;
  double max_x = std::numeric_limits<double>::lowest();
  double max_y = max_x;
  for (const Point2& point : scene.centerline) {
    min_x = std::min(min_x, point.x);
    min_y = std::min(min_y, point.y);
    max_x = std::max(max_x, point.x);
    max_y = std::max(max_y, point.y);
  }
  constexpr double margin = defaults::kRendererAtlasMarginM;
  min_x -= margin;
  min_y -= margin;
  max_x += margin;
  max_y += margin;
  Atlas atlas;
  atlas.min_x = min_x;
  atlas.max_y = max_y;
  atlas.pixels_per_metre = scene.atlas_pixels_per_metre;
  const int width = std::max(
      defaults::kRendererMinimumAtlasDimensionPixels,
      static_cast<int>(
          std::ceil((max_x - min_x) * atlas.pixels_per_metre)));
  const int height = std::max(
      defaults::kRendererMinimumAtlasDimensionPixels,
      static_cast<int>(
          std::ceil((max_y - min_y) * atlas.pixels_per_metre)));
  atlas.bgr.create(height, width, CV_8UC3);
  atlas.semantic = cv::Mat(height, width, CV_8UC1,
                           cv::Scalar(static_cast<int>(SemanticClass::Background)));
  atlas.instance = cv::Mat(height, width, CV_32SC1, cv::Scalar(0));

  cv::RNG rng(static_cast<std::uint64_t>(scene.seed));
  cv::Mat background_texture;
  if (!scene.background_texture_path.empty()) {
    background_texture = cv::imread(scene.background_texture_path, cv::IMREAD_COLOR);
    if (background_texture.empty()) {
      throw std::runtime_error("cannot load background texture: " +
                               scene.background_texture_path);
    }
  }
  for (int y = 0; y < height; ++y) {
    auto* row = atlas.bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < width; ++x) {
      if (!background_texture.empty()) {
        row[x] = background_texture.at<cv::Vec3b>(y % background_texture.rows,
                                                  x % background_texture.cols);
      } else {
        const int checker =
            ((x / defaults::kRendererBackgroundCheckerWidthPixels) +
             (y / defaults::kRendererBackgroundCheckerHeightPixels)) &
            1;
        const int noise = rng.uniform(
            defaults::kRendererBackgroundNoiseMin,
            defaults::kRendererBackgroundNoiseMaxExclusive);
        cv::Vec3b background;
        for (std::size_t channel = 0;
             channel < defaults::kRendererBackgroundBaseBgr.size();
             ++channel) {
          background[channel] = cv::saturate_cast<std::uint8_t>(
              defaults::kRendererBackgroundBaseBgr[channel] + noise +
              checker *
                  defaults::kRendererBackgroundCheckerBgrIncrement[channel]);
        }
        row[x] = background;
      }
    }
  }
  for (int index = 0; index < defaults::kRendererClutterCount; ++index) {
    const cv::Point centre(rng.uniform(0, width), rng.uniform(0, height));
    const int radius = rng.uniform(
        defaults::kRendererClutterRadiusMinPixels,
        defaults::kRendererClutterRadiusMaxExclusivePixels);
    const int shade = rng.uniform(defaults::kRendererClutterShadeMin,
                                  defaults::kRendererClutterShadeMaxExclusive);
    const cv::Scalar colour(
        shade + defaults::kRendererClutterBgrIncrement[0],
        shade + defaults::kRendererClutterBgrIncrement[1],
        shade + defaults::kRendererClutterBgrIncrement[2]);
    cv::circle(atlas.bgr, centre, radius, colour, -1, cv::LINE_AA);
  }

  std::vector<Point2> left;
  std::vector<Point2> right;
  left.reserve(scene.centerline.size());
  right.reserve(scene.centerline.size());
  const double half_width = scene.road_width_m * 0.5;
  for (std::size_t index = 0; index < scene.centerline.size(); ++index) {
    const Point2& previous =
        scene.centerline[(index + scene.centerline.size() - 1) %
                         scene.centerline.size()];
    const Point2& next = scene.centerline[(index + 1) % scene.centerline.size()];
    const double length = std::max(std::hypot(next.x - previous.x, next.y - previous.y), kEpsilon);
    const double nx = -(next.y - previous.y) / length;
    const double ny = (next.x - previous.x) / length;
    left.push_back({scene.centerline[index].x + nx * half_width,
                    scene.centerline[index].y + ny * half_width});
    right.push_back({scene.centerline[index].x - nx * half_width,
                     scene.centerline[index].y - ny * half_width});
  }
  std::vector<cv::Point> road;
  std::vector<cv::Point> left_pixels;
  std::vector<cv::Point> right_pixels;
  for (const Point2& point : left) {
    left_pixels.push_back(atlas_point(point, atlas));
    road.push_back(left_pixels.back());
  }
  for (auto iterator = right.rbegin(); iterator != right.rend(); ++iterator) {
    right_pixels.push_back(atlas_point(*iterator, atlas));
    road.push_back(right_pixels.back());
  }
  std::vector<std::vector<cv::Point>> polygons{road};
  cv::fillPoly(atlas.bgr, polygons,
               configured_scalar(defaults::kRendererRoadBgr), cv::LINE_8);
  cv::fillPoly(atlas.semantic, polygons,
               cv::Scalar(static_cast<int>(SemanticClass::DrivableSurface)),
               cv::LINE_8);

  cv::Mat road_texture;
  if (!scene.road_texture_path.empty()) {
    road_texture = cv::imread(scene.road_texture_path, cv::IMREAD_COLOR);
    if (road_texture.empty()) {
      throw std::runtime_error("cannot load road texture: " +
                               scene.road_texture_path);
    }
  }
  cv::Mat road_noise(height, width, CV_8SC1);
  rng.fill(road_noise, cv::RNG::NORMAL, 0,
           defaults::kRendererRoadNoiseStandardDeviation);
  for (int y = 0; y < height; ++y) {
    auto* colour = atlas.bgr.ptr<cv::Vec3b>(y);
    const auto* labels = atlas.semantic.ptr<std::uint8_t>(y);
    const auto* noise = road_noise.ptr<std::int8_t>(y);
    for (int x = 0; x < width; ++x) {
      if (labels[x] == static_cast<std::uint8_t>(SemanticClass::DrivableSurface)) {
        if (!road_texture.empty()) {
          colour[x] = road_texture.at<cv::Vec3b>(y % road_texture.rows,
                                                 x % road_texture.cols);
        } else {
          for (int channel = 0; channel < 3; ++channel) {
            colour[x][channel] = cv::saturate_cast<std::uint8_t>(
                colour[x][channel] + noise[x]);
          }
        }
      }
    }
  }

  const int lane_width = std::max(
      defaults::kRendererMinimumLaneWidthPixels,
      static_cast<int>(std::round(defaults::kRendererLaneWidthM *
                                  atlas.pixels_per_metre)));
  const std::vector<std::vector<cv::Point>> left_curves{left_pixels};
  const std::vector<std::vector<cv::Point>> right_curves{right_pixels};
  cv::polylines(atlas.bgr, left_curves, true,
                configured_scalar(defaults::kRendererLaneBgr), lane_width,
                cv::LINE_AA);
  cv::polylines(atlas.bgr, right_curves, true,
                configured_scalar(defaults::kRendererLaneBgr), lane_width,
                cv::LINE_AA);
  cv::polylines(atlas.semantic, left_curves, true,
                cv::Scalar(static_cast<int>(SemanticClass::LaneMarking)), lane_width,
                cv::LINE_8);
  cv::polylines(atlas.semantic, right_curves, true,
                cv::Scalar(static_cast<int>(SemanticClass::LaneMarking)), lane_width,
                cv::LINE_8);
  return atlas;
}

cv::Mat make_stop_texture() {
  constexpr int size = defaults::kRendererStopTextureSizePixels;
  cv::Mat texture(size, size, CV_8UC4, cv::Scalar(0, 0, 0, 0));
  auto octagon = [](double radius) {
    std::vector<cv::Point> points;
    for (int index = 0; index < 8; ++index) {
      const double angle = kPi / 8.0 + index * kPi / 4.0;
      points.emplace_back(
          static_cast<int>(std::lround(size * 0.5 + radius * std::cos(angle))),
          static_cast<int>(std::lround(size * 0.5 + radius * std::sin(angle))));
    }
    return points;
  };
  cv::fillConvexPoly(
      texture, octagon(defaults::kRendererStopTextureBorderRadiusPixels),
      configured_scalar(defaults::kRendererStopTextureBorderBgra), cv::LINE_AA);
  cv::fillConvexPoly(
      texture, octagon(defaults::kRendererStopTextureFaceRadiusPixels),
      configured_scalar(defaults::kRendererStopTextureFaceBgra), cv::LINE_AA);
  const std::string text = "STOP";
  const int font = cv::FONT_HERSHEY_DUPLEX;
  const double scale = defaults::kRendererStopTextureFontScale;
  const int thickness = defaults::kRendererStopTextureFontThickness;
  int baseline = 0;
  const cv::Size text_size = cv::getTextSize(text, font, scale, thickness, &baseline);
  cv::putText(texture, text,
              cv::Point((size - text_size.width) / 2,
                        (size + text_size.height) / 2),
              font, scale,
              configured_scalar(defaults::kRendererStopTextureTextBgra),
              thickness, cv::LINE_AA);
  return texture;
}

}  // namespace

double CameraProfile::fps() const {
  return static_cast<double>(fps_numerator) /
         static_cast<double>(fps_denominator);
}

double CameraProfile::frame_period_s() const { return 1.0 / fps(); }

void CameraProfile::apply_nominal_intrinsics() {
  if (nominal_hfov_rad <= 0.0 || nominal_hfov_rad >= kPi) {
    throw std::invalid_argument("camera HFOV must be between zero and pi");
  }
  fx = lens_model == LensModel::FisheyeEquidistant
           ? static_cast<double>(width) / nominal_hfov_rad
           : (static_cast<double>(width) * 0.5) /
                 std::tan(nominal_hfov_rad * 0.5);
  fy = fx;
  cx = (static_cast<double>(width) - 1.0) * 0.5;
  cy = (static_cast<double>(height) - 1.0) * 0.5;
}

void CameraProfile::apply_opencv_calibration(const std::string& path) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("cannot open OpenCV calibration: " + path);
  }
  cv::Mat matrix;
  cv::Mat coefficients;
  fs["camera_matrix"] >> matrix;
  if (matrix.empty()) fs["K"] >> matrix;
  fs["distortion_coefficients"] >> coefficients;
  if (coefficients.empty()) fs["D"] >> coefficients;
  if (matrix.rows != 3 || matrix.cols != 3 || coefficients.empty()) {
    throw std::runtime_error(
        "calibration must contain camera_matrix/K and distortion_coefficients/D");
  }
  matrix.convertTo(matrix, CV_64F);
  coefficients = coefficients.reshape(1, 1);
  coefficients.convertTo(coefficients, CV_64F);
  int calibrated_width = width;
  int calibrated_height = height;
  if (!fs["image_width"].empty()) fs["image_width"] >> calibrated_width;
  if (!fs["image_height"].empty()) fs["image_height"] >> calibrated_height;
  if (calibrated_width <= 0 || calibrated_height <= 0) {
    throw std::runtime_error("calibration image dimensions must be positive");
  }
  const double scale_x = static_cast<double>(width) / calibrated_width;
  const double scale_y = static_cast<double>(height) / calibrated_height;
  fx = matrix.at<double>(0, 0) * scale_x;
  fy = matrix.at<double>(1, 1) * scale_y;
  cx = matrix.at<double>(0, 2) * scale_x;
  cy = matrix.at<double>(1, 2) * scale_y;
  distortion.fill(0.0);
  for (int index = 0;
       index < std::min(coefficients.cols,
                        static_cast<int>(distortion.size()));
       ++index) {
    distortion[index] = coefficients.at<double>(0, index);
  }
  std::string model;
  if (!fs["lens_model"].empty()) {
    fs["lens_model"] >> model;
    lens_model = lens_from_string(model);
  }
  validate();
}

void CameraProfile::validate() const {
  if (id.empty()) throw std::invalid_argument("camera profile ID is empty");
  if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0) {
    throw std::invalid_argument("NV12 camera dimensions must be positive and even");
  }
  if (fps_numerator <= 0 || fps_denominator <= 0) {
    throw std::invalid_argument("camera frame rate must be positive");
  }
  if (pixel_format != PixelFormat::Nv12VideoRange) {
    throw std::invalid_argument("unsupported camera pixel format");
  }
  if (fx <= 0.0 || fy <= 0.0) {
    throw std::invalid_argument("camera focal lengths must be positive");
  }
  if (!std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) ||
      !std::isfinite(cy)) {
    throw std::invalid_argument("camera intrinsics must be finite");
  }
  if (exposure_s < 0.0 || rolling_readout_s < 0.0) {
    throw std::invalid_argument("camera timing values must not be negative");
  }
  if (mount_z_m <= 0.0) {
    throw std::invalid_argument("camera must be above the ground plane");
  }
}

CameraProfile CameraProfile::elp_112() {
  return make_camera_profile("elp");
}

CameraProfile CameraProfile::stress_720p_200() {
  return make_camera_profile("stress");
}

CameraProfile CameraProfile::imx219_160_provisional() {
  return make_camera_profile("imx219");
}

Scene Scene::generate(const SceneConfig& config) {
  if (config.control_points < defaults::kSceneGenerationMinimumControlPoints ||
      config.samples_per_segment <
          defaults::kSceneGenerationMinimumSamplesPerSegment) {
    throw std::invalid_argument(
        "track generator sampling is below the configured minimum");
  }
  if (config.road_width_m <= defaults::kSceneGenerationMinimumRoadWidthM ||
      config.base_radius_m <= config.road_width_m) {
    throw std::invalid_argument("invalid track radius or width");
  }
  if (config.radius_jitter_m < 0.0 ||
      config.atlas_pixels_per_metre <=
          defaults::kSceneGenerationMinimumAtlasPixelsPerMetre ||
      config.obstacle_count < 0 || config.stop_sign_count < 0) {
    throw std::invalid_argument("invalid track texture or object configuration");
  }
  Scene scene;
  scene.seed = config.seed;
  scene.road_width_m = config.road_width_m;
  scene.atlas_pixels_per_metre = config.atlas_pixels_per_metre;
  scene.background_texture_path = config.background_texture_path;
  scene.road_texture_path = config.road_texture_path;
  scene.camera =
      make_camera_profile(defaults::kSceneGenerationCameraProfile);
  std::mt19937_64 generator(config.seed);
  std::uniform_real_distribution<double> radius_delta(-config.radius_jitter_m,
                                                       config.radius_jitter_m);
  std::vector<Point2> controls;
  controls.reserve(config.control_points);
  for (int index = 0; index < config.control_points; ++index) {
    const double angle = 2.0 * kPi * static_cast<double>(index) /
                         static_cast<double>(config.control_points);
    const double radius = config.base_radius_m + radius_delta(generator);
    controls.push_back({radius * std::cos(angle), radius * std::sin(angle)});
  }
  for (int segment = 0; segment < config.control_points; ++segment) {
    const Point2& p0 = controls[(segment + config.control_points - 1) % config.control_points];
    const Point2& p1 = controls[segment];
    const Point2& p2 = controls[(segment + 1) % config.control_points];
    const Point2& p3 = controls[(segment + 2) % config.control_points];
    for (int sample = 0; sample < config.samples_per_segment; ++sample) {
      scene.centerline.push_back(catmull_rom(
          p0, p1, p2, p3,
          static_cast<double>(sample) / static_cast<double>(config.samples_per_segment)));
    }
  }
  scene.start.pose.x = scene.centerline.front().x;
  scene.start.pose.y = scene.centerline.front().y;
  scene.start.pose.yaw = std::atan2(scene.centerline[1].y - scene.centerline[0].y,
                                    scene.centerline[1].x - scene.centerline[0].x);

  std::uniform_real_distribution<double> size_distribution(
      defaults::kProceduralObjectsObstacleSizeMinM,
      defaults::kProceduralObjectsObstacleSizeMaxM);
  std::uniform_int_distribution<int> colour_distribution(
      defaults::kProceduralObjectsObstacleColourMin,
      defaults::kProceduralObjectsObstacleColourMax);
  std::uint32_t instance_id = 1;
  for (int object_index = 0; object_index < config.obstacle_count; ++object_index) {
    const std::size_t index = static_cast<std::size_t>(
        (object_index + 0.5) * scene.centerline.size() /
        std::max(1, config.obstacle_count));
    const Point2& previous =
        scene.centerline[(index + scene.centerline.size() - 1) %
                         scene.centerline.size()];
    const Point2& next = scene.centerline[(index + 1) % scene.centerline.size()];
    const double tangent = std::atan2(next.y - previous.y, next.x - previous.x);
    const double side = object_index % 2 == 0 ? 1.0 : -1.0;
    const double offset =
        side * (scene.road_width_m * 0.5 +
                defaults::kProceduralObjectsObstacleShoulderOffsetM);
    SceneObject object;
    object.instance_id = instance_id++;
    object.type = ObjectType::Box;
    object.semantic_class = SemanticClass::Obstacle;
    object.position = {scene.centerline[index].x - std::sin(tangent) * offset,
                       scene.centerline[index].y + std::cos(tangent) * offset};
    object.yaw_rad = tangent + size_distribution(generator);
    object.width_m = size_distribution(generator);
    object.depth_m = size_distribution(generator);
    object.height_m = size_distribution(generator) +
                      defaults::kProceduralObjectsObstacleHeightAdditionM;
    object.bgr = {static_cast<std::uint8_t>(colour_distribution(generator)),
                  static_cast<std::uint8_t>(colour_distribution(generator)),
                  static_cast<std::uint8_t>(colour_distribution(generator))};
    scene.objects.push_back(object);
  }
  for (int sign_index = 0; sign_index < config.stop_sign_count; ++sign_index) {
    const std::size_t index = static_cast<std::size_t>(
        (sign_index + 1.0) * scene.centerline.size() /
        (config.stop_sign_count + 1.0));
    const Point2& previous =
        scene.centerline[(index + scene.centerline.size() - 1) %
                         scene.centerline.size()];
    const Point2& next = scene.centerline[(index + 1) % scene.centerline.size()];
    const double tangent = std::atan2(next.y - previous.y, next.x - previous.x);
    const double offset =
        scene.road_width_m * 0.5 +
        defaults::kProceduralObjectsStopSignShoulderOffsetM;
    SceneObject sign;
    sign.instance_id = instance_id++;
    sign.type = ObjectType::StopSign;
    sign.semantic_class = SemanticClass::StopSign;
    sign.position = {scene.centerline[index].x - std::sin(tangent) * offset,
                     scene.centerline[index].y + std::cos(tangent) * offset};
    sign.yaw_rad = tangent + kPi;
    sign.width_m = defaults::kProceduralObjectsStopSignWidthM;
    sign.depth_m = defaults::kProceduralObjectsStopSignDepthM;
    sign.height_m = defaults::kProceduralObjectsStopSignHeightM;
    sign.bgr = configured_object_bgr(
        defaults::kProceduralObjectsStopSignBgr);
    scene.objects.push_back(sign);
  }
  scene.validate();
  return scene;
}

void Scene::validate() const {
  if (schema_version != kSceneSchemaVersion) {
    throw std::invalid_argument("unsupported scene schema version");
  }
  if (centerline.size() < static_cast<std::size_t>(
                              defaults::kSceneGenerationMinimumCenterlinePoints)) {
    throw std::invalid_argument("scene centerline is too short");
  }
  if (road_width_m <= defaults::kSceneGenerationMinimumRoadWidthM ||
      atlas_pixels_per_metre <=
          defaults::kSceneGenerationMinimumAtlasPixelsPerMetre) {
    throw std::invalid_argument("invalid scene dimensions");
  }
  if (vehicle.wheelbase_m <= 0.0 || vehicle.body_width_m <= 0.0 ||
      vehicle.front_overhang_m < 0.0 || vehicle.rear_overhang_m < 0.0 ||
      vehicle.max_steering_rad <= 0.0 || vehicle.max_steering_rad >= kPi * 0.5 ||
      vehicle.steering_time_constant_s < 0.0 ||
      vehicle.motor_time_constant_s < 0.0 ||
      !std::isfinite(vehicle.wheelbase_m) ||
      !std::isfinite(vehicle.body_width_m) ||
      !std::isfinite(vehicle.front_overhang_m) ||
      !std::isfinite(vehicle.rear_overhang_m)) {
    throw std::invalid_argument("invalid vehicle configuration");
  }
  for (std::size_t index = 0; index < centerline.size(); ++index) {
    const Point2& point = centerline[index];
    const Point2& next = centerline[(index + 1) % centerline.size()];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        point_distance(point, next) <= kEpsilon) {
      throw std::invalid_argument(
          "scene centerline must contain finite, distinct consecutive points");
    }
  }
  std::unordered_map<std::uint32_t, bool> identifiers;
  for (const SceneObject& object : objects) {
    if (object.instance_id == 0 ||
        object.instance_id >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        identifiers.contains(object.instance_id)) {
      throw std::invalid_argument("scene object instance IDs must be unique and non-zero");
    }
    if (object.width_m <= 0.0 || object.depth_m <= 0.0 ||
        object.height_m <= 0.0) {
      throw std::invalid_argument("scene object dimensions must be positive");
    }
    const int semantic = static_cast<int>(object.semantic_class);
    if (semantic < static_cast<int>(SemanticClass::Background) ||
        semantic > static_cast<int>(SemanticClass::Obstacle) ||
        !std::isfinite(object.position.x) ||
        !std::isfinite(object.position.y) || !std::isfinite(object.base_z_m) ||
        !std::isfinite(object.yaw_rad)) {
      throw std::invalid_argument("invalid scene object label or pose");
    }
    identifiers[object.instance_id] = true;
  }
  camera.validate();
}

void Scene::save(const std::string& path) const {
  validate();
  cv::FileStorage fs(path, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
  if (!fs.isOpened()) throw std::runtime_error("cannot open scene for writing: " + path);
  const std::filesystem::path scene_directory =
      std::filesystem::absolute(std::filesystem::path(path)).parent_path();
  auto portable_texture_path = [&](const std::string& texture_path) {
    if (texture_path.empty()) return std::string{};
    std::filesystem::path texture(texture_path);
    if (texture.is_relative()) texture = std::filesystem::absolute(texture);
    std::error_code error;
    const std::filesystem::path relative =
        std::filesystem::relative(texture, scene_directory, error);
    return error ? texture.lexically_normal().string()
                 : relative.lexically_normal().string();
  };
  fs << "schema_version" << schema_version << "seed" << std::to_string(seed)
     << "road_width_m" << road_width_m << "atlas_pixels_per_metre"
     << atlas_pixels_per_metre << "background_texture_path"
     << portable_texture_path(background_texture_path) << "road_texture_path"
     << portable_texture_path(road_texture_path);
  fs << "vehicle"
     << "{"
     << "wheelbase_m" << vehicle.wheelbase_m << "body_width_m"
     << vehicle.body_width_m << "front_overhang_m"
     << vehicle.front_overhang_m << "rear_overhang_m"
     << vehicle.rear_overhang_m << "max_steering_rad"
     << vehicle.max_steering_rad << "steering_time_constant_s"
     << vehicle.steering_time_constant_s << "motor_time_constant_s"
     << vehicle.motor_time_constant_s << "}";
  fs << "start"
     << "{"
     << "x" << start.pose.x << "y" << start.pose.y << "yaw" << start.pose.yaw
     << "speed_mps" << start.speed_mps << "steering_rad" << start.steering_rad
     << "}";
  write_camera(fs, camera);
  fs << "centerline"
     << "[";
  for (const Point2& point : centerline) fs << "{" << "x" << point.x << "y" << point.y << "}";
  fs << "]";
  fs << "objects"
     << "[";
  for (const SceneObject& object : objects) {
    fs << "{"
       << "instance_id" << static_cast<int>(object.instance_id) << "type"
       << object_to_string(object.type) << "semantic_class"
       << static_cast<int>(object.semantic_class) << "x" << object.position.x << "y"
       << object.position.y << "base_z_m" << object.base_z_m << "yaw_rad"
       << object.yaw_rad << "width_m" << object.width_m << "depth_m"
       << object.depth_m << "height_m" << object.height_m << "bgr"
       << "[" << static_cast<int>(object.bgr[0]) << static_cast<int>(object.bgr[1])
       << static_cast<int>(object.bgr[2]) << "]"
       << "}";
  }
  fs << "]";
}

Scene Scene::load(const std::string& path) {
  cv::FileStorage fs(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  if (!fs.isOpened()) throw std::runtime_error("cannot open scene: " + path);
  Scene scene;
  fs["schema_version"] >> scene.schema_version;
  std::string seed;
  fs["seed"] >> seed;
  scene.seed = std::stoull(seed);
  fs["road_width_m"] >> scene.road_width_m;
  fs["atlas_pixels_per_metre"] >> scene.atlas_pixels_per_metre;
  fs["background_texture_path"] >> scene.background_texture_path;
  fs["road_texture_path"] >> scene.road_texture_path;
  const std::filesystem::path scene_directory =
      std::filesystem::absolute(std::filesystem::path(path)).parent_path();
  auto resolve_texture_path = [&](std::string& texture_path) {
    if (texture_path.empty()) return;
    std::filesystem::path texture(texture_path);
    if (texture.is_relative()) {
      texture_path = (scene_directory / texture).lexically_normal().string();
    }
  };
  resolve_texture_path(scene.background_texture_path);
  resolve_texture_path(scene.road_texture_path);
  const cv::FileNode vehicle = fs["vehicle"];
  vehicle["wheelbase_m"] >> scene.vehicle.wheelbase_m;
  if (!vehicle["body_width_m"].empty()) {
    vehicle["body_width_m"] >> scene.vehicle.body_width_m;
  }
  if (!vehicle["front_overhang_m"].empty()) {
    vehicle["front_overhang_m"] >> scene.vehicle.front_overhang_m;
  }
  if (!vehicle["rear_overhang_m"].empty()) {
    vehicle["rear_overhang_m"] >> scene.vehicle.rear_overhang_m;
  }
  vehicle["max_steering_rad"] >> scene.vehicle.max_steering_rad;
  vehicle["steering_time_constant_s"] >> scene.vehicle.steering_time_constant_s;
  vehicle["motor_time_constant_s"] >> scene.vehicle.motor_time_constant_s;
  const cv::FileNode start = fs["start"];
  start["x"] >> scene.start.pose.x;
  start["y"] >> scene.start.pose.y;
  start["yaw"] >> scene.start.pose.yaw;
  start["speed_mps"] >> scene.start.speed_mps;
  start["steering_rad"] >> scene.start.steering_rad;
  scene.camera = read_camera(fs["camera"]);
  for (const auto& node : fs["centerline"]) {
    Point2 point;
    node["x"] >> point.x;
    node["y"] >> point.y;
    scene.centerline.push_back(point);
  }
  for (const auto& node : fs["objects"]) {
    SceneObject object;
    int instance_id = 0;
    int semantic_class = 0;
    std::string type;
    node["instance_id"] >> instance_id;
    node["type"] >> type;
    node["semantic_class"] >> semantic_class;
    object.instance_id = static_cast<std::uint32_t>(instance_id);
    object.type = object_from_string(type);
    object.semantic_class = static_cast<SemanticClass>(semantic_class);
    node["x"] >> object.position.x;
    node["y"] >> object.position.y;
    node["base_z_m"] >> object.base_z_m;
    node["yaw_rad"] >> object.yaw_rad;
    node["width_m"] >> object.width_m;
    node["depth_m"] >> object.depth_m;
    node["height_m"] >> object.height_m;
    int channel = 0;
    for (const auto& value : node["bgr"]) {
      if (channel < 3) object.bgr[channel++] = static_cast<std::uint8_t>(static_cast<int>(value));
    }
    scene.objects.push_back(object);
  }
  scene.validate();
  return scene;
}

double VehicleConfig::body_length_m() const {
  return rear_overhang_m + wheelbase_m + front_overhang_m;
}

double VehicleConfig::minimum_turn_radius_m() const {
  return wheelbase_m / std::tan(max_steering_rad);
}

void step_bicycle(const VehicleConfig& config, VehicleState& state,
                  const VehicleCommand& command, double dt_s) {
  if (dt_s < 0.0) throw std::invalid_argument("simulation dt must not be negative");
  if (dt_s == 0.0) return;
  const double requested_steering =
      clamp_value(command.steering_rad, -config.max_steering_rad,
                  config.max_steering_rad);
  const double steering_alpha =
      config.steering_time_constant_s <= 0.0
          ? 1.0
          : 1.0 - std::exp(-dt_s / config.steering_time_constant_s);
  const double motor_alpha =
      config.motor_time_constant_s <= 0.0
          ? 1.0
          : 1.0 - std::exp(-dt_s / config.motor_time_constant_s);
  state.steering_rad += (requested_steering - state.steering_rad) * steering_alpha;
  state.speed_mps += (command.target_speed_mps - state.speed_mps) * motor_alpha;
  const double yaw_rate =
      state.speed_mps /
      std::max(config.wheelbase_m,
               defaults::kRendererMinimumBicycleWheelbaseM) *
      std::tan(state.steering_rad);
  const double midpoint_yaw = state.pose.yaw + 0.5 * yaw_rate * dt_s;
  state.pose.x += state.speed_mps * std::cos(midpoint_yaw) * dt_s;
  state.pose.y += state.speed_mps * std::sin(midpoint_yaw) * dt_s;
  state.pose.yaw = wrap_angle(state.pose.yaw + yaw_rate * dt_s);
}

class Simulator::Impl {
 public:
  Impl(Scene value, CameraProfile profile)
      : scene(std::move(value)), camera(std::move(profile)) {
    scene.validate();
    camera.validate();
    atlas = build_atlas(scene);
    stop_texture = make_stop_texture();
    rebuild_ray_table();
    reset();
  }

  void rebuild_ray_table() {
    rays.resize(static_cast<std::size_t>(camera.width) * camera.height);
    cv::parallel_for_(cv::Range(0, camera.height), [&](const cv::Range& range) {
      for (int y = range.start; y < range.end; ++y) {
        for (int x = 0; x < camera.width; ++x) {
          rays[static_cast<std::size_t>(y) * camera.width + x] =
              inverse_project_ray(camera, x, y);
        }
      }
    });
  }

  void reset() {
    vehicle = scene.start;
    simulation_time = 0.0;
    scheduled_frame_index = 1;
    next_camera_time = scheduled_time(scheduled_frame_index);
    frame_id = 0;
  }

  [[nodiscard]] double scheduled_time(std::uint64_t index) const {
    return static_cast<double>(index) *
           static_cast<double>(camera.fps_denominator) /
           static_cast<double>(camera.fps_numerator);
  }

  void raster_triangle(const ProjectedVertex& a, const ProjectedVertex& b,
                       const ProjectedVertex& c, const cv::Vec3b& colour,
                       SemanticClass semantic_class, std::uint32_t instance_id,
                       const cv::Mat* texture) {
    if (!a.valid || !b.valid || !c.valid) return;
    const double area = edge_function(a.x, a.y, b.x, b.y, c.x, c.y);
    if (std::abs(area) < defaults::kRendererTriangleAreaEpsilon) return;
    const int min_x = std::max(0, static_cast<int>(std::floor(std::min({a.x, b.x, c.x}))));
    const int max_x = std::min(camera.width - 1,
                               static_cast<int>(std::ceil(std::max({a.x, b.x, c.x}))));
    const int min_y = std::max(0, static_cast<int>(std::floor(std::min({a.y, b.y, c.y}))));
    const int max_y = std::min(camera.height - 1,
                               static_cast<int>(std::ceil(std::max({a.y, b.y, c.y}))));
    if (min_x > max_x || min_y > max_y) return;
    for (int y = min_y; y <= max_y; ++y) {
      auto* depth_row = depth.ptr<float>(y);
      auto* colour_row = render_bgr.ptr<cv::Vec3b>(y);
      auto* semantic_row = render_semantic.ptr<std::uint8_t>(y);
      auto* instance_row = render_instance.ptr<std::int32_t>(y);
      for (int x = min_x; x <= max_x; ++x) {
        const double px = x + 0.5;
        const double py = y + 0.5;
        const double w0 = edge_function(b.x, b.y, c.x, c.y, px, py) / area;
        const double w1 = edge_function(c.x, c.y, a.x, a.y, px, py) / area;
        const double w2 = 1.0 - w0 - w1;
        if (w0 < -defaults::kRendererTriangleEdgeEpsilon ||
            w1 < -defaults::kRendererTriangleEdgeEpsilon ||
            w2 < -defaults::kRendererTriangleEdgeEpsilon) {
          continue;
        }
        const double inverse_depth = w0 / a.depth + w1 / b.depth + w2 / c.depth;
        if (inverse_depth <= 0.0) continue;
        const float pixel_depth = static_cast<float>(1.0 / inverse_depth);
        if (pixel_depth >= depth_row[x]) continue;
        cv::Vec3b pixel = colour;
        if (texture != nullptr) {
          const double tu = (w0 * a.u / a.depth + w1 * b.u / b.depth +
                             w2 * c.u / c.depth) /
                            inverse_depth;
          const double tv = (w0 * a.v / a.depth + w1 * b.v / b.depth +
                             w2 * c.v / c.depth) /
                            inverse_depth;
          const int tx = std::clamp(static_cast<int>(tu * (texture->cols - 1)), 0,
                                    texture->cols - 1);
          const int ty = std::clamp(static_cast<int>(tv * (texture->rows - 1)), 0,
                                    texture->rows - 1);
          const cv::Vec4b texel = texture->at<cv::Vec4b>(ty, tx);
          if (texel[3] < defaults::kRendererTextureAlphaThreshold) continue;
          pixel = {texel[0], texel[1], texel[2]};
        }
        depth_row[x] = pixel_depth;
        colour_row[x] = pixel;
        semantic_row[x] = static_cast<std::uint8_t>(semantic_class);
        instance_row[x] = static_cast<std::int32_t>(instance_id);
      }
    }
  }

  void render_box(const SceneObject& object, const CameraTransform& transform,
                  std::uint32_t instance_override = std::numeric_limits<std::uint32_t>::max(),
                  SemanticClass semantic_override = SemanticClass::Obstacle) {
    const double c = std::cos(object.yaw_rad);
    const double s = std::sin(object.yaw_rad);
    const cv::Vec3d forward(c, s, 0.0);
    const cv::Vec3d left(-s, c, 0.0);
    const cv::Vec3d centre(object.position.x, object.position.y, object.base_z_m);
    const double half_width = object.width_m * 0.5;
    const double half_depth = object.depth_m * 0.5;
    std::array<cv::Vec3d, 8> vertices{};
    int cursor = 0;
    for (int z = 0; z < 2; ++z) {
      for (int longitudinal : {-1, 1}) {
        for (int lateral : {-1, 1}) {
          vertices[cursor++] = centre + forward * (longitudinal * half_depth) +
                               left * (lateral * half_width) +
                               cv::Vec3d(0, 0, z * object.height_m);
        }
      }
    }
    constexpr std::array<std::array<int, 4>, 6> faces{{
        {{0, 1, 3, 2}}, {{4, 6, 7, 5}}, {{0, 4, 5, 1}},
        {{2, 3, 7, 6}}, {{0, 2, 6, 4}}, {{1, 5, 7, 3}},
    }};
    const std::uint32_t instance =
        instance_override == std::numeric_limits<std::uint32_t>::max()
            ? object.instance_id
            : instance_override;
    const SemanticClass semantic = instance_override ==
                                           std::numeric_limits<std::uint32_t>::max()
                                       ? object.semantic_class
                                       : semantic_override;
    for (std::size_t face_index = 0; face_index < faces.size(); ++face_index) {
      const auto& face = faces[face_index];
      std::array<ProjectedVertex, 4> projected{};
      for (int index = 0; index < 4; ++index) {
        projected[index] = project_point(vertices[face[index]], camera, transform);
      }
      const double shade =
          defaults::kRendererBoxFaceShadeBase +
          defaults::kRendererBoxFaceShadeStep *
              static_cast<double>(face_index);
      const cv::Vec3b colour(
          cv::saturate_cast<std::uint8_t>(object.bgr[0] * shade),
          cv::saturate_cast<std::uint8_t>(object.bgr[1] * shade),
          cv::saturate_cast<std::uint8_t>(object.bgr[2] * shade));
      raster_triangle(projected[0], projected[1], projected[2], colour, semantic,
                      instance, nullptr);
      raster_triangle(projected[0], projected[2], projected[3], colour, semantic,
                      instance, nullptr);
    }
  }

  void render_stop_sign(const SceneObject& object,
                        const CameraTransform& transform) {
    SceneObject pole = object;
    pole.type = ObjectType::Box;
    pole.width_m = defaults::kRendererStopPoleWidthM;
    pole.depth_m = defaults::kRendererStopPoleDepthM;
    pole.height_m = object.height_m;
    pole.bgr = configured_object_bgr(defaults::kRendererStopPoleBgr);
    render_box(pole, transform, 0, SemanticClass::Obstacle);

    const cv::Vec3d centre(object.position.x, object.position.y,
                           object.base_z_m + object.height_m);
    const cv::Vec3d horizontal(-std::sin(object.yaw_rad),
                               std::cos(object.yaw_rad), 0.0);
    const cv::Vec3d vertical(0.0, 0.0, 1.0);
    constexpr int subdivisions = defaults::kRendererStopFaceSubdivisions;
    for (int row = 0; row < subdivisions; ++row) {
      for (int column = 0; column < subdivisions; ++column) {
        const double u0 = static_cast<double>(column) / subdivisions;
        const double u1 = static_cast<double>(column + 1) / subdivisions;
        const double v0 = static_cast<double>(row) / subdivisions;
        const double v1 = static_cast<double>(row + 1) / subdivisions;
        auto world = [&](double u, double v) {
          return centre + horizontal * ((u - 0.5) * object.width_m) +
                 vertical * ((0.5 - v) * object.width_m);
        };
        const ProjectedVertex a = project_point(world(u0, v0), camera, transform, u0, v0);
        const ProjectedVertex b = project_point(world(u1, v0), camera, transform, u1, v0);
        const ProjectedVertex c = project_point(world(u1, v1), camera, transform, u1, v1);
        const ProjectedVertex d = project_point(world(u0, v1), camera, transform, u0, v1);
        raster_triangle(a, b, c, cv::Vec3b(0, 0, 0), SemanticClass::StopSign,
                        object.instance_id, &stop_texture);
        raster_triangle(a, c, d, cv::Vec3b(0, 0, 0), SemanticClass::StopSign,
                        object.instance_id, &stop_texture);
      }
    }
  }

  FramePtr render(double timestamp_s) {
    render_bgr.create(camera.height, camera.width, CV_8UC3);
    render_semantic.create(camera.height, camera.width, CV_8UC1);
    render_instance.create(camera.height, camera.width, CV_32SC1);
    depth.create(camera.height, camera.width, CV_32FC1);
    const CameraTransform base_transform = camera_transform(vehicle, camera);
    cv::parallel_for_(cv::Range(0, camera.height), [&](const cv::Range& range) {
      for (int y = range.start; y < range.end; ++y) {
        VehicleState row_state = vehicle;
        if (camera.shutter == ShutterType::Rolling && camera.rolling_readout_s > 0.0) {
          const double offset =
              (static_cast<double>(y) / std::max(1, camera.height - 1) - 0.5) *
              camera.rolling_readout_s;
          row_state.pose.x += row_state.speed_mps * std::cos(row_state.pose.yaw) * offset;
          row_state.pose.y += row_state.speed_mps * std::sin(row_state.pose.yaw) * offset;
          row_state.pose.yaw +=
              row_state.speed_mps /
              std::max(scene.vehicle.wheelbase_m,
                       defaults::kRendererMinimumBicycleWheelbaseM) *
              std::tan(row_state.steering_rad) * offset;
        }
        const CameraTransform transform =
            camera.shutter == ShutterType::Rolling ? camera_transform(row_state, camera)
                                                   : base_transform;
        auto* bgr_row = render_bgr.ptr<cv::Vec3b>(y);
        auto* semantic_row = render_semantic.ptr<std::uint8_t>(y);
        auto* instance_row = render_instance.ptr<std::int32_t>(y);
        auto* depth_row = depth.ptr<float>(y);
        for (int x = 0; x < camera.width; ++x) {
          const cv::Vec3f ray_f = rays[static_cast<std::size_t>(y) * camera.width + x];
          const cv::Vec3d ray = transform.camera_to_world * cv::Vec3d(ray_f);
          if (ray[2] < defaults::kRendererGroundRayZThreshold) {
            const double distance = -transform.origin[2] / ray[2];
            const cv::Vec3d world = transform.origin + ray * distance;
            const int atlas_x = static_cast<int>((world[0] - atlas.min_x) * atlas.pixels_per_metre);
            const int atlas_y = static_cast<int>((atlas.max_y - world[1]) * atlas.pixels_per_metre);
            if (atlas_x >= 0 && atlas_x < atlas.bgr.cols && atlas_y >= 0 &&
                atlas_y < atlas.bgr.rows) {
              bgr_row[x] = atlas.bgr.at<cv::Vec3b>(atlas_y, atlas_x);
              semantic_row[x] = atlas.semantic.at<std::uint8_t>(atlas_y, atlas_x);
              instance_row[x] = atlas.instance.at<std::int32_t>(atlas_y, atlas_x);
              depth_row[x] = static_cast<float>(distance);
              continue;
            }
          }
          const double blend = static_cast<double>(y) / std::max(1, camera.height - 1);
          cv::Vec3b sky;
          for (std::size_t channel = 0;
               channel < defaults::kRendererSkyTopBgr.size(); ++channel) {
            sky[channel] = cv::saturate_cast<std::uint8_t>(
                defaults::kRendererSkyTopBgr[channel] +
                (defaults::kRendererSkyBottomBgr[channel] -
                 defaults::kRendererSkyTopBgr[channel]) *
                    blend);
          }
          bgr_row[x] = sky;
          semantic_row[x] = static_cast<std::uint8_t>(SemanticClass::Background);
          instance_row[x] = 0;
          depth_row[x] = std::numeric_limits<float>::infinity();
        }
      }
    });

    for (const SceneObject& object : scene.objects) {
      if (object.type == ObjectType::StopSign) {
        render_stop_sign(object, base_transform);
      } else {
        render_box(object, base_transform);
      }
    }

    auto frame = std::make_shared<Frame>();
    frame->frame_id = ++frame_id;
    frame->simulation_time_s = timestamp_s;
    frame->exposure_start_s = timestamp_s - camera.exposure_s * 0.5;
    frame->exposure_end_s = timestamp_s + camera.exposure_s * 0.5;
    frame->vehicle = vehicle;
    frame->camera = camera;
    frame->semantic = render_semantic.clone();
    frame->instance = render_instance.clone();

    cv::Mat i420;
    cv::cvtColor(render_bgr, i420, cv::COLOR_BGR2YUV_I420);
    frame->y_plane.create(camera.height, camera.width, CV_8UC1);
    frame->uv_plane.create(camera.height / 2, camera.width, CV_8UC1);
    const std::size_t y_bytes = static_cast<std::size_t>(camera.width) * camera.height;
    const std::size_t chroma_samples = y_bytes / 4;
    std::memcpy(frame->y_plane.data, i420.data, y_bytes);
    const std::uint8_t* u_plane = i420.data + y_bytes;
    const std::uint8_t* v_plane = u_plane + chroma_samples;
    for (int y = 0; y < camera.height / 2; ++y) {
      auto* output = frame->uv_plane.ptr<std::uint8_t>(y);
      for (int x = 0; x < camera.width / 2; ++x) {
        const std::size_t index = static_cast<std::size_t>(y) * (camera.width / 2) + x;
        output[2 * x] = u_plane[index];
        output[2 * x + 1] = v_plane[index];
      }
    }

    struct Bounds {
      int min_x{std::numeric_limits<int>::max()};
      int min_y{std::numeric_limits<int>::max()};
      int max_x{-1};
      int max_y{-1};
      int pixels{0};
    };
    std::unordered_map<std::uint32_t, Bounds> bounds;
    for (int y = 0; y < frame->instance.rows; ++y) {
      const auto* row = frame->instance.ptr<std::int32_t>(y);
      for (int x = 0; x < frame->instance.cols; ++x) {
        const std::uint32_t identifier = static_cast<std::uint32_t>(row[x]);
        if (identifier == 0) continue;
        Bounds& value = bounds[identifier];
        value.min_x = std::min(value.min_x, x);
        value.min_y = std::min(value.min_y, y);
        value.max_x = std::max(value.max_x, x);
        value.max_y = std::max(value.max_y, y);
        ++value.pixels;
      }
    }
    for (const SceneObject& object : scene.objects) {
      const auto iterator = bounds.find(object.instance_id);
      if (iterator == bounds.end()) continue;
      const Bounds& value = iterator->second;
      const int area = std::max(1, (value.max_x - value.min_x + 1) *
                                       (value.max_y - value.min_y + 1));
      const double fill =
          object.type == ObjectType::StopSign
              ? defaults::kRendererStopSignVisibilityFillFraction
              : 1.0;
      Detection detection;
      detection.class_id = static_cast<int>(object.semantic_class);
      detection.instance_id = object.instance_id;
      detection.bbox_xyxy = {value.min_x, value.min_y, value.max_x + 1,
                             value.max_y + 1};
      detection.visibility = std::min(1.0, value.pixels / (area * fill));
      detection.range_m = std::hypot(object.position.x - vehicle.pose.x,
                                     object.position.y - vehicle.pose.y);
      detection.relative_yaw_rad = wrap_angle(object.yaw_rad - vehicle.pose.yaw);
      frame->detections.push_back(detection);
    }
    return frame;
  }

  Scene scene;
  CameraProfile camera;
  VehicleState vehicle;
  Atlas atlas;
  cv::Mat stop_texture;
  std::vector<cv::Vec3f> rays;
  cv::Mat render_bgr;
  cv::Mat render_semantic;
  cv::Mat render_instance;
  cv::Mat depth;
  double simulation_time{0.0};
  double next_camera_time{0.0};
  std::uint64_t scheduled_frame_index{1};
  std::uint64_t frame_id{0};
};

Simulator::Simulator(Scene scene, CameraProfile camera)
    : impl_(std::make_unique<Impl>(std::move(scene), std::move(camera))) {}

Simulator::~Simulator() = default;
Simulator::Simulator(Simulator&&) noexcept = default;
Simulator& Simulator::operator=(Simulator&&) noexcept = default;

void Simulator::reset() { impl_->reset(); }

void Simulator::reset(Scene scene, CameraProfile camera) {
  impl_ = std::make_unique<Impl>(std::move(scene), std::move(camera));
}

void Simulator::set_vehicle_state(VehicleState state) {
  if (!std::isfinite(state.pose.x) || !std::isfinite(state.pose.y) ||
      !std::isfinite(state.pose.yaw) || !std::isfinite(state.speed_mps) ||
      !std::isfinite(state.steering_rad)) {
    throw std::invalid_argument("vehicle state must be finite");
  }
  state.steering_rad =
      clamp_value(state.steering_rad, -impl_->scene.vehicle.max_steering_rad,
                  impl_->scene.vehicle.max_steering_rad);
  impl_->vehicle = state;
}

FrameBatch Simulator::advance(const VehicleCommand& command, double dt_s) {
  if (dt_s < 0.0) throw std::invalid_argument("simulation dt must not be negative");
  FrameBatch frames;
  double remaining = dt_s;
  while (impl_->next_camera_time <=
         impl_->simulation_time + remaining +
             defaults::kRendererScheduleComparisonEpsilonS) {
    const double substep = std::max(0.0, impl_->next_camera_time - impl_->simulation_time);
    step_bicycle(impl_->scene.vehicle, impl_->vehicle, command, substep);
    impl_->simulation_time += substep;
    remaining = std::max(0.0, remaining - substep);
    frames.push_back(impl_->render(impl_->next_camera_time));
    ++impl_->scheduled_frame_index;
    impl_->next_camera_time =
        impl_->scheduled_time(impl_->scheduled_frame_index);
  }
  step_bicycle(impl_->scene.vehicle, impl_->vehicle, command, remaining);
  impl_->simulation_time += remaining;
  return frames;
}

FramePtr Simulator::render_now() { return impl_->render(impl_->simulation_time); }
const Scene& Simulator::scene() const { return impl_->scene; }
const CameraProfile& Simulator::camera() const { return impl_->camera; }
const VehicleState& Simulator::vehicle_state() const { return impl_->vehicle; }
double Simulator::simulation_time_s() const { return impl_->simulation_time; }

cv::Mat frame_to_bgr(const Frame& frame) {
  if (frame.y_plane.empty() || frame.uv_plane.empty()) {
    throw std::invalid_argument("frame does not contain NV12 planes");
  }
  const int width = frame.y_plane.cols;
  const int height = frame.y_plane.rows;
  cv::Mat i420(height * 3 / 2, width, CV_8UC1);
  const std::size_t y_bytes = static_cast<std::size_t>(width) * height;
  const std::size_t chroma_samples = y_bytes / 4;
  std::memcpy(i420.data, frame.y_plane.data, y_bytes);
  std::uint8_t* u_plane = i420.data + y_bytes;
  std::uint8_t* v_plane = u_plane + chroma_samples;
  for (int y = 0; y < height / 2; ++y) {
    const auto* input = frame.uv_plane.ptr<std::uint8_t>(y);
    for (int x = 0; x < width / 2; ++x) {
      const std::size_t index = static_cast<std::size_t>(y) * (width / 2) + x;
      u_plane[index] = input[2 * x];
      v_plane[index] = input[2 * x + 1];
    }
  }
  cv::Mat bgr;
  cv::cvtColor(i420, bgr, cv::COLOR_YUV2BGR_I420);
  return bgr;
}

}  // namespace jetracer::sim
