#include "color_lane_native.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core/version.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace {

int integer_value(const cv::FileNode& parent, const char* name) {
  const cv::FileNode value = parent[name];
  if (value.empty() || !value.isInt()) {
    throw std::runtime_error(std::string("missing integer setting: ") + name);
  }
  return static_cast<int>(value);
}

double real_value(const cv::FileNode& parent, const char* name) {
  const cv::FileNode value = parent[name];
  if (value.empty() || (!value.isReal() && !value.isInt())) {
    throw std::runtime_error(std::string("missing numeric setting: ") + name);
  }
  return static_cast<double>(value);
}

cv::Scalar hsv_scalar(const cv::FileNode& value, const char* name) {
  if (!value.isSeq() || value.size() != 3) {
    throw std::runtime_error(std::string("invalid HSV setting: ") + name);
  }
  return cv::Scalar(static_cast<int>(value[0]), static_cast<int>(value[1]),
                    static_cast<int>(value[2]));
}

std::vector<cv::Point2f> normalized_points(const cv::FileNode& value,
                                           const char* name) {
  if (!value.isSeq()) {
    throw std::runtime_error(std::string("invalid point sequence: ") + name);
  }
  std::vector<cv::Point2f> points;
  for (const cv::FileNode& point : value) {
    if (!point.isSeq() || point.size() != 2) {
      throw std::runtime_error(std::string("invalid point in: ") + name);
    }
    points.emplace_back(static_cast<float>(static_cast<double>(point[0])),
                        static_cast<float>(static_cast<double>(point[1])));
  }
  return points;
}

double quantile(std::vector<double> values, double probability) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double position = probability * static_cast<double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

void validate_config(const ColorLaneNativeConfig& config) {
  if (config.profile_id.empty() || config.processing_width <= 0 ||
      config.processing_height <= 0 || config.hsv_ranges.empty()) {
    throw std::runtime_error("invalid color-lane identity or dimensions");
  }
  if (config.roi_top_fraction < 0.0 || config.roi_top_fraction >= 1.0) {
    throw std::runtime_error("color-lane ROI fraction must be in [0, 1)");
  }
  for (const int kernel : {config.morphology_close_kernel,
                           config.morphology_open_kernel}) {
    if (kernel < 0 || (kernel > 0 && kernel % 2 == 0)) {
      throw std::runtime_error("morphology kernels must be zero or odd");
    }
  }
  if (config.minimum_run_width_px <= 0 ||
      config.minimum_lane_width_px <= 0 || config.polynomial_degree < 1 ||
      config.fit_iterations <= 0 ||
      config.minimum_fit_points <= config.polynomial_degree ||
      config.residual_floor_px <= 0.0 ||
      config.residual_quantile <= 0.0 || config.residual_quantile >= 1.0 ||
      config.residual_multiplier <= 0.0 || config.path_sample_count < 2 ||
      config.road_class_id == 0) {
    throw std::runtime_error("invalid color-lane fitting parameters");
  }
  for (const ColorLaneHsvRange& range : config.hsv_ranges) {
    const double limits[3] = {179.0, 255.0, 255.0};
    for (int channel = 0; channel < 3; ++channel) {
      if (range.lower[channel] < 0.0 ||
          range.lower[channel] > range.upper[channel] ||
          range.upper[channel] > limits[channel]) {
        throw std::runtime_error("invalid color-lane HSV range");
      }
    }
  }
  if (config.birdseye_enabled &&
      (config.birdseye_source_points.size() != 4 ||
       config.birdseye_destination_points.size() != 4)) {
    throw std::runtime_error("bird's-eye transform requires four point pairs");
  }
}

}  // namespace

ColorLaneNativeConfig load_color_lane_native_config(const std::string& path) {
  cv::FileStorage storage(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  if (!storage.isOpened()) {
    throw std::runtime_error("cannot open color-lane profile: " + path);
  }
  if (integer_value(storage.root(), "schema_version") != 1) {
    throw std::runtime_error("unsupported color-lane profile schema");
  }
  ColorLaneNativeConfig config;
  storage["profile_id"] >> config.profile_id;
  config.processing_width = integer_value(storage.root(), "processing_width");
  config.processing_height = integer_value(storage.root(), "processing_height");
  const cv::FileNode raw_ranges = storage["hsv_ranges"];
  if (!raw_ranges.isSeq()) {
    throw std::runtime_error("color-lane HSV ranges must be a sequence");
  }
  for (const cv::FileNode& raw_range : raw_ranges) {
    config.hsv_ranges.push_back(ColorLaneHsvRange{
        hsv_scalar(raw_range["lower"], "lower"),
        hsv_scalar(raw_range["upper"], "upper")});
  }
  config.roi_top_fraction = real_value(storage.root(), "roi_top_fraction");
  config.morphology_close_kernel =
      integer_value(storage.root(), "morphology_close_kernel");
  config.morphology_open_kernel =
      integer_value(storage.root(), "morphology_open_kernel");
  config.minimum_run_width_px =
      integer_value(storage.root(), "minimum_run_width_px");
  config.minimum_lane_width_px =
      integer_value(storage.root(), "minimum_lane_width_px");
  config.polynomial_degree = integer_value(storage.root(), "polynomial_degree");
  config.fit_iterations = integer_value(storage.root(), "fit_iterations");
  config.minimum_fit_points =
      integer_value(storage.root(), "minimum_fit_points");
  config.residual_floor_px = real_value(storage.root(), "residual_floor_px");
  config.residual_quantile = real_value(storage.root(), "residual_quantile");
  config.residual_multiplier = real_value(storage.root(), "residual_multiplier");
  config.path_sample_count = integer_value(storage.root(), "path_sample_count");
  const int road_class_id = integer_value(storage.root(), "road_class_id");
  if (road_class_id < 0 || road_class_id > 255) {
    throw std::runtime_error("road class ID is outside uint8 range");
  }
  config.road_class_id = static_cast<std::uint8_t>(road_class_id);
  const cv::FileNode birdseye = storage["birdseye"];
  if (!birdseye.isMap()) {
    throw std::runtime_error("color-lane profile requires birdseye settings");
  }
  config.birdseye_enabled = integer_value(birdseye, "enabled") != 0;
  config.birdseye_source_points =
      normalized_points(birdseye["source_points"], "source_points");
  config.birdseye_destination_points =
      normalized_points(birdseye["destination_points"], "destination_points");
  validate_config(config);
  return config;
}

ColorLaneNativeProcessor::ColorLaneNativeProcessor(ColorLaneNativeConfig config)
    : config_(std::move(config)) {
  validate_config(config_);
  if (config_.morphology_close_kernel > 0) {
    close_kernel_ = cv::Mat::ones(config_.morphology_close_kernel,
                                  config_.morphology_close_kernel, CV_8UC1);
  }
  if (config_.morphology_open_kernel > 0) {
    open_kernel_ = cv::Mat::ones(config_.morphology_open_kernel,
                                 config_.morphology_open_kernel, CV_8UC1);
  }
  row_basis_ = cv::Mat(config_.processing_height,
                       config_.polynomial_degree + 1, CV_64F);
  for (int row = 0; row < config_.processing_height; ++row) {
    const double normalized =
        static_cast<double>(row) / config_.processing_height;
    for (int column = 0; column <= config_.polynomial_degree; ++column) {
      row_basis_.at<double>(row, column) =
          std::pow(normalized, config_.polynomial_degree - column);
    }
  }
  if (config_.birdseye_enabled) {
    std::vector<cv::Point2f> source = config_.birdseye_source_points;
    std::vector<cv::Point2f> destination = config_.birdseye_destination_points;
    for (cv::Point2f& point : source) {
      point.x *= static_cast<float>(config_.processing_width - 1);
      point.y *= static_cast<float>(config_.processing_height - 1);
    }
    for (cv::Point2f& point : destination) {
      point.x *= static_cast<float>(config_.processing_width - 1);
      point.y *= static_cast<float>(config_.processing_height - 1);
    }
    homography_ = cv::getPerspectiveTransform(source, destination);
    inverse_homography_ = cv::getPerspectiveTransform(destination, source);
  }
}

ColorLaneNativeResult ColorLaneNativeProcessor::process(
    const cv::Mat& image_bgr) const {
  if (image_bgr.empty() ||
      (image_bgr.type() != CV_8UC3 && image_bgr.type() != CV_8UC4)) {
    throw std::invalid_argument(
        "color-lane input must be a non-empty BGR8 or BGRx8 image");
  }
  cv::Mat working;
  const cv::Size processing_size(config_.processing_width,
                                 config_.processing_height);
  if (image_bgr.size() == processing_size && image_bgr.type() == CV_8UC3) {
    working = image_bgr;
  } else {
    cv::resize(image_bgr, working, processing_size, 0.0, 0.0, cv::INTER_AREA);
  }
  if (working.type() == CV_8UC4) {
    cv::cvtColor(working, working, cv::COLOR_BGRA2BGR);
  }
  if (!homography_.empty()) {
    cv::warpPerspective(working, working, homography_, working.size());
  }
  cv::Mat hsv;
  cv::cvtColor(working, hsv, cv::COLOR_BGR2HSV);
  cv::Mat threshold = cv::Mat::zeros(hsv.size(), CV_8UC1);
  for (const ColorLaneHsvRange& range : config_.hsv_ranges) {
    cv::Mat selected;
    cv::inRange(hsv, range.lower, range.upper, selected);
    cv::bitwise_or(threshold, selected, threshold);
  }
  threshold.rowRange(0, roi_top_row()).setTo(0);
  if (!close_kernel_.empty()) {
    cv::morphologyEx(threshold, threshold, cv::MORPH_CLOSE, close_kernel_);
  }
  if (!open_kernel_.empty()) {
    cv::morphologyEx(threshold, threshold, cv::MORPH_OPEN, open_kernel_);
  }

  std::vector<double> rows;
  std::vector<double> left_edges;
  std::vector<double> right_edges;
  rows.reserve(static_cast<std::size_t>(config_.processing_height));
  left_edges.reserve(rows.capacity());
  right_edges.reserve(rows.capacity());
  for (int row = roi_top_row(); row < config_.processing_height; ++row) {
    const std::uint8_t* pixels = threshold.ptr<std::uint8_t>(row);
    int valid_runs = 0;
    int left = 0;
    int right = 0;
    int column = 0;
    while (column < config_.processing_width) {
      while (column < config_.processing_width && pixels[column] == 0) ++column;
      if (column >= config_.processing_width) break;
      const int start = column;
      while (column < config_.processing_width && pixels[column] != 0) ++column;
      const int end = column - 1;
      if (end - start + 1 < config_.minimum_run_width_px) continue;
      if (valid_runs == 0) left = start;
      right = end;
      ++valid_runs;
    }
    if (valid_runs < 2 || right - left < config_.minimum_lane_width_px) continue;
    rows.push_back(static_cast<double>(row));
    left_edges.push_back(static_cast<double>(left));
    right_edges.push_back(static_cast<double>(right));
  }

  ColorLaneNativeResult result;
  result.labels = cv::Mat::zeros(config_.processing_height,
                                 config_.processing_width, CV_8UC1);
  result.observed_rows = static_cast<int>(rows.size());
  result.birdseye_applied = !homography_.empty();
  const PolynomialFit left_fit = robust_fit(rows, left_edges);
  const PolynomialFit right_fit = robust_fit(rows, right_edges);
  if (left_fit.coefficients.empty() || right_fit.coefficients.empty()) {
    return result;
  }

  int valid_rows = 0;
  for (int row = roi_top_row(); row < config_.processing_height; ++row) {
    const double normalized_row =
        static_cast<double>(row) / config_.processing_height;
    const int left = std::max(
        0, static_cast<int>(std::lround(evaluate(left_fit.coefficients,
                                                 normalized_row) *
                                        config_.processing_width)));
    const int right = std::min(
        config_.processing_width - 1,
        static_cast<int>(std::lround(evaluate(right_fit.coefficients,
                                              normalized_row) *
                                     config_.processing_width)));
    if (right - left < config_.minimum_lane_width_px) continue;
    std::uint8_t* output = result.labels.ptr<std::uint8_t>(row);
    std::fill(output + left, output + right + 1, config_.road_class_id);
    ++valid_rows;
  }
  if (!inverse_homography_.empty()) {
    cv::warpPerspective(result.labels, result.labels, inverse_homography_,
                        result.labels.size(), cv::INTER_NEAREST);
  }

  const int available_rows = config_.processing_height - roi_top_row();
  const auto inlier_fraction = [](const std::vector<std::uint8_t>& inliers) {
    if (inliers.empty()) return 0.0;
    return static_cast<double>(
               std::count(inliers.begin(), inliers.end(), std::uint8_t{1})) /
           static_cast<double>(inliers.size());
  };
  result.left_inlier_fraction = inlier_fraction(left_fit.inliers);
  result.right_inlier_fraction = inlier_fraction(right_fit.inliers);
  const double observation_coverage = std::min(
      1.0, static_cast<double>(rows.size()) / std::max(1, available_rows));
  const double valid_coverage =
      static_cast<double>(valid_rows) / std::max(1, available_rows);
  result.confidence =
      std::clamp(observation_coverage *
                     std::min(result.left_inlier_fraction,
                              result.right_inlier_fraction) *
                     valid_coverage,
                 0.0, 1.0);

  result.center_path_normalized.reserve(
      static_cast<std::size_t>(config_.path_sample_count));
  for (int index = 0; index < config_.path_sample_count; ++index) {
    const double fraction = static_cast<double>(index) /
                            static_cast<double>(config_.path_sample_count - 1);
    const double row = roi_top_row() +
                       fraction * (config_.processing_height - 1 - roi_top_row());
    const double normalized_row = row / config_.processing_height;
    const double center =
        0.5 * (evaluate(left_fit.coefficients, normalized_row) +
               evaluate(right_fit.coefficients, normalized_row));
    if (center >= 0.0 && center <= 1.0) {
      result.center_path_normalized.emplace_back(
          static_cast<float>(center), static_cast<float>(normalized_row));
    }
  }
  return result;
}

ColorLaneNativeProcessor::PolynomialFit ColorLaneNativeProcessor::robust_fit(
    const std::vector<double>& rows, const std::vector<double>& values) const {
  PolynomialFit result;
  if (rows.size() != values.size() ||
      rows.size() < static_cast<std::size_t>(config_.minimum_fit_points)) {
    return result;
  }
  result.inliers.assign(rows.size(), std::uint8_t{1});
  for (int iteration = 0; iteration < config_.fit_iterations; ++iteration) {
    const int count = static_cast<int>(std::count(
        result.inliers.begin(), result.inliers.end(), std::uint8_t{1}));
    if (count < config_.minimum_fit_points) return PolynomialFit{};
    const int coefficient_count = config_.polynomial_degree + 1;
    cv::Mat normal = cv::Mat::zeros(coefficient_count, coefficient_count, CV_64F);
    cv::Mat right_hand_side = cv::Mat::zeros(coefficient_count, 1, CV_64F);
    for (std::size_t index = 0; index < rows.size(); ++index) {
      if (!result.inliers[index]) continue;
      const int source_row = static_cast<int>(std::lround(rows[index]));
      const double target = values[index] / config_.processing_width;
      const double* basis = row_basis_.ptr<double>(source_row);
      for (int row = 0; row < coefficient_count; ++row) {
        right_hand_side.at<double>(row, 0) += basis[row] * target;
        for (int column = 0; column < coefficient_count; ++column) {
          normal.at<double>(row, column) += basis[row] * basis[column];
        }
      }
    }
    cv::Mat coefficients;
    if (!cv::solve(normal, right_hand_side, coefficients, cv::DECOMP_CHOLESKY)) {
      return PolynomialFit{};
    }
    result.coefficients.resize(
        static_cast<std::size_t>(config_.polynomial_degree + 1));
    for (int index = 0; index <= config_.polynomial_degree; ++index) {
      result.coefficients[static_cast<std::size_t>(index)] =
          coefficients.at<double>(index, 0);
    }
    std::vector<double> residuals(rows.size());
    std::vector<double> inlier_residuals;
    inlier_residuals.reserve(static_cast<std::size_t>(count));
    for (std::size_t index = 0; index < rows.size(); ++index) {
      residuals[index] =
          std::abs(evaluate(result.coefficients,
                            rows[index] / config_.processing_height) -
                   values[index] / config_.processing_width) *
          config_.processing_width;
      if (result.inliers[index]) inlier_residuals.push_back(residuals[index]);
    }
    const double threshold =
        std::max(config_.residual_floor_px,
                 quantile(std::move(inlier_residuals),
                          config_.residual_quantile) *
                     config_.residual_multiplier);
    for (std::size_t index = 0; index < residuals.size(); ++index) {
      result.inliers[index] = residuals[index] < threshold ? 1 : 0;
    }
  }
  return result;
}

double ColorLaneNativeProcessor::evaluate(
    const std::vector<double>& coefficients, double value) const {
  double result = 0.0;
  for (const double coefficient : coefficients) {
    result = result * value + coefficient;
  }
  return result;
}

int ColorLaneNativeProcessor::roi_top_row() const {
  return static_cast<int>(
      std::lround(config_.roi_top_fraction * config_.processing_height));
}
