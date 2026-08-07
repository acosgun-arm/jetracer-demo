#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

struct ColorLaneHsvRange {
  cv::Scalar lower;
  cv::Scalar upper;
};

struct ColorLaneNativeConfig {
  std::string profile_id;
  int processing_width = 0;
  int processing_height = 0;
  std::vector<ColorLaneHsvRange> hsv_ranges;
  double roi_top_fraction = 0.0;
  int morphology_close_kernel = 0;
  int morphology_open_kernel = 0;
  int minimum_run_width_px = 0;
  int minimum_lane_width_px = 0;
  int polynomial_degree = 0;
  int fit_iterations = 0;
  int minimum_fit_points = 0;
  double residual_floor_px = 0.0;
  double residual_quantile = 0.0;
  double residual_multiplier = 0.0;
  int path_sample_count = 0;
  std::uint8_t road_class_id = 0;
  bool birdseye_enabled = false;
  std::vector<cv::Point2f> birdseye_source_points;
  std::vector<cv::Point2f> birdseye_destination_points;
};

ColorLaneNativeConfig load_color_lane_native_config(const std::string& path);

struct ColorLaneNativeResult {
  cv::Mat labels;
  double confidence = 0.0;
  int observed_rows = 0;
  double left_inlier_fraction = 0.0;
  double right_inlier_fraction = 0.0;
  std::vector<cv::Point2f> center_path_normalized;
  bool birdseye_applied = false;
};

class ColorLaneNativeProcessor {
 public:
  explicit ColorLaneNativeProcessor(ColorLaneNativeConfig config);

  const ColorLaneNativeConfig& config() const { return config_; }
  ColorLaneNativeResult process(const cv::Mat& image_bgr) const;

 private:
  struct PolynomialFit {
    std::vector<double> coefficients;
    std::vector<std::uint8_t> inliers;
  };

  PolynomialFit robust_fit(const std::vector<double>& rows,
                           const std::vector<double>& values) const;
  double evaluate(const std::vector<double>& coefficients,
                  double value) const;
  int roi_top_row() const;

  ColorLaneNativeConfig config_;
  cv::Mat homography_;
  cv::Mat inverse_homography_;
  cv::Mat close_kernel_;
  cv::Mat open_kernel_;
  cv::Mat row_basis_;
};

int benchmark_color_lane_image(const std::string& profile_path,
                               const std::string& image_path,
                               int iterations, int warmup_iterations);
