#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace jetracer::sim {

class CoreMLSegmentationSession final {
 public:
  CoreMLSegmentationSession(
      const std::string& model_path, const std::string& input_name,
      const std::string& output_name, int input_width, int input_height,
      int output_width, int output_height,
      std::vector<int> source_road_class_ids, std::uint8_t road_class_id,
      float input_scale, std::vector<float> mean_rgb,
      std::vector<float> std_rgb, const std::string& compute_units);
  ~CoreMLSegmentationSession();

  CoreMLSegmentationSession(const CoreMLSegmentationSession&) = delete;
  CoreMLSegmentationSession& operator=(const CoreMLSegmentationSession&) =
      delete;
  CoreMLSegmentationSession(CoreMLSegmentationSession&&) noexcept;
  CoreMLSegmentationSession& operator=(CoreMLSegmentationSession&&) noexcept;

  [[nodiscard]] std::vector<std::uint8_t> infer(
      const std::uint8_t* image_bgr, int image_width, int image_height);
  [[nodiscard]] int output_width() const noexcept;
  [[nodiscard]] int output_height() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jetracer::sim
