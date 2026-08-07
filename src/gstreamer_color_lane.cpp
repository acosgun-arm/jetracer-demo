#include "gstreamer_color_lane.hpp"

#include "color_lane_native.hpp"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

void print_pipeline_error(GstElement* pipeline, const char* fallback) {
  GstBus* bus = gst_element_get_bus(pipeline);
  GstMessage* message = gst_bus_pop_filtered(
      bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  if (message != nullptr && GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
    GError* error = nullptr;
    gchar* debug = nullptr;
    gst_message_parse_error(message, &error, &debug);
    std::cerr << (error != nullptr ? error->message : fallback) << "\n";
    if (error != nullptr) g_error_free(error);
    g_free(debug);
  } else {
    std::cerr << fallback << "\n";
  }
  if (message != nullptr) gst_message_unref(message);
  gst_object_unref(bus);
}

double percentile(const std::vector<double>& sorted, double probability) {
  if (sorted.empty()) return 0.0;
  const double position = probability * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

}  // namespace

int benchmark_gstreamer_color_lane(
    const GstreamerColorLaneBenchmarkConfig& config) {
  if (config.profile_path.empty() || config.device.empty() ||
      config.input_width == 0 || config.input_height == 0 || config.fps == 0 ||
      config.flip_method < 0 || config.flip_method > 7 ||
      !std::isfinite(config.duration_s) || config.duration_s <= 0.0 ||
      config.startup_timeout_ms <= 0 || config.frame_timeout_ms <= 0) {
    std::cerr << "invalid GStreamer color-lane benchmark configuration\n";
    return EXIT_FAILURE;
  }

  try {
    ColorLaneNativeProcessor processor(
        load_color_lane_native_config(config.profile_path));
    gst_init(nullptr, nullptr);
    std::ostringstream description;
    description << "v4l2src device=" << config.device
                << " io-mode=2 ! image/jpeg,width=" << config.input_width
                << ",height=" << config.input_height << ",framerate="
                << config.fps << "/1 ! nvjpegdec ! nvvidconv flip-method="
                << config.flip_method
                << " ! video/x-raw(memory:NVMM),format=RGBA,width="
                << processor.config().processing_width << ",height="
                << processor.config().processing_height
                << " ! nvvidconv ! video/x-raw,format=BGRx"
                   " ! videoconvert ! video/x-raw,format=BGR"
                   " ! appsink name=color_lane_sink max-buffers=1 drop=true "
                   "sync=false";

    GError* parse_error = nullptr;
    GstElement* pipeline =
        gst_parse_launch(description.str().c_str(), &parse_error);
    if (pipeline == nullptr || parse_error != nullptr) {
      const std::string message =
          parse_error != nullptr ? parse_error->message : "unknown error";
      if (parse_error != nullptr) g_error_free(parse_error);
      if (pipeline != nullptr) gst_object_unref(pipeline);
      throw std::runtime_error("cannot create color-lane pipeline: " + message);
    }
    GstElement* sink_element =
        gst_bin_get_by_name(GST_BIN(pipeline), "color_lane_sink");
    if (sink_element == nullptr) {
      gst_object_unref(pipeline);
      throw std::runtime_error("color-lane pipeline has no appsink");
    }
    GstAppSink* sink = GST_APP_SINK(sink_element);
    const auto pipeline_started_at = std::chrono::steady_clock::now();
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) ==
        GST_STATE_CHANGE_FAILURE) {
      print_pipeline_error(pipeline, "cannot start color-lane pipeline");
      gst_object_unref(sink_element);
      gst_object_unref(pipeline);
      return EXIT_FAILURE;
    }

    std::uint64_t frames = 0;
    std::uint64_t buffer_offset_gaps = 0;
    std::uint64_t fit_failures = 0;
    guint64 previous_offset = GST_BUFFER_OFFSET_NONE;
    double confidence_sum = 0.0;
    double minimum_confidence = std::numeric_limits<double>::infinity();
    double observed_rows_sum = 0.0;
    std::vector<double> latencies_ms;
    std::chrono::steady_clock::time_point first_received_at{};
    std::chrono::steady_clock::time_point last_received_at{};

    while (true) {
      const int timeout_ms =
          frames == 0 ? config.startup_timeout_ms : config.frame_timeout_ms;
      GstSample* sample = gst_app_sink_try_pull_sample(
          sink, static_cast<GstClockTime>(timeout_ms) * GST_MSECOND);
      if (sample == nullptr) {
        print_pipeline_error(pipeline, "color-lane frame timeout");
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        return EXIT_FAILURE;
      }
      const auto received_at = std::chrono::steady_clock::now();
      GstBuffer* buffer = gst_sample_get_buffer(sample);
      GstCaps* caps = gst_sample_get_caps(sample);
      GstVideoInfo video_info;
      gst_video_info_init(&video_info);
      GstVideoFrame video_frame;
      if (caps == nullptr || !gst_video_info_from_caps(&video_info, caps) ||
          GST_VIDEO_INFO_FORMAT(&video_info) != GST_VIDEO_FORMAT_BGR ||
          !gst_video_frame_map(&video_frame, &video_info, buffer,
                               GST_MAP_READ)) {
        gst_sample_unref(sample);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(sink_element);
        gst_object_unref(pipeline);
        throw std::runtime_error("cannot map color-lane BGRx frame");
      }
      const int width = GST_VIDEO_INFO_WIDTH(&video_info);
      const int height = GST_VIDEO_INFO_HEIGHT(&video_info);
      const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&video_frame, 0);
      cv::Mat frame(height, width, CV_8UC3,
                    GST_VIDEO_FRAME_PLANE_DATA(&video_frame, 0),
                    static_cast<std::size_t>(stride));
      const auto inference_started_at = std::chrono::steady_clock::now();
      const ColorLaneNativeResult result = processor.process(frame);
      const auto inference_completed_at = std::chrono::steady_clock::now();
      latencies_ms.push_back(std::chrono::duration<double, std::milli>(
                                 inference_completed_at - inference_started_at)
                                 .count());
      confidence_sum += result.confidence;
      minimum_confidence = std::min(minimum_confidence, result.confidence);
      observed_rows_sum += result.observed_rows;
      if (result.center_path_normalized.empty()) ++fit_failures;
      const guint64 offset = GST_BUFFER_OFFSET(buffer);
      if (previous_offset != GST_BUFFER_OFFSET_NONE &&
          offset != GST_BUFFER_OFFSET_NONE && offset > previous_offset + 1U) {
        buffer_offset_gaps += offset - previous_offset - 1U;
      }
      previous_offset = offset;
      gst_video_frame_unmap(&video_frame);
      gst_sample_unref(sample);
      if (frames == 0) first_received_at = received_at;
      last_received_at = received_at;
      ++frames;
      if (std::chrono::duration<double>(received_at - first_received_at).count() >=
          config.duration_s) {
        break;
      }
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(sink_element);
    gst_object_unref(pipeline);
    const double elapsed_s =
        std::chrono::duration<double>(last_received_at - first_received_at)
            .count();
    const double delivered_fps =
        frames > 1 && elapsed_s > 0.0 ? (frames - 1) / elapsed_s : 0.0;
    const double startup_s =
        std::chrono::duration<double>(first_received_at - pipeline_started_at)
            .count();
    std::sort(latencies_ms.begin(), latencies_ms.end());
    const double mean_latency_ms =
        std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
        latencies_ms.size();
    std::cout << std::fixed << std::setprecision(6)
              << "{\n"
              << "  \"mode\": \"gstreamer_native_color_lane_benchmark\",\n"
              << "  \"actuators_accessed\": false,\n"
              << "  \"profile_id\": \"" << processor.config().profile_id
              << "\",\n"
              << "  \"decoder\": \"nvjpegdec\",\n"
              << "  \"rotation\": \"nvvidconv\",\n"
              << "  \"cpu_format\": \"BGR\",\n"
              << "  \"requested_fps\": " << config.fps << ",\n"
              << "  \"duration_s\": " << elapsed_s << ",\n"
              << "  \"startup_s\": " << startup_s << ",\n"
              << "  \"frames\": " << frames << ",\n"
              << "  \"delivered_fps\": " << delivered_fps << ",\n"
              << "  \"buffer_offset_gaps\": " << buffer_offset_gaps << ",\n"
              << "  \"mean_inference_latency_ms\": " << mean_latency_ms
              << ",\n"
              << "  \"p50_inference_latency_ms\": "
              << percentile(latencies_ms, 0.50) << ",\n"
              << "  \"p95_inference_latency_ms\": "
              << percentile(latencies_ms, 0.95) << ",\n"
              << "  \"p99_inference_latency_ms\": "
              << percentile(latencies_ms, 0.99) << ",\n"
              << "  \"maximum_inference_latency_ms\": "
              << latencies_ms.back() << ",\n"
              << "  \"inference_capacity_fps\": "
              << 1000.0 / mean_latency_ms << ",\n"
              << "  \"fit_failures\": " << fit_failures << ",\n"
              << "  \"mean_confidence\": " << confidence_sum / frames << ",\n"
              << "  \"minimum_confidence\": " << minimum_confidence << ",\n"
              << "  \"mean_observed_rows\": " << observed_rows_sum / frames
              << "\n"
              << "}\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "color-lane camera benchmark failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
