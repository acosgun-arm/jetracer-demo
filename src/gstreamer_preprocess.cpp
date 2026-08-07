#include "gstreamer_preprocess.hpp"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

void print_gstreamer_error(GstElement* pipeline, const char* fallback) {
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

}  // namespace

int benchmark_gstreamer_preprocess(
    const GstreamerPreprocessBenchmarkConfig& config) {
  if (config.device.empty() || config.input_width == 0 ||
      config.input_height == 0 || config.fps == 0 ||
      config.output_width == 0 || config.output_height == 0 ||
      config.flip_method < 0 || config.flip_method > 7 ||
      !std::isfinite(config.duration_s) || config.duration_s <= 0.0 ||
      config.startup_timeout_ms <= 0 || config.frame_timeout_ms <= 0) {
    std::cerr << "invalid GStreamer preprocess benchmark configuration\n";
    return EXIT_FAILURE;
  }

  gst_init(nullptr, nullptr);
  std::ostringstream description;
  description << "v4l2src device=" << config.device
              << " io-mode=2 ! image/jpeg,width=" << config.input_width
              << ",height=" << config.input_height << ",framerate="
              << config.fps << "/1 ! nvjpegdec ! nvvidconv flip-method="
              << config.flip_method
              << " ! video/x-raw(memory:NVMM),format=RGBA,width="
              << config.output_width << ",height=" << config.output_height
              << " ! appsink name=benchmark_sink max-buffers=1 drop=true "
                 "sync=false";

  GError* parse_error = nullptr;
  GstElement* pipeline = gst_parse_launch(description.str().c_str(), &parse_error);
  if (pipeline == nullptr || parse_error != nullptr) {
    std::cerr << "cannot create preprocessing pipeline: "
              << (parse_error != nullptr ? parse_error->message : "unknown error")
              << "\n";
    if (parse_error != nullptr) g_error_free(parse_error);
    if (pipeline != nullptr) gst_object_unref(pipeline);
    return EXIT_FAILURE;
  }
  GstElement* sink_element =
      gst_bin_get_by_name(GST_BIN(pipeline), "benchmark_sink");
  if (sink_element == nullptr) {
    std::cerr << "preprocessing pipeline has no appsink\n";
    gst_object_unref(pipeline);
    return EXIT_FAILURE;
  }
  GstAppSink* sink = GST_APP_SINK(sink_element);
  const auto pipeline_started_at = std::chrono::steady_clock::now();
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) ==
      GST_STATE_CHANGE_FAILURE) {
    print_gstreamer_error(pipeline, "cannot start preprocessing pipeline");
    gst_object_unref(sink_element);
    gst_object_unref(pipeline);
    return EXIT_FAILURE;
  }

  std::uint64_t frames = 0;
  std::uint64_t buffer_offset_gaps = 0;
  double interval_mean_s = 0.0;
  double interval_m2_s2 = 0.0;
  double maximum_interval_s = 0.0;
  guint64 previous_offset = GST_BUFFER_OFFSET_NONE;
  std::chrono::steady_clock::time_point first_received_at{};
  std::chrono::steady_clock::time_point previous_received_at{};
  std::chrono::steady_clock::time_point last_received_at{};

  while (true) {
    const int timeout_ms =
        frames == 0 ? config.startup_timeout_ms : config.frame_timeout_ms;
    GstSample* sample = gst_app_sink_try_pull_sample(
        sink, static_cast<GstClockTime>(timeout_ms) * GST_MSECOND);
    if (sample == nullptr) {
      print_gstreamer_error(pipeline, "preprocessing frame timeout");
      gst_element_set_state(pipeline, GST_STATE_NULL);
      gst_object_unref(sink_element);
      gst_object_unref(pipeline);
      return EXIT_FAILURE;
    }
    const auto received_at = std::chrono::steady_clock::now();
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    const guint64 offset = GST_BUFFER_OFFSET(buffer);
    if (frames == 0) {
      first_received_at = received_at;
    } else {
      const double interval_s =
          std::chrono::duration<double>(received_at - previous_received_at)
              .count();
      const double delta = interval_s - interval_mean_s;
      interval_mean_s += delta / static_cast<double>(frames);
      interval_m2_s2 += delta * (interval_s - interval_mean_s);
      maximum_interval_s = std::max(maximum_interval_s, interval_s);
      if (previous_offset != GST_BUFFER_OFFSET_NONE &&
          offset != GST_BUFFER_OFFSET_NONE && offset > previous_offset + 1U) {
        buffer_offset_gaps += offset - previous_offset - 1U;
      }
    }
    previous_offset = offset;
    previous_received_at = received_at;
    last_received_at = received_at;
    ++frames;
    gst_sample_unref(sample);
    if (std::chrono::duration<double>(received_at - first_received_at).count() >=
        config.duration_s) {
      break;
    }
  }

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(sink_element);
  gst_object_unref(pipeline);
  const double elapsed_s =
      std::chrono::duration<double>(last_received_at - first_received_at).count();
  const double delivered_fps =
      frames > 1 && elapsed_s > 0.0 ? (frames - 1) / elapsed_s : 0.0;
  const double startup_s =
      std::chrono::duration<double>(first_received_at - pipeline_started_at)
          .count();
  const double interval_std_s = frames > 2
                                    ? std::sqrt(interval_m2_s2 / (frames - 2))
                                    : 0.0;

  std::cout << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"mode\": \"gstreamer_gpu_preprocess_benchmark\",\n"
            << "  \"actuators_accessed\": false,\n"
            << "  \"decoder\": \"nvjpegdec\",\n"
            << "  \"transform\": \"nvvidconv\",\n"
            << "  \"memory\": \"NVMM\",\n"
            << "  \"input_width\": " << config.input_width << ",\n"
            << "  \"input_height\": " << config.input_height << ",\n"
            << "  \"output_width\": " << config.output_width << ",\n"
            << "  \"output_height\": " << config.output_height << ",\n"
            << "  \"requested_fps\": " << config.fps << ",\n"
            << "  \"flip_method\": " << config.flip_method << ",\n"
            << "  \"duration_s\": " << elapsed_s << ",\n"
            << "  \"startup_s\": " << startup_s << ",\n"
            << "  \"frames\": " << frames << ",\n"
            << "  \"buffer_offset_gaps\": " << buffer_offset_gaps << ",\n"
            << "  \"delivered_fps\": " << delivered_fps << ",\n"
            << "  \"mean_interval_ms\": " << interval_mean_s * 1000.0 << ",\n"
            << "  \"interval_std_ms\": " << interval_std_s * 1000.0 << ",\n"
            << "  \"maximum_interval_ms\": " << maximum_interval_s * 1000.0
            << "\n}\n";
  return EXIT_SUCCESS;
}
