#include "jetracer_sim/simulator.hpp"
#include "color_lane_native.hpp"
#ifdef __APPLE__
#include "jetracer_sim/coreml_segmentation.hpp"
#endif

#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/imgproc.hpp>

namespace py = pybind11;
using namespace jetracer::sim;

namespace {

py::array mat_view(const FramePtr& owner, const cv::Mat& matrix,
                   const std::string& format, std::size_t item_size) {
  auto* retained = new FramePtr(owner);
  py::capsule capsule(retained, [](void* pointer) {
    delete static_cast<FramePtr*>(pointer);
  });
  py::array result(py::buffer_info(
      matrix.data, static_cast<py::ssize_t>(item_size), format, 2,
      {matrix.rows, matrix.cols},
      {static_cast<py::ssize_t>(matrix.step[0]),
       static_cast<py::ssize_t>(item_size)}),
                   capsule);
  result.attr("setflags")(false);
  return result;
}

py::array bgr_copy(const Frame& frame) {
  const cv::Mat bgr = frame_to_bgr(frame);
  py::array_t<std::uint8_t> output(
      {static_cast<py::ssize_t>(bgr.rows), static_cast<py::ssize_t>(bgr.cols),
       static_cast<py::ssize_t>(3)});
  std::memcpy(output.mutable_data(), bgr.data,
              static_cast<std::size_t>(bgr.total() * bgr.elemSize()));
  return output;
}

}  // namespace

PYBIND11_MODULE(_native, module) {
  module.doc() = "Native JetRacer high-rate camera simulator";
  module.attr("SCENE_SCHEMA_VERSION") = kSceneSchemaVersion;

  py::class_<ColorLaneNativeProcessor>(module, "NativeColorLaneProcessor")
      .def(py::init([](const std::string& profile_path) {
        return std::make_unique<ColorLaneNativeProcessor>(
            load_color_lane_native_config(profile_path));
      }))
      .def("infer", [](ColorLaneNativeProcessor& processor,
                       py::array_t<std::uint8_t,
                                   py::array::c_style | py::array::forcecast>
                           image) {
        const py::buffer_info input = image.request();
        if (input.ndim != 3 || input.shape[2] != 3) {
          throw std::invalid_argument(
              "native color-lane input must be an HxWx3 uint8 array");
        }
        ColorLaneNativeResult result;
        cv::Mat resized;
        {
          cv::Mat source(static_cast<int>(input.shape[0]),
                         static_cast<int>(input.shape[1]), CV_8UC3, input.ptr,
                         static_cast<std::size_t>(input.strides[0]));
          py::gil_scoped_release release;
          result = processor.process(source);
          cv::resize(result.labels, resized,
                     cv::Size(static_cast<int>(input.shape[1]),
                              static_cast<int>(input.shape[0])),
                     0.0, 0.0, cv::INTER_NEAREST);
        }
        py::array_t<std::uint8_t> labels(
            {static_cast<py::ssize_t>(resized.rows),
             static_cast<py::ssize_t>(resized.cols)});
        std::memcpy(labels.mutable_data(), resized.data,
                    resized.total() * resized.elemSize());
        py::list path;
        for (const cv::Point2f& point : result.center_path_normalized) {
          path.append(py::make_tuple(point.x, point.y));
        }
        return py::make_tuple(labels, result.confidence, result.observed_rows,
                              result.left_inlier_fraction,
                              result.right_inlier_fraction,
                              path, result.birdseye_applied);
      });

#ifdef __APPLE__
  module.attr("COREML_NATIVE_AVAILABLE") = true;
  py::class_<CoreMLSegmentationSession>(module, "CoreMLSegmentationSession")
      .def(py::init<const std::string&, const std::string&, const std::string&,
                    int, int, int, int, std::vector<int>, std::uint8_t, float,
                    std::vector<float>, std::vector<float>, const std::string&>(),
           py::arg("model_path"), py::arg("input_name"),
           py::arg("output_name"), py::arg("input_width"),
           py::arg("input_height"), py::arg("output_width"),
           py::arg("output_height"), py::arg("source_road_class_ids"),
           py::arg("road_class_id"), py::arg("input_scale"),
           py::arg("mean_rgb"), py::arg("std_rgb"),
           py::arg("compute_units"))
      .def_property_readonly("output_width",
                             &CoreMLSegmentationSession::output_width)
      .def_property_readonly("output_height",
                             &CoreMLSegmentationSession::output_height)
      .def("infer", [](CoreMLSegmentationSession& session,
                       py::array_t<std::uint8_t,
                                   py::array::c_style | py::array::forcecast>
                           image) {
        const py::buffer_info input = image.request();
        if (input.ndim != 3 || input.shape[2] != 3) {
          throw std::invalid_argument("Core ML input must be an HxWx3 uint8 array");
        }
        std::vector<std::uint8_t> labels;
        {
          py::gil_scoped_release release;
          labels = session.infer(
              static_cast<const std::uint8_t*>(input.ptr),
              static_cast<int>(input.shape[1]), static_cast<int>(input.shape[0]));
        }
        py::array_t<std::uint8_t> output(
            {session.output_height(), session.output_width()});
        std::memcpy(output.mutable_data(), labels.data(), labels.size());
        return output;
      });
#else
  module.attr("COREML_NATIVE_AVAILABLE") = false;
#endif
  module.attr("ROAD_SURFACE_INSTANCE_ID") =
      defaults::kRendererGroundInstanceIdsDrivableSurface;
  module.attr("ROAD_LEFT_BOUNDARY_INSTANCE_ID") =
      defaults::kRendererGroundInstanceIdsLeftBoundary;
  module.attr("ROAD_RIGHT_BOUNDARY_INSTANCE_ID") =
      defaults::kRendererGroundInstanceIdsRightBoundary;
  module.attr("ROAD_CENTER_DASH_INSTANCE_ID") =
      defaults::kRendererGroundInstanceIdsCenterDash;

  py::enum_<LensModel>(module, "LensModel")
      .value("BROWN_CONRADY", LensModel::BrownConrady)
      .value("FISHEYE_EQUIDISTANT", LensModel::FisheyeEquidistant);
  py::enum_<PixelFormat>(module, "PixelFormat")
      .value("NV12_VIDEO_RANGE", PixelFormat::Nv12VideoRange);
  py::enum_<ShutterType>(module, "ShutterType")
      .value("GLOBAL", ShutterType::Global)
      .value("ROLLING", ShutterType::Rolling);
  py::enum_<SemanticClass>(module, "SemanticClass")
      .value("BACKGROUND", SemanticClass::Background)
      .value("DRIVABLE_SURFACE", SemanticClass::DrivableSurface)
      .value("LANE_MARKING", SemanticClass::LaneMarking)
      .value("STOP_SIGN", SemanticClass::StopSign)
      .value("OBSTACLE", SemanticClass::Obstacle)
      .value("CENTER_MARKING", SemanticClass::CenterMarking);
  py::enum_<ObjectType>(module, "ObjectType")
      .value("BOX", ObjectType::Box)
      .value("CYLINDER", ObjectType::Cylinder)
      .value("STOP_SIGN", ObjectType::StopSign)
      .value("BILLBOARD", ObjectType::Billboard);

  py::class_<Point2>(module, "Point2")
      .def(py::init<>())
      .def_readwrite("x", &Point2::x)
      .def_readwrite("y", &Point2::y);
  py::class_<Pose2D>(module, "Pose2D")
      .def(py::init<>())
      .def_readwrite("x", &Pose2D::x)
      .def_readwrite("y", &Pose2D::y)
      .def_readwrite("yaw", &Pose2D::yaw);
  py::class_<VehicleCommand>(module, "VehicleCommand")
      .def(py::init<>())
      .def(py::init([](double target_speed_mps, double steering_rad) {
             return VehicleCommand{target_speed_mps, steering_rad};
           }),
           py::arg("target_speed_mps"), py::arg("steering_rad"))
      .def_readwrite("target_speed_mps", &VehicleCommand::target_speed_mps)
      .def_readwrite("steering_rad", &VehicleCommand::steering_rad);
  py::class_<VehicleConfig>(module, "VehicleConfig")
      .def(py::init<>())
      .def_readwrite("wheelbase_m", &VehicleConfig::wheelbase_m)
      .def_readwrite("body_width_m", &VehicleConfig::body_width_m)
      .def_readwrite("front_overhang_m", &VehicleConfig::front_overhang_m)
      .def_readwrite("rear_overhang_m", &VehicleConfig::rear_overhang_m)
      .def_readwrite("max_steering_rad", &VehicleConfig::max_steering_rad)
      .def_readwrite("steering_time_constant_s",
                     &VehicleConfig::steering_time_constant_s)
      .def_readwrite("motor_time_constant_s",
                     &VehicleConfig::motor_time_constant_s)
      .def_property_readonly("body_length_m", &VehicleConfig::body_length_m)
      .def_property_readonly("minimum_turn_radius_m",
                             &VehicleConfig::minimum_turn_radius_m);
  py::class_<VehicleState>(module, "VehicleState")
      .def(py::init<>())
      .def_readwrite("pose", &VehicleState::pose)
      .def_readwrite("speed_mps", &VehicleState::speed_mps)
      .def_readwrite("steering_rad", &VehicleState::steering_rad);

  py::class_<CameraProfile>(module, "CameraProfile")
      .def(py::init<>())
      .def_readwrite("id", &CameraProfile::id)
      .def_readwrite("width", &CameraProfile::width)
      .def_readwrite("height", &CameraProfile::height)
      .def_readwrite("fps_numerator", &CameraProfile::fps_numerator)
      .def_readwrite("fps_denominator", &CameraProfile::fps_denominator)
      .def_readwrite("pixel_format", &CameraProfile::pixel_format)
      .def_readwrite("lens_model", &CameraProfile::lens_model)
      .def_readwrite("shutter", &CameraProfile::shutter)
      .def_readwrite("nominal_hfov_rad", &CameraProfile::nominal_hfov_rad)
      .def_readwrite("fx", &CameraProfile::fx)
      .def_readwrite("fy", &CameraProfile::fy)
      .def_readwrite("cx", &CameraProfile::cx)
      .def_readwrite("cy", &CameraProfile::cy)
      .def_readwrite("distortion", &CameraProfile::distortion)
      .def_readwrite("mount_x_m", &CameraProfile::mount_x_m)
      .def_readwrite("mount_y_m", &CameraProfile::mount_y_m)
      .def_readwrite("mount_z_m", &CameraProfile::mount_z_m)
      .def_readwrite("mount_roll_rad", &CameraProfile::mount_roll_rad)
      .def_readwrite("mount_pitch_down_rad",
                     &CameraProfile::mount_pitch_down_rad)
      .def_readwrite("mount_yaw_rad", &CameraProfile::mount_yaw_rad)
      .def_readwrite("mount_provisional", &CameraProfile::mount_provisional)
      .def_readwrite("exposure_s", &CameraProfile::exposure_s)
      .def_readwrite("rolling_readout_s", &CameraProfile::rolling_readout_s)
      .def_readwrite("provisional", &CameraProfile::provisional)
      .def_property_readonly("fps", &CameraProfile::fps)
      .def_property_readonly("frame_period_s", &CameraProfile::frame_period_s)
      .def("apply_nominal_intrinsics", &CameraProfile::apply_nominal_intrinsics)
      .def("apply_opencv_calibration",
           &CameraProfile::apply_opencv_calibration)
      .def("validate", &CameraProfile::validate)
      .def_static("elp_112", &CameraProfile::elp_112)
      .def_static("stress_720p_200", &CameraProfile::stress_720p_200)
      .def_static("imx219_160_provisional",
                  &CameraProfile::imx219_160_provisional);

  py::class_<SceneConfig>(module, "SceneConfig")
      .def(py::init<>())
      .def_readwrite("seed", &SceneConfig::seed)
      .def_readwrite("control_points", &SceneConfig::control_points)
      .def_readwrite("samples_per_segment",
                     &SceneConfig::samples_per_segment)
      .def_readwrite("base_radius_m", &SceneConfig::base_radius_m)
      .def_readwrite("radius_jitter_m", &SceneConfig::radius_jitter_m)
      .def_readwrite("road_width_m", &SceneConfig::road_width_m)
      .def_readwrite("atlas_pixels_per_metre",
                     &SceneConfig::atlas_pixels_per_metre)
      .def_readwrite("obstacle_count", &SceneConfig::obstacle_count)
      .def_readwrite("stop_sign_count", &SceneConfig::stop_sign_count)
      .def_readwrite("background_texture_path",
                     &SceneConfig::background_texture_path)
      .def_readwrite("road_texture_path", &SceneConfig::road_texture_path);

  py::class_<SceneObject>(module, "SceneObject")
      .def(py::init<>())
      .def_readwrite("instance_id", &SceneObject::instance_id)
      .def_readwrite("type", &SceneObject::type)
      .def_readwrite("semantic_class", &SceneObject::semantic_class)
      .def_readwrite("position", &SceneObject::position)
      .def_readwrite("base_z_m", &SceneObject::base_z_m)
      .def_readwrite("yaw_rad", &SceneObject::yaw_rad)
      .def_readwrite("width_m", &SceneObject::width_m)
      .def_readwrite("depth_m", &SceneObject::depth_m)
      .def_readwrite("height_m", &SceneObject::height_m)
      .def_readwrite("collision_width_m", &SceneObject::collision_width_m)
      .def_readwrite("collision_depth_m", &SceneObject::collision_depth_m)
      .def_readwrite("radial_segments", &SceneObject::radial_segments)
      .def_readwrite("texture_path", &SceneObject::texture_path)
      .def_readwrite("bgr", &SceneObject::bgr);
  py::class_<Scene>(module, "Scene")
      .def(py::init<>())
      .def_readwrite("schema_version", &Scene::schema_version)
      .def_readwrite("seed", &Scene::seed)
      .def_readwrite("road_width_m", &Scene::road_width_m)
      .def_readwrite("atlas_pixels_per_metre",
                     &Scene::atlas_pixels_per_metre)
      .def_readwrite("vehicle", &Scene::vehicle)
      .def_readwrite("start", &Scene::start)
      .def_readwrite("camera", &Scene::camera)
      .def_readwrite("background_texture_path", &Scene::background_texture_path)
      .def_readwrite("road_texture_path", &Scene::road_texture_path)
      .def_readwrite("centerline", &Scene::centerline)
      .def_readwrite("objects", &Scene::objects)
      .def_static("generate", &Scene::generate)
      .def_static("load", &Scene::load)
      .def("save", &Scene::save)
      .def("validate", &Scene::validate);

  py::class_<Detection>(module, "Detection")
      .def_readonly("class_id", &Detection::class_id)
      .def_readonly("instance_id", &Detection::instance_id)
      .def_readonly("bbox_xyxy", &Detection::bbox_xyxy)
      .def_readonly("visibility", &Detection::visibility)
      .def_readonly("range_m", &Detection::range_m)
      .def_readonly("forward_m", &Detection::forward_m)
      .def_readonly("lateral_m", &Detection::lateral_m)
      .def_readonly("relative_yaw_rad", &Detection::relative_yaw_rad);

  py::class_<Frame, FramePtr>(module, "Frame")
      .def_readonly("frame_id", &Frame::frame_id)
      .def_readonly("simulation_time_s", &Frame::simulation_time_s)
      .def_readonly("exposure_start_s", &Frame::exposure_start_s)
      .def_readonly("exposure_end_s", &Frame::exposure_end_s)
      .def_readonly("vehicle", &Frame::vehicle)
      .def_readonly("camera", &Frame::camera)
      .def_readonly("detections", &Frame::detections)
      .def_property_readonly("y_plane", [](const FramePtr& frame) {
        return mat_view(frame, frame->y_plane,
                        py::format_descriptor<std::uint8_t>::format(), 1);
      })
      .def_property_readonly("uv_plane", [](const FramePtr& frame) {
        return mat_view(frame, frame->uv_plane,
                        py::format_descriptor<std::uint8_t>::format(), 1);
      })
      .def_property_readonly("semantic", [](const FramePtr& frame) {
        return mat_view(frame, frame->semantic,
                        py::format_descriptor<std::uint8_t>::format(), 1);
      })
      .def_property_readonly("instance", [](const FramePtr& frame) {
        return mat_view(frame, frame->instance,
                        py::format_descriptor<std::uint32_t>::format(), 4);
      })
      .def("to_bgr", [](const Frame& frame) { return bgr_copy(frame); });

  py::class_<Simulator>(module, "Simulator")
      .def(py::init<Scene, CameraProfile>())
      .def("reset", py::overload_cast<>(&Simulator::reset))
      .def("reset",
           py::overload_cast<Scene, CameraProfile>(&Simulator::reset),
           py::arg("scene"), py::arg("camera"))
      .def("set_vehicle_state", &Simulator::set_vehicle_state,
           py::arg("state"))
      .def("advance", &Simulator::advance, py::arg("command"),
           py::arg("dt_s") = 0.005,
           py::call_guard<py::gil_scoped_release>())
      .def("render_now", &Simulator::render_now,
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("scene", &Simulator::scene,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("camera", &Simulator::camera,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("vehicle_state", &Simulator::vehicle_state,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("simulation_time_s",
                             &Simulator::simulation_time_s);
}
