cmake_minimum_required(VERSION 3.10)
project(jetracer_sim VERSION 0.1.0 LANGUAGES CXX)

if(APPLE)
  enable_language(OBJCXX)
endif()

option(JETRACER_SIM_BUILD_PYTHON "Build the pybind11 extension" ON)
option(JETRACER_SIM_BUILD_TESTS "Build native tests" ON)
option(JETRACER_SIM_BUILD_NATIVE_CLI
       "Build the native CLI and its interactive OpenCV dependency" ON)
option(JETRACER_SIM_BUILD_JETSON_RUNTIME
       "Build the headless CUDA/TensorRT Jetson runtime probe" OFF)
set(JETRACER_SIM_EXAMPLE_TEST_TIMEOUT_SECONDS 60 CACHE STRING
    "Timeout for simulator example tests run by CTest")
set(JETRACER_SIM_BENCHMARK_TEST_TIMEOUT_SECONDS 180 CACHE STRING
    "Timeout for simulator benchmark tests run by CTest")
set(JETRACER_SIM_NATIVE_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/native_simulator_defaults.json"
    CACHE FILEPATH "Native simulator defaults JSON used to generate C++ constants")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_VERSION VERSION_LESS 3.12)
  find_package(PythonInterp 3 REQUIRED)
  set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
else()
  find_package(Python COMPONENTS Interpreter REQUIRED)
endif()
set(JETRACER_SIM_OPENCV_COMPONENTS core imgproc imgcodecs videoio)
if(JETRACER_SIM_BUILD_NATIVE_CLI)
  list(APPEND JETRACER_SIM_OPENCV_COMPONENTS highgui)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${JETRACER_SIM_OPENCV_COMPONENTS})
if(OpenCV_VERSION VERSION_LESS 4.1)
  message(FATAL_ERROR "JetRacer simulator requires OpenCV 4.1 or newer")
endif()

set(JETRACER_SIM_GENERATED_INCLUDE_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/generated/include)
set(JETRACER_SIM_GENERATED_DEFAULTS_HEADER
    ${JETRACER_SIM_GENERATED_INCLUDE_DIR}/jetracer_sim/native_defaults.hpp)
if(NOT CMAKE_VERSION VERSION_LESS 3.12)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${JETRACER_SIM_NATIVE_CONFIG}
    ${CMAKE_CURRENT_SOURCE_DIR}/configs/driving_benchmarks.json
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_native_defaults.py)
endif()
execute_process(
  COMMAND
    ${Python_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_native_defaults.py
    --native ${JETRACER_SIM_NATIVE_CONFIG}
    --driving ${CMAKE_CURRENT_SOURCE_DIR}/configs/driving_benchmarks.json
    --output ${JETRACER_SIM_GENERATED_DEFAULTS_HEADER}
  RESULT_VARIABLE JETRACER_SIM_DEFAULTS_RESULT
  ERROR_VARIABLE JETRACER_SIM_DEFAULTS_ERROR)
if(NOT JETRACER_SIM_DEFAULTS_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Failed to generate native defaults: ${JETRACER_SIM_DEFAULTS_ERROR}")
endif()

add_library(jetracer_sim_core src/simulator.cpp)
set_target_properties(jetracer_sim_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(jetracer_sim_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${JETRACER_SIM_GENERATED_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jetracer_sim_core
  PUBLIC opencv_core opencv_imgproc opencv_imgcodecs)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(jetracer_sim_core PUBLIC stdc++fs)
endif()

add_library(jetracer_color_lane STATIC src/color_lane_native.cpp)
set_target_properties(jetracer_color_lane PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(jetracer_color_lane
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jetracer_color_lane
  PUBLIC opencv_core opencv_imgproc)
if(CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang|GNU)$")
  target_compile_options(jetracer_color_lane PRIVATE
    -Wall -Wextra -Wpedantic)
endif()

add_library(jetracer_shadow_controller STATIC src/native_shadow_controller.cpp)
set_target_properties(jetracer_shadow_controller PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(jetracer_shadow_controller
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(jetracer_shadow_controller
  PUBLIC jetracer_color_lane opencv_core)
if(CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang|GNU)$")
  target_compile_options(jetracer_shadow_controller PRIVATE
    -Wall -Wextra -Wpedantic)
endif()

add_executable(jetracer-shadow-replay src/native_shadow_replay.cpp)
target_link_libraries(jetracer-shadow-replay PRIVATE
  jetracer_shadow_controller opencv_videoio)

add_executable(jetracer-color-lane-benchmark
  src/color_lane_benchmark_main.cpp
  src/color_lane_image_benchmark.cpp)
target_link_libraries(jetracer-color-lane-benchmark
  PRIVATE jetracer_color_lane opencv_imgcodecs)
if(CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang|GNU)$")
  target_compile_options(jetracer_sim_core PRIVATE
    -Wall -Wextra -Wpedantic)
endif()

if(JETRACER_SIM_BUILD_NATIVE_CLI)
  add_executable(jetracer-sim src/cli.cpp)
  target_link_libraries(jetracer-sim PRIVATE jetracer_sim_core opencv_highgui)
  if(APPLE)
    target_link_libraries(jetracer-sim PRIVATE "-framework ApplicationServices")
  endif()
endif()

if(JETRACER_SIM_BUILD_PYTHON)
  if(CMAKE_VERSION VERSION_LESS 3.12)
    message(FATAL_ERROR
      "Python extension builds require CMake 3.12 or newer; configure with "
      "-DJETRACER_SIM_BUILD_PYTHON=OFF on JetPack 4")
  endif()
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native src/bindings.cpp)
  target_link_libraries(_native PRIVATE jetracer_sim_core jetracer_color_lane)
  if(APPLE)
    target_sources(_native PRIVATE src/coreml_segmentation.mm)
    target_link_libraries(_native PRIVATE "-framework CoreML" "-framework Foundation")
    target_compile_options(_native PRIVATE
      $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  endif()
  set(JETRACER_SIM_PYTHON_BUILD_DIR
      ${CMAKE_CURRENT_BINARY_DIR}/python/jetracer_sim)
  set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${JETRACER_SIM_PYTHON_BUILD_DIR})
  file(MAKE_DIRECTORY ${JETRACER_SIM_PYTHON_BUILD_DIR})
  foreach(python_file IN ITEMS
      __init__.py
      avoidance.py
      benchmark_controllers.py
      benchmarking.py
      bringup.py
      camera_runtime_config.py
      certification_catalog.py
      clip_benchmark.py
      color_lane.py
      configuration.py
      configuration_space.py
      controller.py
      coreml_adapter.py
      dataset.py
      deployment.py
      deployment_config.py
      deployment_release.py
      deployment_supervisor.py
      detection.py
      document_io.py
      evaluation.py
      frame_source.py
      governor.py
      gui_safety.py
      hardware_actuator.py
      hardware_profiles.py
      inference.py
      model_registry.py
      observability.py
      onnx_adapters.py
      performance_gate.py
      platform_runtime.py
      pretrained.py
      process_safety.py
      real_track_dataset.py
      real_track_profiles.py
      readiness.py
      realtime.py
      realtime_capture.py
      realtime_presentation.py
      resource_paths.py
      speed_certification.py
      state_validation.py
      stop_certification.py
      stopping.py
      synthetic_clip.py
      top_down_video.py
      tracks.py
      vehicle_io.py
      video_lane_calibration.py)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/python/jetracer_sim/${python_file}
      ${JETRACER_SIM_PYTHON_BUILD_DIR}/${python_file}
      COPYONLY)
  endforeach()
  file(MAKE_DIRECTORY ${JETRACER_SIM_PYTHON_BUILD_DIR}/configs)
  foreach(config_file IN ITEMS
      deployment.json
      driving_benchmarks.json
      heading_layer_cspace.json
      jetson_software_baseline.json
      road_segmentation_models.json
      runtime_defaults.json
      stop_sign_benchmark.json
      stop_sign_latency_profiles.json)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/${config_file}
      ${JETRACER_SIM_PYTHON_BUILD_DIR}/configs/${config_file}
      COPYONLY)
  endforeach()
  configure_file(
    ${JETRACER_SIM_NATIVE_CONFIG}
    ${JETRACER_SIM_PYTHON_BUILD_DIR}/configs/native_simulator_defaults.json
    COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configs/off_the_shelf_models.json
    ${JETRACER_SIM_PYTHON_BUILD_DIR}/configs/off_the_shelf_models.json
    COPYONLY)
  file(MAKE_DIRECTORY
    ${JETRACER_SIM_PYTHON_BUILD_DIR}/configs/platforms
    ${JETRACER_SIM_PYTHON_BUILD_DIR}/configs/hardware
    ${JETRACER_SIM_PYTHON_BUILD_DIR}/configs/assets
    ${JETRACER_SIM_PYTHON_BUILD_DIR}/benchmarks)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configs/assets/pedestrian-abhinav-kamal-cc0.jpg
    ${JETRACER_SIM_PYTHON_BUILD_DIR}/configs/assets/pedestrian-abhinav-kamal-cc0.jpg
    COPYONLY)
  foreach(platform_file IN ITEMS sim.json jetracer-pro.json)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/platforms/${platform_file}
      ${JETRACER_SIM_PYTHON_BUILD_DIR}/configs/platforms/${platform_file}
      COPYONLY)
  endforeach()
  foreach(hardware_file IN ITEMS
      cameras.json
      actuator.json
      vehicle_state.json
      jetson_deployment.json
      preflight.json
      bringup_stages.json)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/configs/hardware/${hardware_file}
      ${JETRACER_SIM_PYTHON_BUILD_DIR}/configs/hardware/${hardware_file}
      COPYONLY)
  endforeach()
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/off_the_shelf_model_benchmarks.json
    ${JETRACER_SIM_PYTHON_BUILD_DIR}/benchmarks/off_the_shelf_model_benchmarks.json
    COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/road_model_benchmarks.json
    ${JETRACER_SIM_PYTHON_BUILD_DIR}/benchmarks/road_model_benchmarks.json
    COPYONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/certified_speed_limits.json
    ${JETRACER_SIM_PYTHON_BUILD_DIR}/benchmarks/certified_speed_limits.json
    COPYONLY)
  if(EXISTS
     ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/speed_certification_results.json)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/speed_certification_results.json
      ${JETRACER_SIM_PYTHON_BUILD_DIR}/benchmarks/speed_certification_results.json
      COPYONLY)
  endif()
  install(TARGETS _native LIBRARY DESTINATION jetracer_sim)
  install(
    DIRECTORY python/jetracer_sim/
    DESTINATION jetracer_sim
    FILES_MATCHING PATTERN "*.py")
  install(
    FILES
      configs/deployment.json
      configs/driving_benchmarks.json
      configs/heading_layer_cspace.json
      configs/jetson_software_baseline.json
      configs/road_segmentation_models.json
      configs/runtime_defaults.json
      configs/stop_sign_benchmark.json
      configs/stop_sign_latency_profiles.json
    DESTINATION jetracer_sim/configs)
  install(
    FILES ${JETRACER_SIM_NATIVE_CONFIG}
    DESTINATION jetracer_sim/configs
    RENAME native_simulator_defaults.json)
  install(
    FILES configs/off_the_shelf_models.json
    DESTINATION jetracer_sim/configs)
  install(
    DIRECTORY configs/assets/
    DESTINATION jetracer_sim/configs/assets)
  install(
    DIRECTORY configs/platforms/
    DESTINATION jetracer_sim/configs/platforms
    FILES_MATCHING PATTERN "*.json")
  install(
    DIRECTORY configs/hardware/
    DESTINATION jetracer_sim/configs/hardware
    FILES_MATCHING PATTERN "*.json")
  install(
    FILES
      benchmarks/off_the_shelf_model_benchmarks.json
      benchmarks/road_model_benchmarks.json
      benchmarks/certified_speed_limits.json
    DESTINATION jetracer_sim/benchmarks)
  if(EXISTS
     ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/speed_certification_results.json)
    install(
      FILES benchmarks/speed_certification_results.json
      DESTINATION jetracer_sim/benchmarks)
  endif()
endif()

if(JETRACER_SIM_BUILD_JETSON_RUNTIME)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(JETRACER_GSTREAMER REQUIRED
    gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
  find_path(JETRACER_TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include)
  find_path(JETRACER_CUDA_INCLUDE_DIR cuda_runtime_api.h
    HINTS /usr/local/cuda/include)
  find_library(JETRACER_TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib/aarch64-linux-gnu)
  find_library(JETRACER_CUDART_LIBRARY cudart
    HINTS /usr/local/cuda/lib64
          /usr/local/cuda/targets/aarch64-linux/lib)
  if(NOT JETRACER_TENSORRT_INCLUDE_DIR OR
     NOT JETRACER_CUDA_INCLUDE_DIR OR
     NOT JETRACER_TENSORRT_LIBRARY OR
     NOT JETRACER_CUDART_LIBRARY)
    message(FATAL_ERROR
      "Jetson runtime requires TensorRT and CUDA development packages")
  endif()

  add_executable(jetracer-jetson-runtime
    src/gstreamer_color_lane.cpp
    src/gstreamer_preprocess.cpp
    src/jetson_runtime.cpp
    src/v4l2_capture.cpp)
  target_include_directories(jetracer-jetson-runtime PRIVATE
    ${JETRACER_TENSORRT_INCLUDE_DIR}
    ${JETRACER_CUDA_INCLUDE_DIR}
    ${JETRACER_GSTREAMER_INCLUDE_DIRS})
  target_link_libraries(jetracer-jetson-runtime PRIVATE
    jetracer_color_lane
    ${JETRACER_TENSORRT_LIBRARY}
    ${JETRACER_CUDART_LIBRARY}
    ${JETRACER_GSTREAMER_LIBRARIES})
  if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|GNU)$")
    target_compile_options(jetracer-jetson-runtime PRIVATE
      -Wall -Wextra -Wpedantic)
  endif()
  install(TARGETS jetracer-jetson-runtime RUNTIME DESTINATION bin)
endif()

if(JETRACER_SIM_BUILD_TESTS)
  enable_testing()
  add_executable(jetracer_sim_tests tests/test_simulator.cpp)
  target_link_libraries(jetracer_sim_tests PRIVATE jetracer_sim_core)
  add_test(NAME jetracer_sim_tests COMMAND jetracer_sim_tests)
  add_executable(jetracer_color_lane_native_tests
    tests/test_color_lane_native.cpp)
  target_link_libraries(jetracer_color_lane_native_tests
    PRIVATE jetracer_color_lane)
  add_test(
    NAME jetracer_color_lane_native_tests
    COMMAND jetracer_color_lane_native_tests
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/color_lane/waveshare-sim-white.json)
  add_executable(jetracer_native_shadow_tests
    tests/test_native_shadow_controller.cpp)
  target_link_libraries(jetracer_native_shadow_tests PRIVATE
    jetracer_shadow_controller)
  add_test(
    NAME jetracer_native_shadow_tests
    COMMAND jetracer_native_shadow_tests
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/native_shadow_controller.json)
  if(JETRACER_SIM_BUILD_JETSON_RUNTIME)
    add_test(NAME jetracer_jetson_runtime_self_test
             COMMAND jetracer-jetson-runtime --self-test)
  endif()
  if(JETRACER_SIM_BUILD_PYTHON)
    add_test(
      NAME jetracer_sim_python_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_bindings.py)
    add_test(
      NAME jetracer_sim_inference_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_inference_governor.py)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import cv2"
      RESULT_VARIABLE JETRACER_SIM_CV2_RESULT
      OUTPUT_QUIET ERROR_QUIET)
    if(JETRACER_SIM_CV2_RESULT EQUAL 0)
    add_test(
      NAME jetracer_sim_color_lane_tests
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
                ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_color_lane.py)
      add_test(
        NAME jetracer_sim_video_lane_calibration_tests
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
                ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_video_lane_calibration.py)
    else()
      message(STATUS "Python OpenCV unavailable; skipping colour-lane tests")
    endif()
    add_test(
      NAME jetracer_sim_controller_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_controller.py)
    add_test(
      NAME jetracer_sim_avoidance_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_avoidance.py)
    add_test(
      NAME jetracer_sim_cylinder_robustness_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cylinder_robustness.py)
    add_test(
      NAME jetracer_sim_speed_certification_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_speed_certification.py)
    add_test(
      NAME jetracer_sim_speed_certification_matrix_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_speed_certification_matrix.py)
    add_test(
      NAME jetracer_sim_stop_certification_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_stop_certification.py)
    add_test(
      NAME jetracer_sim_certification_catalog_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_certification_catalog.py)
    add_test(
      NAME jetracer_sim_deployment_stack_plan
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_deployment_stack_benchmark.py
              --config
              ${CMAKE_CURRENT_SOURCE_DIR}/configs/deployment_stack_benchmark.json
              --dry-run
              --overwrite
              --output
              ${CMAKE_CURRENT_BINARY_DIR}/deployment-stack-plan.json)
    add_test(
      NAME jetracer_sim_realtime_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_realtime_runtime.py)
    add_test(
      NAME jetracer_sim_realtime_presentation_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_realtime_presentation.py)
    add_test(
      NAME jetracer_sim_realtime_capture_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_realtime_capture.py)
    set_tests_properties(jetracer_sim_realtime_capture_tests
      PROPERTIES TIMEOUT 30)
    add_test(
      NAME jetracer_sim_camera_runtime_config_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_camera_runtime_config.py)
    set_tests_properties(jetracer_sim_camera_runtime_config_tests
      PROPERTIES TIMEOUT 30)
    add_test(
      NAME jetracer_sim_platform_realtime_smoke
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/examples/realtime_demo.py
              --platform-config
              ${CMAKE_CURRENT_SOURCE_DIR}/configs/platforms/sim.json
              --model-config
              ${CMAKE_CURRENT_SOURCE_DIR}/configs/demo_models.json
              --benchmark-registry
              ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/demo_model_benchmarks.json
              --headless --duration 0.5 --switch-every 0.1 --no-log)
    set_tests_properties(jetracer_sim_platform_realtime_smoke
      PROPERTIES TIMEOUT 15)
    add_test(
      NAME jetracer_sim_onnx_stop_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_onnx_stop.py)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import PIL"
      RESULT_VARIABLE JETRACER_SIM_PIL_RESULT
      OUTPUT_QUIET ERROR_QUIET)
    if(JETRACER_SIM_PIL_RESULT EQUAL 0)
      add_test(
        NAME jetracer_sim_dataset_export_tests
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
                ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dataset_export.py)
      add_test(
        NAME jetracer_sim_synthetic_clip_tests
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
                ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_synthetic_clip.py)
      add_test(
        NAME jetracer_sim_segmentation_evaluation_tests
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
                ${Python_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_segmentation_evaluation.py)
    else()
      message(STATUS "Pillow unavailable; skipping dataset/evaluation tests")
    endif()
    add_test(
      NAME jetracer_sim_driving_benchmark_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_driving_benchmarks.py)
    set_tests_properties(jetracer_sim_driving_benchmark_tests
      PROPERTIES TIMEOUT ${JETRACER_SIM_BENCHMARK_TEST_TIMEOUT_SECONDS})
    add_test(
      NAME jetracer_sim_control_benchmark_smoke
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_control_benchmarks.py
              --platform ${CMAKE_CURRENT_SOURCE_DIR}/configs/platforms/sim.json
              --track waveshare_3x2
              --laps 1
              --output
              ${CMAKE_CURRENT_BINARY_DIR}/control-benchmark-smoke.json
              --overwrite)
    set_tests_properties(jetracer_sim_control_benchmark_smoke
      PROPERTIES TIMEOUT ${JETRACER_SIM_BENCHMARK_TEST_TIMEOUT_SECONDS})
    add_test(
      NAME jetracer_sim_speed_certification_smoke
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tools/find_max_safe_speed.py
              --platform ${CMAKE_CURRENT_SOURCE_DIR}/configs/platforms/sim.json
              --track waveshare_3x2
              --minimum-speed 0.5
              --maximum-speed 0.75
              --coarse-step 0.25
              --tolerance 0.1
              --laps 1
              --trials 1
              --speed-planner off
              --output
              ${CMAKE_CURRENT_BINARY_DIR}/speed-certification-smoke.json
              --overwrite
              --no-update-registry)
    set_tests_properties(jetracer_sim_speed_certification_smoke
      PROPERTIES TIMEOUT 30)
    add_test(
      NAME jetracer_sim_camera_mount_benchmark_smoke
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_camera_mount_benchmarks.py
              --platform ${CMAKE_CURRENT_SOURCE_DIR}/configs/platforms/sim.json
              --track waveshare_3x2
              --laps 1
              --axis height
              --height 0.16
              --output
              ${CMAKE_CURRENT_BINARY_DIR}/camera-mount-benchmark-smoke.json
              --overwrite)
    set_tests_properties(jetracer_sim_camera_mount_benchmark_smoke
      PROPERTIES TIMEOUT ${JETRACER_SIM_BENCHMARK_TEST_TIMEOUT_SECONDS})
    add_test(
      NAME jetracer_sim_driving_benchmark_summary_tests
      COMMAND ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_driving_benchmark_summary.py)
    add_test(
      NAME jetracer_sim_driving_benchmark_comparison_tests
      COMMAND ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_driving_benchmark_comparison.py)
    add_test(
      NAME jetracer_sim_driving_benchmark_promotion_tests
      COMMAND ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_driving_benchmark_promotion.py)
    add_test(
      NAME jetracer_sim_configuration_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_configuration.py)
    set_tests_properties(jetracer_sim_configuration_tests PROPERTIES TIMEOUT 30)
    add_test(
      NAME jetracer_sim_configuration_space_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_configuration_space.py)
    set_tests_properties(jetracer_sim_configuration_space_tests
      PROPERTIES TIMEOUT 30)
    add_test(
      NAME jetracer_sim_real_track_dataset_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_real_track_dataset.py)
    set_tests_properties(jetracer_sim_real_track_dataset_tests
      PROPERTIES TIMEOUT 30)
    add_test(
      NAME jetracer_sim_camera_characterization_tests
      COMMAND ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_camera_characterization.py)
    add_test(
      NAME jetracer_sim_frame_source_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_frame_source.py)
    add_test(
      NAME jetracer_sim_platform_runtime_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_platform_runtime.py)
    add_test(
      NAME jetracer_sim_performance_gate_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_performance_gate.py)
    add_test(
      NAME jetracer_sim_performance_evidence_test
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_realtime_performance_evidence.py)
    add_test(
      NAME jetracer_sim_process_safety_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_process_safety.py)
    add_test(
      NAME jetracer_sim_document_io_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_document_io.py)
    add_test(
      NAME jetracer_sim_resource_path_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_resource_paths.py)
    add_test(
      NAME jetracer_sim_jetson_compatibility_tests
      COMMAND ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_jetson_compatibility.py)
    add_test(
      NAME jetracer_sim_hardware_profile_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hardware_profiles.py)
    add_test(
      NAME jetracer_sim_hardware_actuator_state_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hardware_actuator_state.py)
    add_test(
      NAME jetracer_sim_state_validation_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_state_validation.py)
    add_test(
      NAME jetracer_sim_deployment_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_deployment.py)
    add_test(
      NAME jetracer_sim_deployment_release_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_deployment_release.py)
    set_tests_properties(jetracer_sim_deployment_release_tests
      PROPERTIES TIMEOUT 120)
    add_test(
      NAME jetracer_sim_readiness_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_readiness.py)
    add_test(
      NAME jetracer_sim_bringup_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bringup.py)
    add_test(
      NAME jetracer_sim_clip_benchmark_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_clip_benchmark.py)
    add_test(
      NAME jetracer_sim_closed_loop_test
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/examples/closed_loop_lap.py)
    set_tests_properties(jetracer_sim_closed_loop_test PROPERTIES
      TIMEOUT ${JETRACER_SIM_EXAMPLE_TEST_TIMEOUT_SECONDS})
    add_test(
      NAME jetracer_sim_stop_braking_test
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/examples/stop_sign_braking.py)
    set_tests_properties(jetracer_sim_stop_braking_test PROPERTIES
      TIMEOUT ${JETRACER_SIM_EXAMPLE_TEST_TIMEOUT_SECONDS})
  endif()
endif()

install(TARGETS jetracer_sim_core ARCHIVE DESTINATION lib)
if(JETRACER_SIM_BUILD_NATIVE_CLI)
  install(TARGETS jetracer-sim RUNTIME DESTINATION bin)
endif()
install(DIRECTORY include/ DESTINATION include)
install(FILES ${JETRACER_SIM_GENERATED_DEFAULTS_HEADER}
  DESTINATION include/jetracer_sim)
install(FILES
  configs/camera_characterization.json
  configs/deployment.json
  configs/driving_benchmarks.json
  configs/jetson_software_baseline.json
  configs/road_segmentation_models.json
  configs/runtime_defaults.json
  configs/stop_sign_benchmark.json
  configs/stop_sign_latency_profiles.json
  DESTINATION share/jetracer_sim/configs)
install(
  DIRECTORY configs/hardware/
  DESTINATION share/jetracer_sim/configs/hardware
  FILES_MATCHING PATTERN "*.json")
install(FILES ${JETRACER_SIM_NATIVE_CONFIG}
  DESTINATION share/jetracer_sim/configs
  RENAME native_simulator_defaults.json)
