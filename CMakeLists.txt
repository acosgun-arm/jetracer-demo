cmake_minimum_required(VERSION 3.22)
project(jetracer_sim VERSION 0.1.0 LANGUAGES CXX)

if(APPLE)
  enable_language(OBJCXX)
endif()

option(JETRACER_SIM_BUILD_PYTHON "Build the pybind11 extension" ON)
option(JETRACER_SIM_BUILD_TESTS "Build native tests" ON)
option(JETRACER_SIM_BUILD_NATIVE_CLI
       "Build the native CLI and its interactive OpenCV dependency" ON)
set(JETRACER_SIM_EXAMPLE_TEST_TIMEOUT_SECONDS 60 CACHE STRING
    "Timeout for simulator example tests run by CTest")
set(JETRACER_SIM_NATIVE_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/configs/native_simulator_defaults.json"
    CACHE FILEPATH "Native simulator defaults JSON used to generate C++ constants")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter REQUIRED)
set(JETRACER_SIM_OPENCV_COMPONENTS core imgproc imgcodecs)
if(JETRACER_SIM_BUILD_NATIVE_CLI)
  list(APPEND JETRACER_SIM_OPENCV_COMPONENTS highgui)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${JETRACER_SIM_OPENCV_COMPONENTS})
if(OpenCV_VERSION VERSION_LESS 4.5)
  message(FATAL_ERROR "JetRacer simulator requires OpenCV 4.5 or newer")
endif()

set(JETRACER_SIM_GENERATED_INCLUDE_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/generated/include)
set(JETRACER_SIM_GENERATED_DEFAULTS_HEADER
    ${JETRACER_SIM_GENERATED_INCLUDE_DIR}/jetracer_sim/native_defaults.hpp)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${JETRACER_SIM_NATIVE_CONFIG}
  ${CMAKE_CURRENT_SOURCE_DIR}/configs/driving_benchmarks.json
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_native_defaults.py)
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
target_link_libraries(jetracer_sim_core PUBLIC ${OpenCV_LIBS})
target_compile_options(jetracer_sim_core PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>)

if(JETRACER_SIM_BUILD_NATIVE_CLI)
  add_executable(jetracer-sim src/cli.cpp)
  target_link_libraries(jetracer-sim PRIVATE jetracer_sim_core)
  if(APPLE)
    target_link_libraries(jetracer-sim PRIVATE "-framework ApplicationServices")
  endif()
endif()

if(JETRACER_SIM_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native src/bindings.cpp)
  target_link_libraries(_native PRIVATE jetracer_sim_core)
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
      benchmarking.py
      bringup.py
      clip_benchmark.py
      configuration.py
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
      platform_runtime.py
      pretrained.py
      process_safety.py
      readiness.py
      realtime.py
      realtime_presentation.py
      resource_paths.py
      state_validation.py
      stopping.py
      synthetic_clip.py
      tracks.py
      vehicle_io.py)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/python/jetracer_sim/${python_file}
      ${JETRACER_SIM_PYTHON_BUILD_DIR}/${python_file}
      COPYONLY)
  endforeach()
  file(MAKE_DIRECTORY ${JETRACER_SIM_PYTHON_BUILD_DIR}/configs)
  foreach(config_file IN ITEMS
      deployment.json
      driving_benchmarks.json
      jetson_software_baseline.json
      runtime_defaults.json)
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
    ${JETRACER_SIM_PYTHON_BUILD_DIR}/benchmarks)
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
  install(TARGETS _native LIBRARY DESTINATION jetracer_sim)
  install(
    DIRECTORY python/jetracer_sim/
    DESTINATION jetracer_sim
    FILES_MATCHING PATTERN "*.py")
  install(
    FILES
      configs/deployment.json
      configs/driving_benchmarks.json
      configs/jetson_software_baseline.json
      configs/runtime_defaults.json
    DESTINATION jetracer_sim/configs)
  install(
    FILES ${JETRACER_SIM_NATIVE_CONFIG}
    DESTINATION jetracer_sim/configs
    RENAME native_simulator_defaults.json)
  install(
    FILES configs/off_the_shelf_models.json
    DESTINATION jetracer_sim/configs)
  install(
    DIRECTORY configs/platforms/
    DESTINATION jetracer_sim/configs/platforms
    FILES_MATCHING PATTERN "*.json")
  install(
    DIRECTORY configs/hardware/
    DESTINATION jetracer_sim/configs/hardware
    FILES_MATCHING PATTERN "*.json")
  install(
    FILES benchmarks/off_the_shelf_model_benchmarks.json
    DESTINATION jetracer_sim/benchmarks)
endif()

if(JETRACER_SIM_BUILD_TESTS)
  enable_testing()
  add_executable(jetracer_sim_tests tests/test_simulator.cpp)
  target_link_libraries(jetracer_sim_tests PRIVATE jetracer_sim_core)
  add_test(NAME jetracer_sim_tests COMMAND jetracer_sim_tests)
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
    add_test(
      NAME jetracer_sim_controller_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_controller.py)
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
    set_tests_properties(jetracer_sim_driving_benchmark_tests PROPERTIES TIMEOUT 30)
    add_test(
      NAME jetracer_sim_configuration_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python"
              ${Python_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_configuration.py)
    set_tests_properties(jetracer_sim_configuration_tests PROPERTIES TIMEOUT 30)
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
  configs/runtime_defaults.json
  DESTINATION share/jetracer_sim/configs)
install(
  DIRECTORY configs/hardware/
  DESTINATION share/jetracer_sim/configs/hardware
  FILES_MATCHING PATTERN "*.json")
install(FILES ${JETRACER_SIM_NATIVE_CONFIG}
  DESTINATION share/jetracer_sim/configs
  RENAME native_simulator_defaults.json)
