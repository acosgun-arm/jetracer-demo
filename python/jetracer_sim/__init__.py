"""High-rate camera simulator for JetRacer perception development."""

from ._native import (  # noqa: F401
    CameraProfile,
    Detection,
    Frame,
    LensModel,
    ObjectType,
    PixelFormat,
    Point2,
    Pose2D,
    Scene,
    SceneConfig,
    SceneObject,
    SCENE_SCHEMA_VERSION,
    SemanticClass,
    ShutterType,
    Simulator,
    VehicleCommand,
    VehicleConfig,
    VehicleState,
)
from .configuration import (  # noqa: F401
    CONFIGURATION_SCHEMA_VERSION,
    DEFAULT_DRIVING_BENCHMARK_CONFIG_PATH,
    DEFAULT_NATIVE_SIMULATOR_CONFIG_PATH,
    DEFAULT_RUNTIME_CONFIG_PATH,
    DRIVING_CONFIG_ENVIRONMENT_VARIABLE,
    RUNTIME_CONFIG_ENVIRONMENT_VARIABLE,
    DrivingBenchmarkSuiteConfiguration,
    load_driving_benchmark_configuration,
    load_native_simulator_configuration,
    load_runtime_configuration,
    runtime_config_section,
)
from .governor import (  # noqa: F401
    GovernorConfig,
    GovernorDecision,
    LatencyAwareSpeedGovernor,
)
from .gui_safety import (  # noqa: F401
    UnsafeGuiRequestError,
    validate_gui_request,
)
from .frame_source import (  # noqa: F401
    CapturedFrame,
    FrameSource,
    FrameSourceError,
    FrameSourceStatistics,
    LatestFrameBuffer,
    OpenCVCameraConfig,
    OpenCVCameraFrameSource,
    RecordedVideoConfig,
    RecordedVideoFrameSource,
    ResolvedCameraMode,
    SimulatorFrameSource,
)
from .platform_runtime import (  # noqa: F401
    DEFAULT_PLATFORM_CONFIG_PATH,
    PLATFORM_CONFIGURATION_SCHEMA_VERSION,
    PLATFORM_CONFIG_ENVIRONMENT_VARIABLE,
    PlatformConfiguration,
    PlatformRuntime,
    create_platform_runtime,
    load_platform_configuration,
)
from .vehicle_io import (  # noqa: F401
    ActuatorLimits,
    DryRunVehicleActuator,
    CommandEstimatedVehicleStateSource,
    CommandSpeedEstimatorConfig,
    SimulatorVehicleActuator,
    SimulatorVehicleStateSource,
    UnavailableVehicleStateSource,
    VehicleActuator,
    VehicleActuatorStatus,
    VehicleStateSample,
    VehicleStateSource,
)
from .hardware_actuator import (  # noqa: F401
    ACTUATOR_PROFILE_SCHEMA_VERSION,
    DEFAULT_ACTUATOR_PROFILE_PATH,
    ActuatorOutput,
    ActuatorTransport,
    AxisCalibration,
    CalibratedHardwareVehicleActuator,
    HardwareActuatorProfile,
    RecordingActuatorTransport,
    load_hardware_actuator_profile,
)
from .state_validation import (  # noqa: F401
    DEFAULT_STATE_PROFILE_PATH,
    STATE_PROFILE_SCHEMA_VERSION,
    StateAcceptanceThresholds,
    StateValidationResult,
    VehicleStateProfile,
    evaluate_state_measurements,
    load_vehicle_state_profile,
)
from .deployment import (  # noqa: F401
    DEFAULT_DEPLOYMENT_POLICY_PATH,
    DEPLOYMENT_POLICY_SCHEMA_VERSION,
    DeploymentReport,
    DeploymentVariantStatus,
    RuntimeCapabilities,
    artifact_sha256,
    collect_runtime_capabilities,
    evaluate_deployment,
    filter_deployable_model_variants,
    load_deployment_policy,
)
from .deployment_config import (  # noqa: F401
    DEFAULT_DEPLOYMENT_CONFIGURATION_PATH,
    DEPLOYMENT_CONFIG_ENVIRONMENT_VARIABLE,
    DEPLOYMENT_CONFIGURATION_SCHEMA_VERSION,
    DeploymentConfiguration,
    load_deployment_configuration,
)
from .deployment_release import (  # noqa: F401
    RELEASE_MANIFEST_SCHEMA_VERSION,
    RUNTIME_STATE_SCHEMA_VERSION,
    WheelRecord,
    create_release,
    deployment_status,
    file_sha256,
    prepare_release,
    promote_release,
    release_id_from_link,
    render_systemd_unit as render_deployment_systemd_unit,
    rollback_release,
    verify_release,
)
from .deployment_supervisor import (  # noqa: F401
    DEPLOYMENT_STATUS_SCHEMA_VERSION,
    RUNTIME_PID_SCHEMA_VERSION,
    assess_standby as assess_deployment_standby,
    build_drive_command as build_deployed_drive_command,
    require_drive_authorization,
    run_drive as run_deployed_drive,
    run_standby as run_deployment_standby,
    safe_stop_runtime as safe_stop_deployed_runtime,
    write_deployment_status,
)
from .observability import (  # noqa: F401
    SystemHealthConfig,
    SystemHealthMonitor,
    SystemHealthSnapshot,
)
from .readiness import (  # noqa: F401
    DEFAULT_PREFLIGHT_CONFIG_PATH,
    PREFLIGHT_CONFIGURATION_SCHEMA_VERSION,
    PREFLIGHT_REPORT_SCHEMA_VERSION,
    HardwarePreflightReport,
    PreflightCheck,
    build_preflight_report,
    load_preflight_configuration,
    preflight_authorizes_motion,
    save_preflight_report,
)
from .bringup import (  # noqa: F401
    BRINGUP_PLAN_SCHEMA_VERSION,
    BRINGUP_STATE_SCHEMA_VERSION,
    DEFAULT_BRINGUP_PLAN_PATH,
    BringupPlan,
    BringupStage,
    active_bringup_stage,
    begin_bringup_stage,
    initialize_bringup_state,
    load_bringup_plan,
    load_bringup_state,
    record_bringup_stage,
)
from .process_safety import ShutdownSignalMonitor  # noqa: F401
from .hardware_profiles import (  # noqa: F401
    CAMERA_PROFILE_SCHEMA_VERSION,
    DEFAULT_CAMERA_PROFILE_PATH,
    CameraAcceptanceResult,
    CameraAcceptanceThresholds,
    CameraMode,
    PhysicalCameraProfile,
    evaluate_camera_measurement,
    load_camera_profiles,
)
from .controller import (  # noqa: F401
    RoadSteeringConfig,
    RoadSteeringController,
    SteeringDecision,
)
from .avoidance import (  # noqa: F401
    ObstacleAvoidanceConfig,
    ObstacleAvoidanceController,
    ObstacleAvoidanceDecision,
)
from .benchmarking import (  # noqa: F401
    DRIVING_BENCHMARK_SCHEMA_VERSION,
    DrivingBenchmarkConfig,
    DrivingBenchmarkResult,
    run_driving_benchmark,
    save_driving_benchmark_results,
)
from .clip_benchmark import (  # noqa: F401
    RECORDED_CLIP_BENCHMARK_SCHEMA_VERSION,
    RecordedClipBenchmarkConfig,
    RecordedClipModelResult,
    recorded_clip_report_to_model_benchmarks,
    run_recorded_clip_benchmark,
    save_recorded_clip_benchmark,
)
from .detection import (  # noqa: F401
    ApparentWidthRangeEstimator,
    DetectionAdapter,
    DetectionPipeline,
    ObjectDetection,
    TimedDetections,
    YoloConfig,
    YoloOnnxAdapter,
)
from .inference import (  # noqa: F401
    CallableSegmentationAdapter,
    InferenceMetrics,
    ModelMetadata,
    NumpyRoadSegmentationAdapter,
    NumpyRoadSegmentationConfig,
    SegmentationAdapter,
    SegmentationPipeline,
    SegmentationPrediction,
    TimedSegmentation,
)
from .onnx_adapters import (  # noqa: F401
    OnnxSegmentationAdapter,
    OnnxSegmentationConfig,
)
from .coreml_adapter import (  # noqa: F401
    COREML_VALIDATION_SCHEMA_VERSION,
    CoreMLSegmentationAdapter,
    CoreMLSegmentationConfig,
    coreml_artifact_sha256,
    validate_coreml_artifact,
)
from .realtime import (  # noqa: F401
    InferenceWorkerStatistics,
    LatencyInjectedSegmentationAdapter,
    LatestFrameDetectionWorker,
    LatestFrameSegmentationWorker,
    SemanticMaskSegmentationAdapter,
)
from .model_registry import (  # noqa: F401
    DetectionModelVariant,
    MODEL_REGISTRY_SCHEMA_VERSION,
    ModelBenchmark,
    ModelVariant,
    benchmark_environment,
    benchmark_detection_adapter,
    build_detection_adapter,
    benchmark_segmentation_adapter,
    build_segmentation_adapter,
    load_detection_model_variants,
    load_model_benchmarks,
    load_model_variants,
    save_model_benchmarks,
)
from .dataset import (  # noqa: F401
    DATASET_SCHEMA_VERSION,
    SEMANTIC_CLASSES,
    DatasetExportConfig,
    DatasetExportSummary,
    export_evaluation_dataset,
)
from .synthetic_clip import (  # noqa: F401
    SYNTHETIC_CLIP_SCHEMA_VERSION,
    SyntheticClipExportConfig,
    SyntheticClipExportSummary,
    export_synthetic_track_clip,
)
from .evaluation import (  # noqa: F401
    SEGMENTATION_EVALUATION_SCHEMA_VERSION,
    SegmentationEvaluationResult,
    evaluate_segmentation_clip,
    evaluate_segmentation_dataset,
    save_segmentation_evaluation,
)
from .pretrained import (  # noqa: F401
    DEFAULT_ROAD_SEGMENTATION_MODEL,
    HuggingFaceSegmentationAdapter,
    HuggingFaceSegmentationConfig,
    resolve_source_class_ids,
)
from .tracks import (  # noqa: F401
    WAVESHARE_JETRACER_PRODUCT_URL,
    TrackDefinition,
    benchmark_tracks,
    build_benchmark_scene,
    track_by_id,
)
from .stopping import (  # noqa: F401
    StopSignConfig,
    StopSignController,
    StopSignDecision,
    StopState,
)

__all__ = [name for name in globals() if not name.startswith("_")]
