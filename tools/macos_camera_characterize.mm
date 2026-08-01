#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using Arguments = std::unordered_map<std::string, std::string>;

struct ParsedArguments {
  std::string command;
  Arguments values;
};

ParsedArguments parse_arguments(int argc, const char* argv[]) {
  if (argc < 2) throw std::invalid_argument("expected list or capture");
  ParsedArguments parsed{argv[1], {}};
  for (int index = 2; index < argc; index += 2) {
    if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
      throw std::invalid_argument("expected --name value arguments");
    }
    const std::string key = argv[index];
    if (parsed.values.contains(key)) {
      throw std::invalid_argument("duplicate argument " + key);
    }
    parsed.values.emplace(key, argv[index + 1]);
  }
  return parsed;
}

const std::string& required(const Arguments& arguments,
                            const std::string& name) {
  const auto found = arguments.find(name);
  if (found == arguments.end()) {
    throw std::invalid_argument("missing required argument " + name);
  }
  return found->second;
}

std::string optional(const Arguments& arguments, const std::string& name) {
  const auto found = arguments.find(name);
  return found == arguments.end() ? std::string{} : found->second;
}

bool boolean_value(const Arguments& arguments, const std::string& name) {
  const std::string& text = required(arguments, name);
  if (text == "true") return true;
  if (text == "false") return false;
  throw std::invalid_argument(name + " must be true or false");
}

int integer_value(const Arguments& arguments, const std::string& name) {
  const std::string& text = required(arguments, name);
  std::size_t consumed = 0;
  const int value = std::stoi(text, &consumed);
  if (consumed != text.size()) {
    throw std::invalid_argument(name + " must be an integer");
  }
  return value;
}

double double_value(const Arguments& arguments, const std::string& name) {
  const std::string& text = required(arguments, name);
  std::size_t consumed = 0;
  const double value = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::invalid_argument(name + " must be a finite number");
  }
  return value;
}

NSString* ns_string(const std::string& value) {
  return [NSString stringWithUTF8String:value.c_str()];
}

std::string std_string(NSString* value) {
  return value == nil ? std::string{} : std::string(value.UTF8String);
}

NSString* utc_timestamp() {
  NSISO8601DateFormatter* formatter = [[NSISO8601DateFormatter alloc] init];
  formatter.formatOptions = NSISO8601DateFormatWithInternetDateTime;
  return [formatter stringFromDate:[NSDate date]];
}

NSData* json_data(id value) {
  NSError* error = nil;
  NSData* data = [NSJSONSerialization dataWithJSONObject:value
                                                 options:NSJSONWritingPrettyPrinted |
                                                         NSJSONWritingSortedKeys |
                                                         NSJSONWritingWithoutEscapingSlashes
                                                   error:&error];
  if (data == nil) {
    throw std::runtime_error("failed to encode JSON: " +
                             std_string(error.localizedDescription));
  }
  NSMutableData* terminated = [data mutableCopy];
  const std::uint8_t newline = '\n';
  [terminated appendBytes:&newline length:1];
  return terminated;
}

NSString* fourcc(FourCharCode value) {
  char bytes[5] = {
      static_cast<char>((value >> 24) & 0xff),
      static_cast<char>((value >> 16) & 0xff),
      static_cast<char>((value >> 8) & 0xff),
      static_cast<char>(value & 0xff),
      '\0',
  };
  bool printable = true;
  for (int index = 0; index < 4; ++index) {
    const unsigned char character = static_cast<unsigned char>(bytes[index]);
    printable = printable && character >= 32 && character <= 126;
  }
  return printable ? [NSString stringWithUTF8String:bytes]
                   : [NSString stringWithFormat:@"0x%08x", value];
}

double fps_for_duration(CMTime duration) {
  if (duration.value <= 0 || duration.timescale <= 0) return 0.0;
  return static_cast<double>(duration.timescale) /
         static_cast<double>(duration.value);
}

void authorize_camera(bool request_permission) {
  switch ([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo]) {
    case AVAuthorizationStatusAuthorized:
      return;
    case AVAuthorizationStatusNotDetermined: {
      if (!request_permission) {
        throw std::runtime_error(
            "camera permission is not determined; unlock the Mac and rerun "
            "with --request-permission");
      }
      dispatch_semaphore_t completed = dispatch_semaphore_create(0);
      __block BOOL granted = NO;
      [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                              completionHandler:^(BOOL allowed) {
                                granted = allowed;
                                dispatch_semaphore_signal(completed);
                              }];
      dispatch_semaphore_wait(completed, DISPATCH_TIME_FOREVER);
      if (!granted) {
        throw std::runtime_error("camera permission was not granted");
      }
      return;
    }
    case AVAuthorizationStatusDenied:
      throw std::runtime_error("camera permission is denied for this terminal");
    case AVAuthorizationStatusRestricted:
      throw std::runtime_error("camera access is restricted");
  }
}

NSArray<AVCaptureDevice*>* discovered_devices(bool include_built_in) {
  NSMutableArray<AVCaptureDeviceType>* types = [NSMutableArray array];
  if (@available(macOS 14.0, *)) {
    [types addObject:AVCaptureDeviceTypeExternal];
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [types addObject:AVCaptureDeviceTypeExternalUnknown];
#pragma clang diagnostic pop
  }
  if (include_built_in) {
    [types addObject:AVCaptureDeviceTypeBuiltInWideAngleCamera];
  }
  AVCaptureDeviceDiscoverySession* discovery =
      [AVCaptureDeviceDiscoverySession
          discoverySessionWithDeviceTypes:types
                                 mediaType:AVMediaTypeVideo
                                  position:AVCaptureDevicePositionUnspecified];
  NSMutableArray<AVCaptureDevice*>* unique = [NSMutableArray array];
  NSMutableSet<NSString*>* identifiers = [NSMutableSet set];
  for (AVCaptureDevice* device in discovery.devices) {
    if (![identifiers containsObject:device.uniqueID]) {
      [identifiers addObject:device.uniqueID];
      [unique addObject:device];
    }
  }
  return unique;
}

NSDictionary* frame_rate_range_record(AVFrameRateRange* range) {
  return @{
    @"minimum_fps" : @(range.minFrameRate),
    @"maximum_fps" : @(range.maxFrameRate),
    @"minimum_frame_duration_value" : @(range.minFrameDuration.value),
    @"minimum_frame_duration_timescale" : @(range.minFrameDuration.timescale),
    @"maximum_frame_duration_value" : @(range.maxFrameDuration.value),
    @"maximum_frame_duration_timescale" : @(range.maxFrameDuration.timescale),
  };
}

NSDictionary* format_record(AVCaptureDeviceFormat* format) {
  const CMVideoDimensions dimensions =
      CMVideoFormatDescriptionGetDimensions(format.formatDescription);
  NSMutableArray* ranges = [NSMutableArray array];
  for (AVFrameRateRange* range in format.videoSupportedFrameRateRanges) {
    [ranges addObject:frame_rate_range_record(range)];
  }
  return @{
    @"width" : @(dimensions.width),
    @"height" : @(dimensions.height),
    @"media_subtype" :
        fourcc(CMFormatDescriptionGetMediaSubType(format.formatDescription)),
    @"frame_rate_ranges" : ranges,
  };
}

NSDictionary* device_record(AVCaptureDevice* device) {
  NSMutableArray* formats = [NSMutableArray array];
  for (AVCaptureDeviceFormat* format in device.formats) {
    [formats addObject:format_record(format)];
  }
  return @{
    @"name" : device.localizedName,
    @"unique_id" : device.uniqueID,
    @"model_id" : device.modelID,
    @"manufacturer" : device.manufacturer,
    @"device_type" : device.deviceType,
    @"connected" : @(device.connected),
    @"in_use_by_another_application" : @(device.inUseByAnotherApplication),
    @"formats" : formats,
  };
}

@interface JRFormatSelection : NSObject
@property(nonatomic, strong) AVCaptureDeviceFormat* format;
@property(nonatomic, strong) AVFrameRateRange* range;
@property(nonatomic) CMTime duration;
@end

@implementation JRFormatSelection
@end

JRFormatSelection* choose_format(AVCaptureDevice* device, int width, int height,
                                 double requested_fps) {
  const double tolerance = std::max(requested_fps * 1e-6, 1e-6);
  JRFormatSelection* best = nil;
  double best_distance = INFINITY;
  for (AVCaptureDeviceFormat* format in device.formats) {
    const CMVideoDimensions dimensions =
        CMVideoFormatDescriptionGetDimensions(format.formatDescription);
    if (dimensions.width != width || dimensions.height != height) continue;
    for (AVFrameRateRange* range in format.videoSupportedFrameRateRanges) {
      if (requested_fps + tolerance < range.minFrameRate ||
          requested_fps - tolerance > range.maxFrameRate) {
        continue;
      }
      const double distance =
          std::min(std::abs(requested_fps - range.minFrameRate),
                   std::abs(requested_fps - range.maxFrameRate));
      if (distance < best_distance) {
        best_distance = distance;
        best = [[JRFormatSelection alloc] init];
        best.format = format;
        best.range = range;
        if (std::abs(requested_fps - range.maxFrameRate) <= tolerance) {
          best.duration = range.minFrameDuration;
        } else if (std::abs(requested_fps - range.minFrameRate) <= tolerance) {
          best.duration = range.maxFrameDuration;
        } else {
          best.duration = CMTimeMakeWithSeconds(1.0 / requested_fps, 1000000000);
        }
      }
    }
  }
  if (best == nil) {
    throw std::runtime_error("device does not advertise " +
                             std::to_string(width) + "x" +
                             std::to_string(height) + " at " +
                             std::to_string(requested_fps) +
                             " FPS; run list first");
  }
  return best;
}

NSDictionary* selected_capture_record(JRFormatSelection* selection) {
  const CMVideoDimensions dimensions =
      CMVideoFormatDescriptionGetDimensions(selection.format.formatDescription);
  return @{
    @"width" : @(dimensions.width),
    @"height" : @(dimensions.height),
    @"media_subtype" : fourcc(CMFormatDescriptionGetMediaSubType(
        selection.format.formatDescription)),
    @"advertised_minimum_fps" : @([selection.range minFrameRate]),
    @"advertised_maximum_fps" : @([selection.range maxFrameRate]),
    @"configured_frame_duration_value" : @(selection.duration.value),
    @"configured_frame_duration_timescale" : @(selection.duration.timescale),
    @"configured_fps" : @(fps_for_duration(selection.duration)),
  };
}

NSDictionary* active_capture_record(AVCaptureDevice* device) {
  AVCaptureDeviceFormat* format = device.activeFormat;
  const CMVideoDimensions dimensions =
      CMVideoFormatDescriptionGetDimensions(format.formatDescription);
  const CMTime duration = device.activeVideoMinFrameDuration;
  return @{
    @"width" : @(dimensions.width),
    @"height" : @(dimensions.height),
    @"media_subtype" :
        fourcc(CMFormatDescriptionGetMediaSubType(format.formatDescription)),
    @"frame_duration_value" : @(duration.value),
    @"frame_duration_timescale" : @(duration.timescale),
    @"fps" : @(fps_for_duration(duration)),
  };
}

OSType pixel_format(const std::string& name) {
  if (name == "420v") return kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
  if (name == "420f") return kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;
  if (name == "BGRA") return kCVPixelFormatType_32BGRA;
  throw std::invalid_argument("unsupported output pixel format " + name);
}

@interface JRMovieRecorder : NSObject {
 @private
  NSURL* _url;
  NSString* _codec;
  NSInteger _bitrate;
  NSInteger _keyframeInterval;
  NSTimeInterval _finishTimeout;
  double _nominalFPS;
  AVAssetWriter* _writer;
  AVAssetWriterInput* _input;
  NSString* _errorMessage;
  NSInteger _appendedFrames;
  NSInteger _failedAppends;
}
- (instancetype)initWithURL:(NSURL*)url
                      codec:(NSString*)codec
                    bitrate:(NSInteger)bitrate
           keyframeInterval:(NSInteger)keyframeInterval
              finishTimeout:(NSTimeInterval)finishTimeout
                 nominalFPS:(double)nominalFPS;
- (void)appendSampleBuffer:(CMSampleBufferRef)sampleBuffer;
- (NSDictionary*)finish;
@end

@implementation JRMovieRecorder

- (instancetype)initWithURL:(NSURL*)url
                      codec:(NSString*)codec
                    bitrate:(NSInteger)bitrate
           keyframeInterval:(NSInteger)keyframeInterval
              finishTimeout:(NSTimeInterval)finishTimeout
                 nominalFPS:(double)nominalFPS {
  self = [super init];
  if (self) {
    _url = url;
    _codec = codec;
    _bitrate = bitrate;
    _keyframeInterval = keyframeInterval;
    _finishTimeout = finishTimeout;
    _nominalFPS = nominalFPS;
  }
  return self;
}

- (BOOL)startWithSampleBuffer:(CMSampleBufferRef)sampleBuffer {
  if ([[NSFileManager defaultManager] fileExistsAtPath:_url.path]) {
    _errorMessage = [NSString
        stringWithFormat:@"refusing to overwrite video %@", _url.path];
    return NO;
  }
  CMFormatDescriptionRef format =
      CMSampleBufferGetFormatDescription(sampleBuffer);
  if (format == nil) {
    _errorMessage = @"first video sample has no format description";
    return NO;
  }
  const CMVideoDimensions dimensions =
      CMVideoFormatDescriptionGetDimensions(format);
  AVVideoCodecType codec_type = nil;
  if ([_codec isEqualToString:@"h264"]) {
    codec_type = AVVideoCodecTypeH264;
  } else if ([_codec isEqualToString:@"hevc"]) {
    codec_type = AVVideoCodecTypeHEVC;
  } else {
    _errorMessage = [NSString stringWithFormat:@"unsupported codec %@", _codec];
    return NO;
  }
  NSDictionary* settings = @{
    AVVideoCodecKey : codec_type,
    AVVideoWidthKey : @(dimensions.width),
    AVVideoHeightKey : @(dimensions.height),
    AVVideoCompressionPropertiesKey : @{
      AVVideoAverageBitRateKey : @(_bitrate),
      AVVideoExpectedSourceFrameRateKey : @(llround(_nominalFPS)),
      AVVideoMaxKeyFrameIntervalKey : @(_keyframeInterval),
      AVVideoAllowFrameReorderingKey : @NO,
    },
  };
  NSError* error = nil;
  _writer = [[AVAssetWriter alloc] initWithURL:_url
                                      fileType:AVFileTypeQuickTimeMovie
                                         error:&error];
  if (_writer == nil) {
    _errorMessage = error.localizedDescription;
    return NO;
  }
  _input = [[AVAssetWriterInput alloc] initWithMediaType:AVMediaTypeVideo
                                         outputSettings:settings
                                       sourceFormatHint:format];
  _input.expectsMediaDataInRealTime = YES;
  if (![_writer canAddInput:_input]) {
    _errorMessage = @"movie writer rejected the video input";
    return NO;
  }
  [_writer addInput:_input];
  if (![_writer startWriting]) {
    NSString* writer_error = _writer.error.localizedDescription;
    _errorMessage =
        writer_error != nil ? writer_error : @"movie writer failed to start";
    return NO;
  }
  [_writer startSessionAtSourceTime:CMSampleBufferGetPresentationTimeStamp(
                                        sampleBuffer)];
  return YES;
}

- (void)appendSampleBuffer:(CMSampleBufferRef)sampleBuffer {
  if (_errorMessage != nil) return;
  if (_writer == nil && ![self startWithSampleBuffer:sampleBuffer]) return;
  if (_writer.status != AVAssetWriterStatusWriting) {
    NSString* writer_error = _writer.error.localizedDescription;
    _errorMessage = writer_error != nil ? writer_error : @"movie writer stopped";
    return;
  }
  if (_input.readyForMoreMediaData) {
    if ([_input appendSampleBuffer:sampleBuffer]) {
      ++_appendedFrames;
    } else {
      ++_failedAppends;
    }
  } else {
    ++_failedAppends;
  }
}

- (NSDictionary*)finish {
  if (_writer != nil) {
    [_input markAsFinished];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    [_writer finishWritingWithCompletionHandler:^{
      dispatch_semaphore_signal(completed);
    }];
    const int64_t timeout_nanoseconds =
        static_cast<int64_t>(_finishTimeout * NSEC_PER_SEC);
    if (dispatch_semaphore_wait(
            completed,
            dispatch_time(DISPATCH_TIME_NOW, timeout_nanoseconds)) != 0) {
      _errorMessage = [NSString
          stringWithFormat:@"movie writer did not finish within %.3f seconds",
                           _finishTimeout];
      [_writer cancelWriting];
    } else if (_writer.status != AVAssetWriterStatusCompleted) {
      NSString* writer_error = _writer.error.localizedDescription;
      _errorMessage =
          writer_error != nil ? writer_error : @"movie writer did not complete";
    }
  }
  NSString* status = nil;
  if (_errorMessage != nil) {
    status = @"failed";
  } else if (_writer == nil) {
    status = @"no_frames";
  } else {
    status = @"completed";
  }
  return @{
    @"path" : _url.path,
    @"codec" : _codec,
    @"average_bitrate_bps" : @(_bitrate),
    @"appended_frames" : @(_appendedFrames),
    @"failed_appends" : @(_failedAppends),
    @"status" : status,
    @"error" : _errorMessage != nil ? _errorMessage : [NSNull null],
  };
}

@end

@interface JRCaptureSnapshot : NSObject {
 @public
  double measurementStartedAt;
  double measurementStoppedAt;
  std::vector<double> arrivalTimes;
  std::vector<double> presentationTimes;
  NSInteger droppedCallbacks;
  NSDictionary* dropReasons;
  JRMovieRecorder* recorder;
}
@end

@implementation JRCaptureSnapshot
@end

@interface JRCaptureCollector
    : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate> {
 @private
  BOOL _measuring;
  double _measurementStartedAt;
  std::vector<double> _arrivalTimes;
  std::vector<double> _presentationTimes;
  NSInteger _droppedCallbacks;
  NSMutableDictionary<NSString*, NSNumber*>* _dropReasons;
  JRMovieRecorder* _recorder;
}
- (void)startWithRecorder:(JRMovieRecorder*)recorder;
- (JRCaptureSnapshot*)stopAndSnapshot;
@end

@implementation JRCaptureCollector

- (void)startWithRecorder:(JRMovieRecorder*)recorder {
  _measuring = YES;
  _measurementStartedAt = NSProcessInfo.processInfo.systemUptime;
  _arrivalTimes.clear();
  _presentationTimes.clear();
  _droppedCallbacks = 0;
  _dropReasons = [NSMutableDictionary dictionary];
  _recorder = recorder;
}

- (JRCaptureSnapshot*)stopAndSnapshot {
  _measuring = NO;
  JRCaptureSnapshot* snapshot = [[JRCaptureSnapshot alloc] init];
  snapshot->measurementStartedAt = _measurementStartedAt;
  snapshot->measurementStoppedAt = NSProcessInfo.processInfo.systemUptime;
  snapshot->arrivalTimes = _arrivalTimes;
  snapshot->presentationTimes = _presentationTimes;
  snapshot->droppedCallbacks = _droppedCallbacks;
  NSDictionary* reasons = [_dropReasons copy];
  snapshot->dropReasons = reasons != nil ? reasons : @{};
  snapshot->recorder = _recorder;
  return snapshot;
}

- (void)captureOutput:(AVCaptureOutput*)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection {
  if (!_measuring) return;
  _arrivalTimes.push_back(NSProcessInfo.processInfo.systemUptime);
  const double timestamp = CMTimeGetSeconds(
      CMSampleBufferGetPresentationTimeStamp(sampleBuffer));
  if (std::isfinite(timestamp)) _presentationTimes.push_back(timestamp);
  [_recorder appendSampleBuffer:sampleBuffer];
}

- (void)captureOutput:(AVCaptureOutput*)output
    didDropSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection*)connection {
  if (!_measuring) return;
  ++_droppedCallbacks;
  CFTypeRef reason = CMGetAttachment(
      sampleBuffer, kCMSampleBufferAttachmentKey_DroppedFrameReason, nullptr);
  NSString* label = reason == nil ? @"unknown" : [(__bridge id)reason description];
  _dropReasons[label] = @([_dropReasons[label] integerValue] + 1);
}

@end

std::vector<double> intervals(const std::vector<double>& timestamps) {
  std::vector<double> result;
  if (timestamps.size() < 2) return result;
  result.reserve(timestamps.size() - 1);
  for (std::size_t index = 1; index < timestamps.size(); ++index) {
    result.push_back(timestamps[index] - timestamps[index - 1]);
  }
  return result;
}

double percentile(const std::vector<double>& sorted, double quantile) {
  const std::size_t index = static_cast<std::size_t>(
      std::llround(static_cast<double>(sorted.size() - 1) * quantile));
  return sorted[index];
}

NSDictionary* interval_statistics(const std::vector<double>& seconds) {
  if (seconds.empty()) return nil;
  std::vector<double> milliseconds;
  milliseconds.reserve(seconds.size());
  for (double value : seconds) milliseconds.push_back(value * 1000.0);
  std::sort(milliseconds.begin(), milliseconds.end());
  double sum = 0.0;
  for (double value : milliseconds) sum += value;
  const double mean = sum / static_cast<double>(milliseconds.size());
  double squared_difference = 0.0;
  for (double value : milliseconds) {
    const double difference = value - mean;
    squared_difference += difference * difference;
  }
  const double standard_deviation =
      std::sqrt(squared_difference / static_cast<double>(milliseconds.size()));
  return @{
    @"count" : @(milliseconds.size()),
    @"mean_milliseconds" : @(mean),
    @"standard_deviation_milliseconds" : @(standard_deviation),
    @"minimum_milliseconds" : @(milliseconds.front()),
    @"p50_milliseconds" : @(percentile(milliseconds, 0.50)),
    @"p95_milliseconds" : @(percentile(milliseconds, 0.95)),
    @"p99_milliseconds" : @(percentile(milliseconds, 0.99)),
    @"maximum_milliseconds" : @(milliseconds.back()),
  };
}

NSDictionary* measurement_statistics(JRCaptureSnapshot* snapshot,
                                      double requested_duration,
                                      double nominal_fps) {
  const std::vector<double> arrival_intervals = intervals(snapshot->arrivalTimes);
  const std::vector<double> pts_intervals = intervals(snapshot->presentationTimes);
  const double target_period = 1.0 / nominal_fps;
  NSInteger inferred_missing = 0;
  NSInteger non_monotonic = 0;
  for (double interval : pts_intervals) {
    if (interval <= 0.0) {
      ++non_monotonic;
    } else {
      inferred_missing +=
          std::max(static_cast<long long>(std::llround(interval / target_period)) -
                       1LL,
                   0LL);
    }
  }
  double delivered_fps = 0.0;
  if (snapshot->arrivalTimes.size() >= 2 &&
      snapshot->arrivalTimes.back() > snapshot->arrivalTimes.front()) {
    delivered_fps =
        static_cast<double>(snapshot->arrivalTimes.size() - 1) /
        (snapshot->arrivalTimes.back() - snapshot->arrivalTimes.front());
  }
  NSDictionary* arrival_stats = interval_statistics(arrival_intervals);
  NSDictionary* pts_stats = interval_statistics(pts_intervals);
  return @{
    @"requested_duration_seconds" : @(requested_duration),
    @"observed_duration_seconds" :
        @(snapshot->measurementStoppedAt - snapshot->measurementStartedAt),
    @"received_frames" : @(snapshot->arrivalTimes.size()),
    @"dropped_frame_callbacks" : @(snapshot->droppedCallbacks),
    @"inferred_missing_frames" : @(inferred_missing),
    @"non_monotonic_presentation_timestamps" : @(non_monotonic),
    @"delivered_fps" : @(delivered_fps),
    @"arrival_intervals" :
        arrival_stats != nil ? arrival_stats : [NSNull null],
    @"presentation_timestamp_intervals" :
        pts_stats != nil ? pts_stats : [NSNull null],
    @"drop_reasons" : snapshot->dropReasons != nil ? snapshot->dropReasons : @{},
  };
}

AVCaptureDevice* choose_device(bool allow_built_in,
                               const std::string& requested_id,
                               const std::string& requested_name) {
  NSArray<AVCaptureDevice*>* devices = discovered_devices(allow_built_in);
  if (!requested_id.empty()) {
    for (AVCaptureDevice* device in devices) {
      if ([device.uniqueID isEqualToString:ns_string(requested_id)]) return device;
    }
    throw std::runtime_error("camera device ID was not found: " + requested_id);
  }
  if (!requested_name.empty()) {
    NSMutableArray<AVCaptureDevice*>* matches = [NSMutableArray array];
    for (AVCaptureDevice* device in devices) {
      if ([device.localizedName
              rangeOfString:ns_string(requested_name)
                     options:NSCaseInsensitiveSearch]
              .location != NSNotFound) {
        [matches addObject:device];
      }
    }
    if (matches.count == 1) return matches.firstObject;
    throw std::runtime_error(matches.count == 0
                                 ? "no camera name contains " + requested_name
                                 : "camera name is ambiguous: " + requested_name);
  }
  if (devices.count == 0) {
    throw std::runtime_error(
        allow_built_in
            ? "no camera was found"
            : "no external camera was found; refusing to benchmark the built-in camera");
  }
  return devices.firstObject;
}

int run_list(const Arguments& arguments) {
  const bool request_permission =
      boolean_value(arguments, "--request-permission");
  const bool include_built_in =
      boolean_value(arguments, "--include-built-in");
  authorize_camera(request_permission);
  NSMutableArray* records = [NSMutableArray array];
  for (AVCaptureDevice* device in discovered_devices(include_built_in)) {
    [records addObject:device_record(device)];
  }
  NSDictionary* report = @{
    @"schema_version" : @1,
    @"recorded_at_utc" : utc_timestamp(),
    @"devices" : records,
  };
  [NSFileHandle.fileHandleWithStandardOutput writeData:json_data(report)];
  return EXIT_SUCCESS;
}

int run_capture(const Arguments& arguments) {
  const bool request_permission =
      boolean_value(arguments, "--request-permission");
  const bool allow_built_in = boolean_value(arguments, "--allow-built-in");
  const int width = integer_value(arguments, "--width");
  const int height = integer_value(arguments, "--height");
  const double requested_fps = double_value(arguments, "--fps");
  const double duration = double_value(arguments, "--duration");
  const double warmup = double_value(arguments, "--warmup");
  const std::string output_pixel_format =
      required(arguments, "--pixel-format");
  const bool discard_late_frames =
      boolean_value(arguments, "--discard-late-frames");
  const std::string report_path = required(arguments, "--report");
  const std::string video_path = optional(arguments, "--video");
  if (width <= 0 || height <= 0 || requested_fps <= 0.0 || duration <= 0.0 ||
      warmup < 0.0) {
    throw std::invalid_argument(
        "capture dimensions, FPS, and durations are invalid");
  }
  if ([[NSFileManager defaultManager]
          fileExistsAtPath:ns_string(report_path)]) {
    throw std::runtime_error("refusing to overwrite report " + report_path);
  }
  if (!video_path.empty() && [[NSFileManager defaultManager]
                                 fileExistsAtPath:ns_string(video_path)]) {
    throw std::runtime_error("refusing to overwrite video " + video_path);
  }

  authorize_camera(request_permission);
  AVCaptureDevice* device =
      choose_device(allow_built_in, optional(arguments, "--device-id"),
                    optional(arguments, "--device-name"));
  if (device.inUseByAnotherApplication) {
    throw std::runtime_error("camera is in use by another application");
  }
  JRFormatSelection* selection =
      choose_format(device, width, height, requested_fps);
  NSDictionary* before_start = selected_capture_record(selection);

  NSError* error = nil;
  if (![device lockForConfiguration:&error]) {
    throw std::runtime_error("cannot configure camera: " +
                             std_string(error.localizedDescription));
  }
  @try {
    device.activeFormat = selection.format;
    device.activeVideoMinFrameDuration = selection.duration;
    device.activeVideoMaxFrameDuration = selection.duration;
  } @finally {
    [device unlockForConfiguration];
  }

  AVCaptureSession* session = [[AVCaptureSession alloc] init];
  [session beginConfiguration];
  AVCaptureDeviceInput* input =
      [[AVCaptureDeviceInput alloc] initWithDevice:device error:&error];
  if (input == nil || ![session canAddInput:input]) {
    throw std::runtime_error("capture session rejected the camera input: " +
                             std_string(error.localizedDescription));
  }
  [session addInput:input];
  AVCaptureVideoDataOutput* output = [[AVCaptureVideoDataOutput alloc] init];
  output.alwaysDiscardsLateVideoFrames = discard_late_frames;
  output.videoSettings = @{
    (__bridge NSString*)kCVPixelBufferPixelFormatTypeKey :
        @(pixel_format(output_pixel_format)),
  };
  if (![session canAddOutput:output]) {
    throw std::runtime_error("capture session rejected the video output");
  }
  [session addOutput:output];
  dispatch_queue_t queue = dispatch_queue_create(
      "jetracer.camera.characterization", DISPATCH_QUEUE_SERIAL);
  JRCaptureCollector* collector = [[JRCaptureCollector alloc] init];
  [output setSampleBufferDelegate:collector queue:queue];
  [session commitConfiguration];

  [session startRunning];
  if (!session.running) {
    throw std::runtime_error("capture session did not start");
  }
  NSDictionary* after_start = active_capture_record(device);
  const double active_fps = [after_start[@"fps"] doubleValue];
  if (warmup > 0.0) [NSThread sleepForTimeInterval:warmup];

  JRMovieRecorder* recorder = nil;
  if (!video_path.empty()) {
    const int bitrate = integer_value(arguments, "--bitrate");
    const int keyframe_interval =
        integer_value(arguments, "--keyframe-interval");
    const double finish_timeout =
        double_value(arguments, "--finish-timeout");
    if (bitrate <= 0 || keyframe_interval <= 0 || finish_timeout <= 0.0) {
      throw std::invalid_argument("recording settings must be positive");
    }
    recorder = [[JRMovieRecorder alloc]
             initWithURL:[NSURL fileURLWithPath:ns_string(video_path)]
                   codec:ns_string(required(arguments, "--codec"))
                 bitrate:bitrate
        keyframeInterval:keyframe_interval
           finishTimeout:finish_timeout
              nominalFPS:active_fps > 0.0 ? active_fps : requested_fps];
  }
  dispatch_sync(queue, ^{
    [collector startWithRecorder:recorder];
  });
  [NSThread sleepForTimeInterval:duration];
  [session stopRunning];
  __block JRCaptureSnapshot* snapshot = nil;
  dispatch_sync(queue, ^{
    snapshot = [collector stopAndSnapshot];
  });
  NSDictionary* recording = [snapshot->recorder finish];

  NSDictionary* report = @{
    @"schema_version" : @1,
    @"recorded_at_utc" : utc_timestamp(),
    @"device" : device_record(device),
    @"requested" : @{
      @"width" : @(width),
      @"height" : @(height),
      @"fps" : @(requested_fps),
      @"duration_seconds" : @(duration),
      @"warmup_seconds" : @(warmup),
      @"output_pixel_format" : ns_string(output_pixel_format),
      @"discards_late_frames" : @(discard_late_frames),
    },
    @"selected_before_start" : before_start,
    @"active_after_start" : after_start,
    @"measurement" : measurement_statistics(
        snapshot, duration, active_fps > 0.0 ? active_fps : requested_fps),
    @"recording" : recording != nil ? recording : [NSNull null],
  };
  NSData* encoded = json_data(report);
  NSURL* report_url = [NSURL fileURLWithPath:ns_string(report_path)];
  if (![encoded writeToURL:report_url options:NSDataWritingAtomic error:&error]) {
    throw std::runtime_error("failed to write report: " +
                             std_string(error.localizedDescription));
  }
  [NSFileHandle.fileHandleWithStandardOutput writeData:encoded];
  if (recording != nil && [recording[@"status"] isEqualToString:@"failed"]) {
    throw std::runtime_error(
        "recording failed: " + std_string(recording[@"error"]));
  }
  return EXIT_SUCCESS;
}

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    @try {
      try {
        const ParsedArguments arguments = parse_arguments(argc, argv);
        if (arguments.command == "list") return run_list(arguments.values);
        if (arguments.command == "capture") return run_capture(arguments.values);
        throw std::invalid_argument("expected list or capture");
      } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return EXIT_FAILURE;
      }
    } @catch (NSException* exception) {
      std::fprintf(stderr, "error: AVFoundation exception %s: %s\n",
                   exception.name.UTF8String, exception.reason.UTF8String);
      return EXIT_FAILURE;
    }
  }
}
