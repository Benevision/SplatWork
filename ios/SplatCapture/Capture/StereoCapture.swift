import AVFoundation
import CoreImage
import Foundation
import ImageIO
import os
import simd

/// `manifest.json` of a stereo `.splatcapture` bundle (docs/PROTOCOL.md §8, `source` "iphone-stereo"): photo pairs from
/// the wide and ultra-wide cameras, taken together through the dual-wide virtual camera, each with the camera's factory
/// calibration. No ARKit poses — SplatWork solves the cameras; the pair's fixed offset gives the scene its scale.
struct StereoManifest: Codable, Sendable {
    static let source = "iphone-stereo"

    struct Sensor: Codable, Sendable {
        var id: String                      // "wide" | "ultrawide"
        var kind: String = "rgb"
        var width: Int
        var height: Int
        /// fx, fy, cx, cy in saved-image pixels (from the photo's calibration data), when the phone gave it.
        var intrinsics: [Float]?
        /// AVCameraCalibrationData.extrinsicMatrix as recorded: 3×4 row-major [R | t], t in millimetres, relative to the
        /// device's reference camera. SplatWork derives the rig (rotation and offset between the lenses) from the pair.
        var extrinsics: [Float]?
    }

    struct Frame: Codable, Sendable {
        var index: Int
        var timestamp: Double
        /// sensor id → image path in the bundle
        var images: [String: String]
        /// sensor id → fx, fy, cx, cy for this shot (focus moves them slightly)
        var intrinsics: [String: [Float]]?
        var exposureDuration: Double?
        var iso: Double?
        /// The dual camera's depth map for the wide image: float16 little-endian metres, row by row.
        var depth: String?
        var depthWidth: Int?
        var depthHeight: Int?
        /// fx, fy, cx, cy of the depth map, in depth-map pixels, and the extrinsics of the camera it is relative to.
        var depthIntrinsics: [Float]?
        var depthExtrinsics: [Float]?
    }

    var format = "splatcapture"
    var version = 2
    var source = StereoManifest.source
    var name: String
    var createdAt: Date
    var device: String
    var appVersion: String
    var hasLiDAR = false
    var imageWidth: Int
    var imageHeight: Int
    /// Whether iOS straightened the ultra-wide's lens distortion in the saved photos.
    var geometricDistortionCorrection: Bool
    var rig: Rig
    var frames: [Frame]

    struct Rig: Codable, Sendable {
        var sensors: [Sensor]
        /// Distance between the two lenses in metres (from the calibration), informative.
        var baseline: Float?
    }
}

enum StereoCaptureError: LocalizedError {
    case unsupported(String)
    case noPairs
    var errorDescription: String? {
        switch self {
        case .unsupported(let s): s
        case .noPairs: "No photo pairs were captured, so nothing was saved."
        }
    }
}

/// Live numbers for the stereo screen.
struct StereoStats: Sendable, Equatable {
    var pairs = 0
    var bytes: Int64 = 0
    var hasDepth = false
    var hasCalibration = false
    var droppedBusy = 0
}

/// Records synchronised wide + ultra-wide photo pairs (AVCapturePhotoOutput constituent photo delivery on
/// `.builtInDualWideCamera`) at an interval, with each photo's calibration and, where the format allows, the dual
/// camera's depth map. All AVFoundation work runs on `queue`.
final class StereoCaptureEngine: NSObject, @unchecked Sendable {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "splatcapture.stereo")
    private let writer = DispatchQueue(label: "splatcapture.stereo.writer", qos: .userInitiated)
    private let photoOutput = AVCapturePhotoOutput()
    private let depthOutput = AVCaptureDepthDataOutput()
    private var device: AVCaptureDevice?
    private var configured = false
    private var depthEnabled = false
    private var calibrationEnabled = false
    private var gdc = true
    private var observers: [NSObjectProtocol] = []
    private var fallbacks = 0
    private var zoomed = false

    /// Xcode console line (Run → Console), so a failing camera can be diagnosed from the phone.
    private func log(_ s: String) { print("[SplatWork Stereo] \(s)") }

    // recording state (queue)
    private var recording: Recording?
    private var timer: DispatchSourceTimer?
    private var inFlight = 0
    /// The newest depth map and its time (engine queue only: the depth delegate is called on it).
    private var latestDepth: (AVDepthData, Double)?

    var onStats: (@Sendable (StereoStats) -> Void)?
    var onError: (@Sendable (String) -> Void)?

    static var isSupported: Bool {
        AVCaptureDevice.default(.builtInDualWideCamera, for: .video, position: .back) != nil
    }

    /// Starts the camera (preview) with the stereo configuration.
    func start(settings: CaptureSettings) {
        queue.async { [self] in
            do {
                if !configured { try configure(wantDepth: settings.stereoDepth) }
                applyCamera(settings, recording: false)
                if !session.isRunning { session.startRunning() }
            } catch {
                onError?(error.localizedDescription)
            }
        }
    }

    func stop() {
        queue.async { [self] in
            guard recording == nil else { return }
            if session.isRunning { session.stopRunning() }
        }
    }

    private func configure(wantDepth: Bool) throws {
        guard let dev = AVCaptureDevice.default(.builtInDualWideCamera, for: .video, position: .back) else {
            throw StereoCaptureError.unsupported("Stereo capture needs an iPhone with both a wide and an ultra-wide camera (iPhone 11 or newer).")
        }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .photo
        let input = try AVCaptureDeviceInput(device: dev)
        guard session.canAddInput(input), session.canAddOutput(photoOutput) else {
            throw StereoCaptureError.unsupported("The camera couldn't be set up for stereo capture.")
        }
        session.addInput(input)
        session.addOutput(photoOutput)
        guard photoOutput.isVirtualDeviceConstituentPhotoDeliverySupported else {
            throw StereoCaptureError.unsupported("This iPhone can't take photos from both cameras at once.")
        }
        photoOutput.isVirtualDeviceConstituentPhotoDeliveryEnabled = true
        photoOutput.maxPhotoQualityPrioritization = .balanced
        try dev.lockForConfiguration()
        // Frame and depth on the wide camera (the dual-wide device shows the ultra-wide below its switch-over zoom).
        if fallbacks < 2, let z = dev.virtualDeviceSwitchOverVideoZoomFactors.first?.doubleValue {
            dev.videoZoomFactor = CGFloat(z); zoomed = true
        }
        // The calibration (intrinsics, the lenses' offset) describes the uncorrected lens; some phones only deliver it
        // with iOS's distortion correction off. Try with it on (straighter ultra-wide photos) first.
        if !photoOutput.isCameraCalibrationDataDeliverySupported, dev.isGeometricDistortionCorrectionSupported {
            dev.isGeometricDistortionCorrectionEnabled = false
        }
        gdc = dev.isGeometricDistortionCorrectionSupported ? dev.isGeometricDistortionCorrectionEnabled : false
        calibrationEnabled = photoOutput.isCameraCalibrationDataDeliverySupported
        if wantDepth, fallbacks == 0, session.canAddOutput(depthOutput) {
            let formats = dev.activeFormat.supportedDepthDataFormats.filter {
                let t = CMFormatDescriptionGetMediaSubType($0.formatDescription)
                return t == kCVPixelFormatType_DepthFloat16 || t == kCVPixelFormatType_DisparityFloat16
                    || t == kCVPixelFormatType_DepthFloat32 || t == kCVPixelFormatType_DisparityFloat32
            }
            if let best = formats.max(by: { CMVideoFormatDescriptionGetDimensions($0.formatDescription).width < CMVideoFormatDescriptionGetDimensions($1.formatDescription).width }) {
                session.addOutput(depthOutput)
                depthOutput.isFilteringEnabled = true
                depthOutput.setDelegate(self, callbackQueue: queue)
                dev.activeDepthDataFormat = best
                depthEnabled = true
            }
        }
        dev.unlockForConfiguration()
        // Store photos in sensor orientation (no EXIF rotation), so depth maps and calibration line up with the pixels.
        if let c = photoOutput.connection(with: .video), c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }
        device = dev
        configured = true
        log("configured: calibration \(calibrationEnabled ? "on" : "off"), distortion correction \(gdc ? "on" : "off"), "
            + "depth \(depthEnabled ? "on" : "off"), zoom \(dev.videoZoomFactor), format \(dev.activeFormat.formatDescription.dimensions)")
        watchSession()
    }

    /// Runtime errors and interruptions used to stop the preview silently (a frozen frame). Log them, and on a runtime
    /// error fall back step by step — without the depth stream, then without the wide-camera zoom — and restart.
    private func watchSession() {
        guard observers.isEmpty else { return }
        let nc = NotificationCenter.default
        observers.append(nc.addObserver(forName: AVCaptureSession.runtimeErrorNotification, object: session, queue: nil) { [weak self] n in
            let e = n.userInfo?[AVCaptureSessionErrorKey] as? NSError
            let text = e.map { "\($0.domain) \($0.code): \($0.localizedDescription)" } ?? "unknown"
            guard let self else { return }
            self.queue.async { self.recover(text) }
        })
        observers.append(nc.addObserver(forName: AVCaptureSession.wasInterruptedNotification, object: session, queue: nil) { [weak self] n in
            let reason = (n.userInfo?[AVCaptureSessionInterruptionReasonKey] as? Int) ?? -1
            self?.log("interrupted (reason \(reason))")
            if reason == AVCaptureSession.InterruptionReason.videoDeviceInUseByAnotherClient.rawValue {
                self?.onError?("The camera is in use by something else (another app, or the LiDAR mode still closing). Switch modes again or reopen the app.")
            }
        })
        observers.append(nc.addObserver(forName: AVCaptureSession.interruptionEndedNotification, object: session, queue: nil) { [weak self] _ in
            guard let self else { return }
            self.log("interruption ended")
            self.queue.async { if !self.session.isRunning { self.session.startRunning() } }
        })
    }

    private func recover(_ error: String) {
        log("runtime error: \(error)")
        guard fallbacks < 2, let dev = device else {
            onError?("The camera stopped (\(error)). Stereo capture may not work on this iPhone.")
            return
        }
        fallbacks += 1
        session.beginConfiguration()
        if depthEnabled { session.removeOutput(depthOutput); depthEnabled = false; log("fallback: depth stream off") }
        if fallbacks >= 2, zoomed, (try? dev.lockForConfiguration()) != nil {
            dev.videoZoomFactor = 1; zoomed = false; dev.unlockForConfiguration(); log("fallback: zoom 1×")
        }
        session.commitConfiguration()
        session.startRunning()
        if fallbacks == 1 { onError?("This iPhone can't stream stereo depth while taking photos from both cameras, so pairs are recorded without the depth map.") }
    }

    /// Shutter cap (whole session) and torch (while recording). The photos follow the video auto-exposure, which
    /// picked 1/33 s indoors without a cap.
    private func applyCamera(_ settings: CaptureSettings, recording: Bool) {
        guard let dev = device, (try? dev.lockForConfiguration()) != nil else { return }
        CameraLimits.capShutter(dev, seconds: settings.maxShutterSeconds)
        CameraLimits.torch(dev, on: recording && settings.torchWhileRecording)
        dev.unlockForConfiguration()
        log(String(format: "shutter cap %@, torch %@", settings.maxShutterSeconds > 0 ? "1/\(Int((1 / settings.maxShutterSeconds).rounded())) s" : "auto",
                   recording && settings.torchWhileRecording ? "on" : "off"))
    }

    // MARK: Recording

    func startRecording(settings: CaptureSettings, completion: @escaping @Sendable (Result<Void, any Error>) -> Void) {
        queue.async { [self] in
            do {
                if !configured { try configure(wantDepth: settings.stereoDepth) }
                if !session.isRunning { session.startRunning() }
                applyCamera(settings, recording: true)
                let (url, name) = CaptureStore.uniqueURL(for: CaptureStore.defaultName())
                recording = try Recording(directory: url, name: name, gdc: gdc, settings: settings)
                inFlight = 0
                publish()
                let t = DispatchSource.makeTimerSource(queue: queue)
                t.schedule(deadline: .now() + 0.3, repeating: max(0.3, settings.stereoInterval))
                t.setEventHandler { [weak self] in self?.shoot() }
                t.resume()
                timer = t
                completion(.success(()))
            } catch {
                completion(.failure(error))
            }
        }
    }

    func stopRecording(completion: @escaping @Sendable (Result<CaptureSummary, any Error>) -> Void) {
        queue.async { [self] in
            timer?.cancel(); timer = nil
            if let s = recording?.settings { applyCamera(s, recording: false) }
            guard let rec = recording else { completion(.failure(StereoCaptureError.noPairs)); return }
            // Let photos still in flight arrive (up to ~3 s), then write the manifest.
            @Sendable func finish(_ tries: Int) {
                if inFlight > 0 && tries > 0 {
                    queue.asyncAfter(deadline: .now() + 0.1) { finish(tries - 1) }
                    return
                }
                recording = nil
                writer.async { completion(rec.finish()) }
            }
            finish(30)
        }
    }

    private func shoot() {
        guard let rec = recording, let dev = device else { return }
        guard inFlight == 0 else { rec.droppedBusy += 1; publish(); return }   // the last pair is still processing
        let codec: AVVideoCodecType = rec.settings.imageFormat == .heic && photoOutput.availablePhotoCodecTypes.contains(.hevc) ? .hevc : .jpeg
        let s = AVCapturePhotoSettings(format: [AVVideoCodecKey: codec])
        s.virtualDeviceConstituentPhotoDeliveryEnabledDevices = dev.constituentDevices
        s.isCameraCalibrationDataDeliveryEnabled = calibrationEnabled
        s.photoQualityPrioritization = .speed
        s.flashMode = .off
        inFlight = 1
        rec.beginPair(depth: depthEnabled ? latestDepth : nil)
        photoOutput.capturePhoto(with: s, delegate: self)
    }

    private func publish() {
        guard let rec = recording else { return }
        onStats?(StereoStats(pairs: rec.pairCount, bytes: rec.bytes, hasDepth: depthEnabled, hasCalibration: calibrationEnabled,
                             droppedBusy: rec.droppedBusy))
    }
}

extension StereoCaptureEngine: AVCapturePhotoCaptureDelegate {
    /// What the engine queue needs from a delivered photo (AVCapturePhoto itself isn't Sendable).
    private struct PhotoBits: @unchecked Sendable {
        let data: Data?, ultraWide: Bool, calibration: AVCameraCalibrationData?, dims: CMVideoDimensions
        let metadata: [String: Any], time: Double
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: (any Error)?) {
        let bits = PhotoBits(data: error == nil ? photo.fileDataRepresentation() : nil,
                             ultraWide: photo.sourceDeviceType == .builtInUltraWideCamera,
                             calibration: photo.cameraCalibrationData, dims: photo.resolvedSettings.photoDimensions,
                             metadata: photo.metadata, time: photo.timestamp.seconds)
        queue.async { [self] in
            guard let rec = recording, let data = bits.data else { return }
            rec.add(photo: data, sensor: bits.ultraWide ? "ultrawide" : "wide", calibration: bits.calibration,
                    dimensions: bits.dims, metadata: bits.metadata, timestamp: bits.time, writer: writer)
        }
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings, error: (any Error)?) {
        queue.async { [self] in
            if let error { onError?("A stereo photo failed: \(error.localizedDescription)") }
            recording?.endPair(writer: writer)
            inFlight = 0
            publish()
        }
    }
}

extension StereoCaptureEngine: AVCaptureDepthDataOutputDelegate {
    func depthDataOutput(_ output: AVCaptureDepthDataOutput, didOutput depthData: AVDepthData, timestamp: CMTime,
                         connection: AVCaptureConnection) {
        latestDepth = (depthData, timestamp.seconds)   // already on `queue` (setDelegate callbackQueue)
    }
}

/// One stereo bundle on disk. `beginPair` / `add` / `endPair` run on the engine queue; files are written on `writer`.
private final class Recording: @unchecked Sendable {
    let directory: URL
    let name: String
    let gdc: Bool
    let settings: CaptureSettings
    let createdAt = Date()
    var droppedBusy = 0
    private let lock = NSLock()
    private var frames: [StereoManifest.Frame] = []
    private var sensors: [String: StereoManifest.Sensor] = [:]
    private var current: StereoManifest.Frame?
    private var pendingDepth: (AVDepthData, Double)?
    private var byteCount: Int64 = 0
    private var wroteThumbnail = false

    var pairCount: Int { lock.withLock { frames.count } }
    var bytes: Int64 { lock.withLock { byteCount } }

    init(directory: URL, name: String, gdc: Bool, settings: CaptureSettings) throws {
        self.directory = directory; self.name = name; self.gdc = gdc; self.settings = settings
        do {
            try FileManager.default.createDirectory(at: directory.appendingPathComponent("images"), withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: directory.appendingPathComponent("depth"), withIntermediateDirectories: true)
        } catch { throw CaptureError.cannotCreate(error.localizedDescription) }
    }

    func beginPair(depth: (AVDepthData, Double)?) {
        let i = lock.withLock { frames.count }
        current = StereoManifest.Frame(index: i, timestamp: 0, images: [:], intrinsics: [:])
        pendingDepth = depth
    }

    func add(photo data: Data, sensor: String, calibration: AVCameraCalibrationData?, dimensions: CMVideoDimensions,
             metadata: [String: Any], timestamp: Double, writer: DispatchQueue) {
        guard var f = current else { return }
        let ext = (CGImageSourceCreateWithData(data as CFData, nil).flatMap { CGImageSourceGetType($0) } as String?) == "public.heic" ? "heic" : "jpg"
        let rel = String(format: "images/%06d_%@.%@", f.index, sensor, ext)
        f.timestamp = timestamp
        f.images[sensor] = rel
        // the size of this lens's photo (the resolved settings may describe the virtual device's)
        let props = CGImageSourceCreateWithData(data as CFData, nil).flatMap { CGImageSourceCopyPropertiesAtIndex($0, 0, nil) as? [String: Any] }
        let w = (props?[kCGImagePropertyPixelWidth as String] as? Int) ?? Int(dimensions.width)
        let h = (props?[kCGImagePropertyPixelHeight as String] as? Int) ?? Int(dimensions.height)
        var k: [Float]?
        if let c = calibration {
            let ref = c.intrinsicMatrixReferenceDimensions
            let sx = Float(w) / Float(max(ref.width, 1)), sy = Float(h) / Float(max(ref.height, 1))
            let m = c.intrinsicMatrix
            k = [m.columns.0.x * sx, m.columns.1.y * sy, m.columns.2.x * sx, m.columns.2.y * sy]
            f.intrinsics?[sensor] = k
        }
        if let exif = metadata[kCGImagePropertyExifDictionary as String] as? [String: Any] {
            f.exposureDuration = f.exposureDuration ?? (exif[kCGImagePropertyExifExposureTime as String] as? Double)
            f.iso = f.iso ?? ((exif[kCGImagePropertyExifISOSpeedRatings as String] as? [Double])?.first)
        }
        current = f
        if sensors[sensor] == nil {
            var ext3x4: [Float]?
            if let e = calibration?.extrinsicMatrix {
                // matrix_float4x3: 4 columns (R columns, then t) of 3 rows → row-major 3×4
                ext3x4 = (0..<3).flatMap { r in [e.columns.0[r], e.columns.1[r], e.columns.2[r], e.columns.3[r]] }
            }
            sensors[sensor] = StereoManifest.Sensor(id: sensor, width: w, height: h, intrinsics: k, extrinsics: ext3x4)
        }
        let url = directory.appendingPathComponent(rel)
        let thumb = sensor == "wide" && !wroteThumbnail
        if thumb { wroteThumbnail = true }
        writer.async { [self] in
            do { try data.write(to: url); lock.withLock { byteCount += Int64(data.count) } } catch {}
            if thumb { writeThumbnail(data) }
        }
    }

    /// Both photos (or whatever arrived) are in: keep the pair, with the depth map that came closest to it.
    func endPair(writer: DispatchQueue) {
        guard var f = current else { return }
        current = nil
        guard f.images["wide"] != nil || f.images["ultrawide"] != nil else { return }
        if let (d, t) = pendingDepth, abs(t - f.timestamp) < 0.25 || f.timestamp == 0 {
            let conv = d.depthDataType == kCVPixelFormatType_DepthFloat16 ? d : d.converting(toDepthDataType: kCVPixelFormatType_DepthFloat16)
            let map = conv.depthDataMap
            let dw = CVPixelBufferGetWidth(map), dh = CVPixelBufferGetHeight(map)
            let rel = String(format: "depth/%06d_wide.f16", f.index)
            f.depth = rel; f.depthWidth = dw; f.depthHeight = dh
            if let c = conv.cameraCalibrationData {
                let ref = c.intrinsicMatrixReferenceDimensions, m = c.intrinsicMatrix
                let sx = Float(dw) / Float(max(ref.width, 1)), sy = Float(dh) / Float(max(ref.height, 1))
                f.depthIntrinsics = [m.columns.0.x * sx, m.columns.1.y * sy, m.columns.2.x * sx, m.columns.2.y * sy]
                let e = c.extrinsicMatrix
                f.depthExtrinsics = (0..<3).flatMap { r in [e.columns.0[r], e.columns.1[r], e.columns.2[r], e.columns.3[r]] }
            }
            CVPixelBufferLockBaseAddress(map, .readOnly)
            var out = Data(capacity: dw * dh * 2)
            if let base = CVPixelBufferGetBaseAddress(map) {
                let stride = CVPixelBufferGetBytesPerRow(map)
                for y in 0..<dh { out.append(base.advanced(by: y * stride).assumingMemoryBound(to: UInt8.self), count: dw * 2) }
            }
            CVPixelBufferUnlockBaseAddress(map, .readOnly)
            let url = directory.appendingPathComponent(rel), bytes = out
            writer.async { [self] in try? bytes.write(to: url); lock.withLock { byteCount += Int64(bytes.count) } }
        }
        lock.withLock { frames.append(f) }
        if f.index % 25 == 0 { let snapshot = manifest(); writer.async { [self] in try? write(snapshot) } }
    }

    func finish() -> Result<CaptureSummary, any Error> {
        let m = manifest()
        guard !m.frames.isEmpty else {
            try? FileManager.default.removeItem(at: directory)
            return .failure(StereoCaptureError.noPairs)
        }
        do { try write(m) } catch { return .failure(error) }
        let ts = m.frames.map(\.timestamp).filter { $0 > 0 }
        return .success(CaptureSummary(url: directory, name: name, frameCount: m.frames.count, bytes: bytes, pointCount: 0,
                                       duration: (ts.max() ?? 0) - (ts.min() ?? 0)))
    }

    private func manifest() -> StereoManifest {
        let fs = lock.withLock { frames }
        let ss = ["wide", "ultrawide"].compactMap { sensors[$0] }
        var baseline: Float?
        if let a = sensors["wide"]?.extrinsics, let b = sensors["ultrawide"]?.extrinsics, a.count == 12, b.count == 12 {
            // camera centres in the reference frame: c = −Rᵀ t (mm)
            func centre(_ e: [Float]) -> SIMD3<Float> {
                let R = simd_float3x3(rows: [SIMD3(e[0], e[1], e[2]), SIMD3(e[4], e[5], e[6]), SIMD3(e[8], e[9], e[10])])
                return -(R.transpose * SIMD3(e[3], e[7], e[11]))
            }
            baseline = simd_length(centre(a) - centre(b)) / 1000
        }
        let w = sensors["wide"]
        return StereoManifest(name: name, createdAt: createdAt, device: DeviceInfo.modelName, appVersion: DeviceInfo.appVersion,
                              imageWidth: w?.width ?? 0, imageHeight: w?.height ?? 0, geometricDistortionCorrection: gdc,
                              rig: .init(sensors: ss, baseline: baseline), frames: fs)
    }

    private func write(_ m: StereoManifest) throws {
        let data = try ManifestCoding.encoder().encode(m)
        try data.write(to: directory.appendingPathComponent("manifest.json"), options: .atomic)
    }

    private func writeThumbnail(_ data: Data) {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let img = CGImageSourceCreateThumbnailAtIndex(src, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                                                      kCGImageSourceThumbnailMaxPixelSize: 480,
                                                                      kCGImageSourceCreateThumbnailWithTransform: true] as CFDictionary)
        else { return }
        // portrait, like the LiDAR captures' thumbnails
        let ci = CIImage(cgImage: img).oriented(.right)
        let ctx = CIContext()
        if let jpg = ctx.jpegRepresentation(of: ci, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, options: [:]) {
            try? jpg.write(to: directory.appendingPathComponent("thumbnail.jpg"))
        }
    }
}
