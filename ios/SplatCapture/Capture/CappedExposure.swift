import AVFoundation
import Foundation

/// Auto-exposure with a hard shutter limit, done by the app itself: a few times a second it sets a custom exposure
/// (shutter ≤ the limit) and steers ISO by the camera's own metering (`exposureTargetOffset`). The camera's built-in
/// limit (`activeMaxExposureDuration`) and a one-off custom exposure were both ignored or undone in practice — ARKit
/// and the stereo photo pipeline still chose 1/33–1/100 s — so this keeps re-applying it.
///
/// `frozen` holds the current exposure (a locked recording); `stop()` hands exposure back to the camera.
final class CappedExposure: @unchecked Sendable {
    private let queue: DispatchQueue
    private var timer: DispatchSourceTimer?
    private weak var device: AVCaptureDevice?
    private var cap: Double = 0
    private var lastLog = Date.distantPast
    private(set) var frozen = false
    private let log: @Sendable (String) -> Void

    init(queue: DispatchQueue, log: @escaping @Sendable (String) -> Void) {
        self.queue = queue
        self.log = log
    }

    /// Starts steering `device` with shutter ≤ `seconds` (0 = leave exposure to the camera).
    private struct Box: @unchecked Sendable { let device: AVCaptureDevice }

    func start(_ device: AVCaptureDevice, seconds: Double) {
        let box = Box(device: device)
        queue.async { [self] in
            let device = box.device
            stopTimer()
            self.device = device
            cap = seconds
            frozen = false
            guard seconds > 0 else { release(device); return }
            guard device.isExposureModeSupported(.custom) else {
                // the dual-wide camera: its own limit (activeMaxExposureDuration, set by the caller) holds instead
                log("custom exposure not supported on this camera: using its own shutter limit")
                return
            }
            log(String(format: "capped auto-exposure on: shutter ≤ 1/%.0f s", 1 / seconds))
            tick()
            let t = DispatchSource.makeTimerSource(queue: queue)
            t.schedule(deadline: .now() + 0.25, repeating: 0.25)
            t.setEventHandler { [weak self] in self?.tick() }
            t.resume()
            timer = t
        }
    }

    /// Hold the current shutter and ISO (exposure lock while recording) or resume steering.
    func freeze(_ on: Bool) { queue.async { [self] in frozen = on; if on { report(force: true, why: "locked") } } }

    func stop() {
        queue.async { [self] in
            stopTimer()
            if let d = device { release(d) }
        }
    }

    private func stopTimer() { timer?.cancel(); timer = nil }

    private func release(_ d: AVCaptureDevice) {
        guard (try? d.lockForConfiguration()) != nil else { return }
        if d.isExposureModeSupported(.continuousAutoExposure) { d.exposureMode = .continuousAutoExposure }
        d.unlockForConfiguration()
    }

    private func tick() {
        guard !frozen, let d = device, cap > 0 else { return }
        let f = d.activeFormat
        let minD = f.minExposureDuration.seconds, maxD = f.maxExposureDuration.seconds
        let limit = min(max(cap, minD), maxD)
        var dur = d.exposureDuration.seconds
        var iso = Double(d.iso)
        guard dur.isFinite, dur > 0, iso.isFinite, iso > 0 else { return }
        // metering: how far (EV) the current exposure is from what the camera wants; + = too bright
        let off = Double(d.exposureTargetOffset)
        var gainEV = off.isFinite ? -off : 0
        if abs(gainEV) < 0.12 && dur <= limit * 1.02 && d.exposureMode == .custom { return }
        gainEV = max(-2, min(2, gainEV * 0.6))          // smooth: no visible pumping between frames
        let light = dur * iso * pow(2, gainEV)          // exposure the scene needs (shutter × ISO)
        // fastest-shutter-first: keep the shutter at the limit and get the brightness from ISO …
        dur = limit
        iso = light / dur
        let minISO = Double(f.minISO), maxISO = Double(f.maxISO)
        if iso < minISO {                                // … too bright even at the lowest ISO: shorter shutter
            iso = minISO
            dur = max(minD, light / iso)
        } else if iso > maxISO {                          // … too dark: stay sharp, accept a darker frame
            iso = maxISO
        }
        guard (try? d.lockForConfiguration()) != nil else { return }
        d.setExposureModeCustom(duration: CMTime(seconds: dur, preferredTimescale: 1_000_000), iso: Float(iso), completionHandler: nil)
        d.unlockForConfiguration()
        report(force: false, why: "")
    }

    private func report(force: Bool, why: String) {
        guard let d = device, force || Date().timeIntervalSince(lastLog) > 3 else { return }
        lastLog = Date()
        log(String(format: "exposure%@: 1/%.0f s, ISO %.0f, metering %+.2f EV, mode %@", why.isEmpty ? "" : " (\(why))",
                   1 / max(d.exposureDuration.seconds, 1e-6), d.iso, d.exposureTargetOffset,
                   d.exposureMode == .custom ? "custom" : d.exposureMode == .locked ? "locked" : "auto"))
    }
}
