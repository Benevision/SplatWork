import AVFoundation

/// Camera settings shared by the LiDAR (ARKit) and stereo modes. Must be called with the device locked for configuration.
enum CameraLimits {
    /// Caps auto-exposure's shutter at `seconds` (0 = no cap): the camera keeps adapting to bright and dark parts of a
    /// scene but raises ISO instead of going slower. Motion blur ruins splats; noise averages out across views.
    /// (A one-off custom exposure was undone by ARKit in some captures: frames at 1/91 s, ISO 50.)
    static func capShutter(_ device: AVCaptureDevice, seconds: Double) {
        let f = device.activeFormat
        let maxD = f.maxExposureDuration.seconds, minD = f.minExposureDuration.seconds
        let d = seconds > 0 ? min(max(seconds, minD), maxD) : maxD
        device.activeMaxExposureDuration = CMTime(seconds: d, preferredTimescale: 1_000_000)
    }

    /// Torch on (full) or off, where the camera has one.
    static func torch(_ device: AVCaptureDevice, on: Bool) {
        guard device.hasTorch, device.isTorchAvailable else { return }
        if on { try? device.setTorchModeOn(level: AVCaptureDevice.maxAvailableTorchLevel) } else if device.torchMode != .off { device.torchMode = .off }
    }
}
