import Foundation
import Observation

/// User-facing settings, persisted to UserDefaults.
@MainActor
@Observable
final class AppSettings {
    private static let key = "captureSettings.v1"

    var capture: CaptureSettings {
        didSet { persist() }
    }

    /// Last pairing code typed for each Mac (keyed by Mac name).
    var pairingCodes: [String: String] {
        didSet { UserDefaults.standard.set(pairingCodes, forKey: "pairingCodes") }
    }

    let videoFormats: [VideoFormatOption]

    init() {
        if let data = UserDefaults.standard.data(forKey: Self.key),
           let decoded = Self.decodeTolerant(data) {
            capture = decoded
        } else {
            capture = CaptureSettings()
        }
        pairingCodes = (UserDefaults.standard.dictionary(forKey: "pairingCodes") as? [String: String]) ?? [:]
        // Once: a saved "Auto" shutter (from before 1/250 s was the default) let the camera pick 1/33 s indoors, the
        // main cause of blurred captures. Choosing Auto again afterwards sticks.
        if !UserDefaults.standard.bool(forKey: "shutterLimitMigrated") {
            UserDefaults.standard.set(true, forKey: "shutterLimitMigrated")
            if capture.maxShutterSeconds <= 0 { capture.maxShutterSeconds = 1.0 / 250 }
        }
        videoFormats = VideoFormatOption.available()
        if !DeviceInfo.hasLiDAR {
            capture.recordDepth = false
        }
    }

    /// Saved settings laid over the current defaults, so adding a setting in an app update keeps every
    /// value the user already chose (plain decoding fails on the missing key and would reset them all).
    static func decodeTolerant(_ data: Data) -> CaptureSettings? {
        if let d = try? JSONDecoder().decode(CaptureSettings.self, from: data) { return d }
        guard let saved = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let defData = try? JSONEncoder().encode(CaptureSettings()),
              var merged = try? JSONSerialization.jsonObject(with: defData) as? [String: Any] else { return nil }
        for (k, v) in saved where merged[k] != nil { merged[k] = v }
        guard let m = try? JSONSerialization.data(withJSONObject: merged) else { return nil }
        return try? JSONDecoder().decode(CaptureSettings.self, from: m)
    }

    func resetToDefaults() {
        capture = CaptureSettings()
        if !DeviceInfo.hasLiDAR { capture.recordDepth = false }
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(capture) {
            UserDefaults.standard.set(data, forKey: Self.key)
        }
    }
}
