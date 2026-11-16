import Foundation
import simd

/// SplatWork Capture's stereo mode (`source` "iphone-stereo", PROTOCOL.md §8): photo pairs from the iPhone's wide and
/// ultra-wide cameras with their factory calibration, but no ARKit poses. They become a photo set in which every pair
/// is a rig group — the wide camera is the rig frame, the ultra-wide sits at its calibrated offset — so the camera solve
/// registers both lenses (the ultra-wide's wider view helps), keeps them together and comes out in metres. The dual
/// camera's depth map, when the phone recorded one, is resampled onto the wide photo.
enum StereoCaptureIngest {
    static let source = "iphone-stereo"

    struct Manifest: Decodable {
        struct Sensor: Decodable { var id: String; var width: Int?; var height: Int?; var intrinsics: [Float]?; var extrinsics: [Float]? }
        struct Rig: Decodable { var sensors: [Sensor]?; var baseline: Float? }
        struct Frame: Decodable {
            var index: Int
            var timestamp: Double?
            var images: [String: String]
            var intrinsics: [String: [Float]]?
            var depth: String?
            var depthWidth: Int?
            var depthHeight: Int?
            var depthIntrinsics: [Float]?
            var depthExtrinsics: [Float]?
        }
        var source: String?
        var name: String?
        var geometricDistortionCorrection: Bool?
        var rig: Rig?
        var frames: [Frame]
    }

    /// Whether `url` (bundle folder or zip) is a stereo capture without poses.
    static func isStereoBundle(_ url: URL) -> Bool {
        let fm = FileManager.default
        var data: Data?
        if url.pathExtension.lowercased() == "zip" {
            let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
            p.arguments = ["-p", url.path, "*manifest.json"]
            let pipe = Pipe(); p.standardOutput = pipe; p.standardError = FileHandle.nullDevice
            if (try? p.run()) != nil { data = pipe.fileHandleForReading.readDataToEndOfFile(); p.waitUntilExit() }
        } else {
            var m = url.appendingPathComponent("manifest.json")
            if !fm.fileExists(atPath: m.path),
               let sub = (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: nil))?
                .first(where: { fm.fileExists(atPath: $0.appendingPathComponent("manifest.json").path) }) {
                m = sub.appendingPathComponent("manifest.json")
            }
            data = try? Data(contentsOf: m)
        }
        guard let data, let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return (obj["source"] as? String) == source
    }

    /// Apple's `extrinsicMatrix` (3×4 row-major [R | t], t in mm, camera from reference) as (R, t in metres).
    static func extrinsics(_ e: [Float]?) -> (simd_float3x3, SIMD3<Float>)? {
        guard let e, e.count == 12 else { return nil }
        let R = simd_float3x3(rows: [SIMD3(e[0], e[1], e[2]), SIMD3(e[4], e[5], e[6]), SIMD3(e[8], e[9], e[10])])
        return (R, SIMD3(e[3], e[7], e[11]) / 1000)
    }

    /// The ultra-wide relative to the wide (the rig frame): rig_from_cam rotation and the camera centre in metres.
    /// nil when the calibration is missing or implausible (lenses 5–60 mm apart, turned < 5°).
    static func rigOffset(wide: [Float]?, ultra: [Float]?) -> (R: simd_float3x3, centre: SIMD3<Float>)? {
        guard let (Rw, tw) = extrinsics(wide), let (Ru, tu) = extrinsics(ultra) else { return nil }
        // x_w = Rw x_ref + tw, x_u = Ru x_ref + tu  ⇒  x_w = (Rw Ruᵀ) x_u + (tw − Rw Ruᵀ tu)
        let R = Rw * Ru.transpose
        let c = tw - R * tu
        let angle = acos(max(-1, min(1, (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2))) * 180 / .pi
        let d = simd_length(c)
        guard d > 0.005, d < 0.06, angle < 5 else { return nil }
        return (R, c)
    }

    static func ingest(_ url: URL, options: PhotoIngestOptions = .init(), into root: URL,
                       progress: PipelineProgressHandler? = nil) async throws -> FrameSet {
        let stage = "Importing stereo pairs"
        let fm = FileManager.default
        let scratch = try fm.ensureDirectory(root.appendingPathComponent(".import-tmp"))
        defer { try? fm.removeItem(at: scratch) }
        let bundle = try await CaptureImporter.resolveBundle(url, scratch: scratch)
        let m = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: bundle.appendingPathComponent("manifest.json")))
        guard !m.frames.isEmpty else { throw PipelineError.noInput("the stereo capture has no photo pairs") }
        let sensors = Dictionary((m.rig?.sensors ?? []).map { ($0.id, $0) }, uniquingKeysWith: { a, _ in a })
        let offset = rigOffset(wide: sensors["wide"]?.extrinsics, ultra: sensors["ultrawide"]?.extrinsics)
        if offset == nil {
            progress?(PipelineProgress(stage: stage, fraction: -1, message: "no usable lens calibration — the two cameras are solved separately and the scale isn't metric"))
        } else if let o = offset {
            progress?(PipelineProgress(stage: stage, fraction: 0, message: String(format: "lenses %.1f mm apart", simd_length(o.centre) * 1000)))
        }
        let wideRef = extrinsics(sensors["wide"]?.extrinsics)
        try fm.ensureDirectory(root.appendingPathComponent("images"))
        let ext = options.encoding == .jpeg ? "jpg" : "png"
        let counter = ProgressCounter(total: m.frames.count)

        let pairs: [[IngestedFrame]] = try await parallelMap(m.frames.count) { i in
            let f = m.frames[i]
            defer { let n = counter.increment(); progress?(PipelineProgress(stage: stage, fraction: 0.95 * Double(n) / Double(m.frames.count), message: "pair \(n) of \(m.frames.count)")) }
            var out: [IngestedFrame] = []
            for (s, sensor) in ["wide", "ultrawide"].enumerated() {
                guard let rel = f.images[sensor] else { continue }
                let src = bundle.appendingPathComponent(rel)
                guard let img = try? ImageUtil.load(src, maxSize: options.maxImageSize) else { continue }
                let name = String(format: "images/%06d_%@.%@", f.index, s == 0 ? "w" : "u", ext)
                try ImageUtil.write(img, to: root.appendingPathComponent(name), encoding: options.encoding)
                var fr = IngestedFrame(image: name, width: img.width, height: img.height)
                fr.sharpness = ImageMetrics.sharpness(img)
                fr.source = src.path
                fr.timestamp = f.timestamp
                fr.cameraGroup = s
                // the calibration's focal length (scaled if the photo was downsized), else the EXIF one
                let ow = Float(sensors[sensor]?.width ?? img.width), scale = Float(img.width) / max(ow, 1)
                if let k = f.intrinsics?[sensor] ?? sensors[sensor]?.intrinsics, k.count == 4 {
                    fr.focalPx = k[0] * scale; fr.cx = k[2] * scale; fr.cy = k[3] * scale
                } else {
                    fr.focalPx = ImageUtil.exifFocalPixels(src, width: img.width, height: img.height)
                }
                if let o = offset {
                    fr.group = f.index
                    let R = s == 0 ? matrix_identity_float3x3 : o.R
                    fr.rotationToRig = (0..<3).map { r in [R[0, r], R[1, r], R[2, r]] }   // row-major
                    fr.rigCentre = s == 0 ? [0, 0, 0] : [o.centre.x, o.centre.y, o.centre.z]
                }
                if s == 0, let d = f.depth, let dw = f.depthWidth, let dh = f.depthHeight, let dk = f.depthIntrinsics, dk.count == 4,
                   let k = fr.focalPx, depthIsWide(f.depthExtrinsics, wide: wideRef),
                   let depth = try? PreparedDataset.readDepth(bundle.appendingPathComponent(d), width: dw, height: dh) {
                    // resample onto the wide photo's pixel grid (same lens: only scale and principal point differ)
                    let tw = max(1, img.width / 4), th = max(1, img.height / 4)
                    let cx = (fr.cx ?? Float(img.width) / 2) / 4, cy = (fr.cy ?? Float(img.height) / 2) / 4, fx = k / 4
                    var res = [Float](repeating: 0, count: tw * th)
                    for y in 0..<th {
                        for x in 0..<tw {
                            let u = (Float(x) - cx) / fx * dk[0] + dk[2], v = (Float(y) - cy) / fx * dk[1] + dk[3]
                            let ui = Int(u.rounded()), vi = Int(v.rounded())
                            guard ui >= 0, vi >= 0, ui < dw, vi < dh else { continue }
                            let z = depth[vi * dw + ui]
                            if z.isFinite, z > 0.05, z < 20 { res[y * tw + x] = z }
                        }
                    }
                    let dn = String(format: "depth/%06d_w.bin", f.index)
                    try fm.ensureDirectory(root.appendingPathComponent("depth"))
                    try PreparedDataset.writeDepth(res, to: root.appendingPathComponent(dn))
                    fr.depth = dn; fr.depthWidth = tw; fr.depthHeight = th
                }
                out.append(fr)
            }
            return out
        }

        var set = FrameSet(root: root)
        set.frames = pairs.flatMap { $0 }
        guard !set.frames.isEmpty else { throw PipelineError.noInput("no stereo photos could be read") }
        set.isMultiViewRig = offset != nil
        try set.save()
        let depthCount = set.frames.filter { $0.depth != nil }.count
        progress?(PipelineProgress(stage: stage, fraction: 1, message: "\(m.frames.count) pairs, \(set.frames.count) photos"
                                   + (depthCount > 0 ? ", \(depthCount) depth maps" : "")))
        return set
    }

    /// The depth map is relative to the wide camera (its calibration names the same camera pose), so it lines up with
    /// the wide photo. Unknown → assume so only when the wide's own extrinsics are unknown too.
    static func depthIsWide(_ e: [Float]?, wide: (simd_float3x3, SIMD3<Float>)?) -> Bool {
        guard let (Rd, td) = extrinsics(e) else { return wide == nil }
        guard let (Rw, tw) = wide else { return true }
        let dR = Rd * Rw.transpose
        let angle = acos(max(-1, min(1, (dR[0, 0] + dR[1, 1] + dR[2, 2] - 1) / 2))) * 180 / .pi
        return angle < 0.5 && simd_length(td - tw) < 0.002
    }
}
