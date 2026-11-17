import Foundation

/// Dense points after the camera solve (Settings: Camera solving → Dense points; splatprep `dense.py`).
///
/// A splatprep solve (on this Mac or a worker) makes them itself (`dense` in its request) — on a worker that is where
/// the photos already are, and an NVIDIA worker can run COLMAP's multi-view stereo. Every other dataset (Object
/// Capture, captures with LiDAR depth, COLMAP / Nerfstudio imports) gets them here afterwards with splatprep's
/// `dense` task: fused LiDAR depth, or predicted depth with the Models pack. Never fails the preparation: when it
/// can't run, training simply starts from the sparse points, with a note saying why.
enum DensePoints {
    static let stage = "Dense points"

    /// The splatprep solve should make them itself: on, and the frames carry no sensor depth (LiDAR depth is fused
    /// afterwards from the finished dataset, where the depth maps are).
    static func inSolve(_ o: PrepOptions, frames: FrameSet) -> ProcessingRequest.Dense? {
        guard let d = o.dense.request, !frames.frames.contains(where: { $0.depth != nil }) else { return nil }
        return d
    }

    /// Dense points for a finished dataset, unless the solve already made them (`solved`: its `dense.json`).
    static func apply(_ ds: PreparedDataset, _ o: PrepOptions, solved: JSONValue? = nil,
                      runner: SplatPrepRunner = SplatPrepRunner(), progress: PipelineProgressHandler?) async throws -> PreparedDataset {
        guard let req = o.dense.request else { return ds }
        if let solved {
            progress?(PipelineProgress(stage: stage, fraction: -1, message: summary(solved)))
            return ds
        }
        var r = ProcessingRequest(task: .dense)
        r.features = nil; r.matching = nil; r.mapper = nil; r.cameraModel = nil
        r.outputDir = "."; r.datasetDir = "."
        r.dense = req
        progress?(PipelineProgress(stage: stage, fraction: 0, message: "building a dense point cloud"))
        let stats: [String: JSONValue]
        do {
            stats = try await runner.run(r, workDir: ds.root, progress: { p in
                progress?(PipelineProgress(stage: stage, fraction: p.fraction, message: p.message))
            })
        } catch let e as CancellationError {
            throw e
        } catch {
            if Task.isCancelled { throw CancellationError() }
            progress?(PipelineProgress(stage: stage, fraction: -1, message:
                "skipped: \(error.localizedDescription) — training starts from the sparse points"))
            return ds
        }
        if !o.keepIntermediates { try? FileManager.default.removeItem(at: ds.root.appendingPathComponent("request.json")) }
        progress?(PipelineProgress(stage: stage, fraction: 1, message: summary(.object(stats))))
        guard stats["status"]?.string == "ok" else { return ds }
        return (try? PreparedDataset.load(ds.root)) ?? ds
    }

    /// "248,902 dense points (fused LiDAR depth, 7 s)" / "skipped: … — training starts from the sparse points".
    static func summary(_ v: JSONValue) -> String {
        guard v["status"]?.string == "ok" else {
            let why = v["reason"]?.string ?? "unknown reason"
            return "\(v["status"]?.string ?? "skipped"): \(why) — training starts from the sparse points"
        }
        let n = Int(v["numPoints"]?.double ?? 0)
        let how: String
        switch (v["method"]?.string, v["depthSource"]?.string) {
        case ("mvs", _): how = "multi-view stereo"
        case (_, "sensor"): how = "fused sensor depth"
        case (_, "predicted"): how = "fused predicted depth"
        default: how = v["method"]?.string ?? "dense"
        }
        let secs = Int((v["seconds"]?.double ?? 0).rounded())
        return "\(n.formatted()) dense points (\(how), \(secs) s)"
    }
}
