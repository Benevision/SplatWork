import CryptoKit
import Foundation
import ImageIO

/// How photos are sent to a worker (Settings → Uploads to workers). The project keeps its full-quality photos; only
/// the copy that is sent is re-saved. Measured on a 1,229-photo drone survey (3200 px, 5.6 GB): JPEG 90 sends 75 %
/// of the bytes at 39 dB from the original, JPEG 85 54 % at 36 dB — both far beyond what a splat reproduces
/// (about 20–25 dB), so training results don't change.
enum UploadQuality: String, CaseIterable, Identifiable, Sendable {
    case standard, compact, original
    var id: String { rawValue }
    var title: String {
        switch self {
        case .standard: return "Standard (JPEG 90, about ¾ the size)"
        case .compact: return "Compact (JPEG 85, about half the size)"
        case .original: return "Original (as imported)"
        }
    }
    /// ImageIO's quality scale (not libjpeg's): 0.75 ≈ libjpeg 90 (76 % of the bytes, 38 dB), 0.60 ≈ libjpeg 85
    /// (54 %, 36 dB), measured on the drone survey's 3200 px photos.
    var jpegQuality: Double? { switch self { case .standard: return 0.75; case .compact: return 0.60; case .original: return nil } }
    static let defaultsKey = "upload.quality"
    static var current: UploadQuality { UserDefaults.standard.string(forKey: defaultsKey).flatMap(UploadQuality.init) ?? .standard }
}

/// Uploads a folder to a worker in batches of about 200 MB that are combined there (`/v1/datasets/compose`, hard
/// links), so an interrupted upload resumes: the batches the worker already holds are remembered (per worker and
/// folder, keyed by the files' names, sizes and dates) and skipped next time. JPEG photos are re-saved for the
/// upload at the chosen `UploadQuality`, keeping their metadata (GPS, a drone's height above take-off).
/// Workers without composing (before 1.3.8) get the folder as one zip, as before.
enum UploadBatches {
    static let batchBytes = 200 * 1_048_576
    static let batchFiles = 600

    struct File { var rel: String; var url: URL; var size: Int; var mtime: Double }
    struct Batch { var files: [File]; var bytes: Int; var key: String }

    static func files(in dir: URL) -> [File] {
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey, .isRegularFileKey]
        guard let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: keys, options: [.skipsHiddenFiles]) else { return [] }
        let base = dir.standardizedFileURL.path
        var out: [File] = []
        for case let u as URL in e {
            guard let v = try? u.resourceValues(forKeys: Set(keys)), v.isRegularFile == true else { continue }
            let p = u.standardizedFileURL.path
            let rel = p.hasPrefix(base + "/") ? String(p.dropFirst(base.count + 1)) : u.lastPathComponent
            out.append(File(rel: rel, url: u, size: v.fileSize ?? 0, mtime: v.contentModificationDate?.timeIntervalSince1970 ?? 0))
        }
        return out.sorted { $0.rel < $1.rel }
    }

    /// Photos and other large files first, then the small files (frames.json, dataset.json, masks) in batches of
    /// their own: those are rewritten on every attempt, and mixing them in would resend a photo batch each time.
    static func batches(_ files: [File], quality: UploadQuality) -> [Batch] {
        let big = files.filter { isPhoto($0.rel) || $0.size >= 1_048_576 }
        let small = files.filter { !(isPhoto($0.rel) || $0.size >= 1_048_576) }
        return pack(big, quality: quality) + pack(small, quality: quality)
    }

    static func pack(_ files: [File], quality: UploadQuality) -> [Batch] {
        var out: [Batch] = [], cur: [File] = [], bytes = 0
        func flush() {
            guard !cur.isEmpty else { return }
            var h = SHA256()
            // the encoding itself, so a change in how photos are re-saved sends them again
            h.update(data: Data("\(quality.rawValue)|\(quality.jpegQuality ?? 1)|meta2".utf8))
            for f in cur { h.update(data: Data("\(f.rel)|\(f.size)|\(Int(f.mtime))\n".utf8)) }
            out.append(Batch(files: cur, bytes: bytes, key: h.finalize().map { String(format: "%02x", $0) }.joined()))
            cur = []; bytes = 0
        }
        for f in files {
            if !cur.isEmpty && (bytes + f.size > batchBytes || cur.count >= batchFiles) { flush() }
            cur.append(f); bytes += f.size
        }
        flush()
        return out
    }

    static func isPhoto(_ rel: String) -> Bool {
        let ext = (rel as NSString).pathExtension.lowercased()
        return ext == "jpg" || ext == "jpeg"
    }

    /// Re-saves a JPEG at `quality`, keeping its metadata. False (and nothing written) when it wouldn't be smaller.
    static func recompress(_ src: URL, to dst: URL, quality: Double) -> Bool {
        guard let s = CGImageSourceCreateWithURL(src as CFURL, nil), let type = CGImageSourceGetType(s),
              let img = CGImageSourceCreateImageAtIndex(s, 0, nil),
              let d = CGImageDestinationCreateWithURL(dst as CFURL, type, 1, nil) else { return false }
        // the full metadata, XMP included (DJI's height above take-off), not just the EXIF properties
        CGImageDestinationAddImageAndMetadata(d, img, CGImageSourceCopyMetadataAtIndex(s, 0, nil),
                                              [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(d) else { try? FileManager.default.removeItem(at: dst); return false }
        let a = (try? src.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        let b = (try? dst.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? .max
        if b >= a { try? FileManager.default.removeItem(at: dst); return false }
        return true
    }

    /// The batch as a folder named like the original (so the worker finds the same root in every batch).
    static func stage(_ b: Batch, rootName: String, quality: UploadQuality, in tmp: URL) async throws -> URL {
        let root = tmp.appendingPathComponent(rootName, isDirectory: true)
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        try await withThrowingTaskGroup(of: Void.self) { group in
            var running = 0
            for f in b.files {
                if running >= max(2, ProcessInfo.processInfo.activeProcessorCount) { try await group.next(); running -= 1 }
                group.addTask {
                    let dst = root.appendingPathComponent(f.rel)
                    try fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
                    if let q = quality.jpegQuality, isPhoto(f.rel), recompress(f.url, to: dst, quality: q) { return }
                    if (try? fm.linkItem(at: f.url, to: dst)) == nil { try fm.copyItem(at: f.url, to: dst) }
                }
                running += 1
            }
            try await group.waitForAll()
        }
        return root
    }

    // MARK: remembered batches

    struct State: Codable { var batches: [String: String] = [:]; var updated = Date() }

    static func stateURL(worker: URL, dir: URL) -> URL {
        let h = SHA256.hash(data: Data((worker.absoluteString + "|" + dir.standardizedFileURL.path).utf8)).prefix(12)
            .map { String(format: "%02x", $0) }.joined()
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SplatWork/uploads", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base.appendingPathComponent("\(h).json")
    }

    static func load(_ url: URL) -> State { (try? JSONDecoder().decode(State.self, from: Data(contentsOf: url))) ?? State() }
    static func save(_ s: State, _ url: URL) {
        var s = s; s.updated = Date()
        if let d = try? JSONEncoder().encode(s) { try? d.write(to: url, options: .atomic) }
    }

    /// Forgets remembered uploads older than two weeks.
    static func prune() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("SplatWork/uploads")
        for u in (try? FileManager.default.contentsOfDirectory(at: base, includingPropertiesForKeys: [.contentModificationDateKey])) ?? [] {
            if let d = try? u.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate, Date().timeIntervalSince(d) > 14 * 86_400 {
                try? FileManager.default.removeItem(at: u)
            }
        }
    }
}

/// Progress across batches: bytes done so far plus the current batch's share. Reports "resumed" batches at once.
final class BatchProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var done = 0
    let total: Int
    let report: (@Sendable (Double) -> Void)?
    init(total: Int, report: (@Sendable (Double) -> Void)?) { self.total = max(total, 1); self.report = report }
    func finished(_ bytes: Int) { lock.lock(); done += bytes; let f = Double(done) / Double(total); lock.unlock(); report?(min(1, f)) }
    func partial(_ bytes: Int, _ f: Double) {
        lock.lock(); let v = (Double(done) + Double(bytes) * f) / Double(total); lock.unlock(); report?(min(1, v))
    }
}

extension WorkerClient {
    /// Uploads a folder; returns its `datasetId`. Batched and resumable where the worker can compose (see
    /// `UploadBatches`), photos re-saved at `quality`.
    func uploadDataset(directory: URL, quality: UploadQuality = .current,
                       progress: (@Sendable (Double) -> Void)? = nil) async throws -> String {
        let files = UploadBatches.files(in: directory)
        let batches = UploadBatches.batches(files, quality: quality)
        // /v1/datasets/compose: every worker since 1.3.8 (advertised as "chain" only with the processing runtime)
        let inf = try? await info(timeout: 10)
        let ver = (inf?.workerVersion ?? "0").split(separator: ".").compactMap { Int($0) }
        let canCompose = (inf?.capabilities?.contains("chain") ?? false) || ver.lexicographicallyPrecedes([1, 3, 8]) == false && ver.count >= 2
        let tmpRoot = FileManager.default.temporaryDirectory.appendingPathComponent("splat-upload-\(UUID().uuidString.prefix(8))")
        defer { try? FileManager.default.removeItem(at: tmpRoot) }
        let name = directory.lastPathComponent
        let prog = BatchProgress(total: batches.reduce(0) { $0 + $1.bytes }, report: progress)

        func send(_ b: UploadBatches.Batch, _ i: Int) async throws -> String {
            let tmp = tmpRoot.appendingPathComponent("b\(i)")
            defer { try? FileManager.default.removeItem(at: tmp) }
            let staged = try await UploadBatches.stage(b, rootName: name, quality: quality, in: tmp)
            let zip = tmp.appendingPathComponent("batch.zip")
            try await ZipUtil.zip(directory: staged, to: zip)
            return try await uploadDataset(zip: zip) { f in prog.partial(b.bytes, f) }
        }

        guard canCompose, batches.count > 1 else {
            // one zip (small folders, or workers without /v1/datasets/compose)
            let all = UploadBatches.Batch(files: files, bytes: files.reduce(0) { $0 + $1.size }, key: "")
            return try await send(all, 0)
        }

        UploadBatches.prune()
        let stateURL = UploadBatches.stateURL(worker: baseURL, dir: directory)
        var state = UploadBatches.load(stateURL)
        let present = Set(((try? await listDatasetIds()) ?? []))
        var ids: [String] = []
        for (i, b) in batches.enumerated() {
            try Task.checkCancellation()
            if let id = state.batches[b.key], present.contains(id) {
                ids.append(id); prog.finished(b.bytes); continue      // already on the worker from an earlier try
            }
            let id = try await send(b, i)
            state.batches[b.key] = id
            UploadBatches.save(state, stateURL)
            ids.append(id); prog.finished(b.bytes)
        }
        // combine, at most 16 layers per call (the worker's limit)
        var layer = ids
        while layer.count > 1 {
            var next: [String] = []
            for chunk in stride(from: 0, to: layer.count, by: 16).map({ Array(layer[$0..<min($0 + 16, layer.count)]) }) {
                next.append(chunk.count == 1 ? chunk[0] : try await compose(chunk.map { ComposeLayer(datasetId: $0) }))
            }
            layer = next
        }
        return layer[0]
    }
}
