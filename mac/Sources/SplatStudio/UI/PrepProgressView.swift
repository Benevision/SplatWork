import AppKit
import ImageIO
import SwiftUI
import simd

/// The New Project sheet while a dataset is being prepared: where the work runs, every step with its time, what the
/// current step is doing (counters, camera-solve sub-steps, upload speed), and live previews of its output — the
/// latest photos, masks and sky as coloured overlays, and a top-down map of the solved cameras and points.
struct PrepProgressView: View {
    let prep: PrepSession
    let title: String
    let summary: String
    /// Name of the device the work was sent to (nil: this Mac).
    let workerName: String?
    let preview: PrepPreview
    @State private var showLog = false

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            VStack(alignment: .leading, spacing: 14) {
                header
                stepStrip
                HStack(alignment: .top, spacing: 14) {
                    activeCard.frame(width: 330)
                    previewPanel.frame(maxWidth: .infinity)
                }
                .frame(minHeight: 300, maxHeight: .infinity)
                if prep.thenTrain {
                    Label("Training starts by itself when the dataset is ready — follow it in the Train panel and the banner at the bottom of the viewer.",
                          systemImage: "arrow.turn.down.right")
                        .font(.callout).foregroundStyle(.secondary)
                }
                logSection
            }
        }
        .task(id: prep.projectDir) {
            while !Task.isCancelled {
                if let dir = prep.projectDir { await preview.refresh(dataset: dir.appendingPathComponent("dataset"), step: prep.step, since: prep.previewAnyAge ? nil : prep.runStarted) }
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    // MARK: header

    private var header: some View {
        HStack(alignment: .center, spacing: 16) {
            ZStack {
                Circle().stroke(.quaternary, lineWidth: 7)
                Circle().trim(from: 0, to: max(0.005, prep.overall))
                    .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 7, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.easeOut(duration: 0.6), value: prep.overall)
                Text("\(Int(prep.overall * 100)) %").font(.headline.monospacedDigit())
            }
            .frame(width: 64, height: 64)
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.title2.bold()).lineLimit(1)
                Text(summary).font(.callout).foregroundStyle(.secondary).lineLimit(2)
                HStack(spacing: 14) {
                    Label("Elapsed \(formatDuration(prep.elapsed))", systemImage: "clock")
                    if let eta = prep.eta { Label("About \(formatDuration(eta)) left", systemImage: "hourglass") }
                    else { Label("Estimating time left…", systemImage: "hourglass") }
                    heartbeat
                }
                .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
    }

    private var heartbeat: some View {
        let s = prep.sinceLastUpdate
        let color: Color = s < 15 ? .green : s < 90 ? .orange : .red
        return HStack(spacing: 5) {
            Circle().fill(color).frame(width: 7, height: 7)
            Text(s < 3 ? "working" : "quiet for \(formatDuration(s))")
        }
        .help("Time since the current step last reported progress. Some steps (such as placing cameras) report rarely: orange or red only means it has been quiet, not that it has failed.")
    }

    // MARK: steps

    /// The steps this run goes through: the ones seen so far, then the ones still expected.
    private var plannedSteps: [PrepSession.Step] {
        var list = prep.spans.map(\.step)
        let remote = workerName != nil
        var expected: [PrepSession.Step] = [.images]
        if remote { expected.append(.uploading) }
        expected.append(.masks)
        if prep.expectsSky { expected.append(.sky) }
        expected += [.cameras, .finalising]
        if remote { expected.append(.downloading) }
        for s in expected where !list.contains(s) {
            // insert in pipeline order after the last step already seen that comes before it
            let order = PrepSession.Step.allCases
            let pos = list.lastIndex { order.firstIndex(of: $0)! < order.firstIndex(of: s)! }.map { $0 + 1 } ?? 0
            if (list.last.map { order.firstIndex(of: $0)! } ?? -1) >= order.firstIndex(of: s)! && pos < list.count { continue }
            list.insert(s, at: min(pos, list.count))
        }
        var seen = Set<PrepSession.Step>()
        return list.filter { seen.insert($0).inserted }
    }

    private var stepStrip: some View {
        let steps = plannedSteps
        let current = prep.step
        let doneSet = Set(prep.spans.filter { $0.end != nil && $0.step != current }.map(\.step))
        return HStack(spacing: 0) {
            ForEach(Array(steps.enumerated()), id: \.element) { i, s in
                let done = doneSet.contains(s)
                let active = s == current && prep.running
                VStack(spacing: 5) {
                    ZStack {
                        Circle().fill(done ? AnyShapeStyle(Color.green.opacity(0.18)) : active ? AnyShapeStyle(Color.accentColor.opacity(0.22)) : AnyShapeStyle(.quaternary.opacity(0.5)))
                        Image(systemName: done ? "checkmark" : s.symbol)
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(done ? AnyShapeStyle(Color.green) : active ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.tertiary))
                            .symbolEffect(.pulse, isActive: active)
                    }
                    .frame(width: 36, height: 36)
                    .overlay(Circle().stroke(active ? Color.accentColor : .clear, lineWidth: 2))
                    Text(s.title).font(.caption.weight(active ? .semibold : .regular))
                        .foregroundStyle(done || active ? .primary : .secondary).lineLimit(1)
                    Text(prep.duration(of: s).map { formatDuration($0) } ?? " ")
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
                .frame(minWidth: 70)
                if i < steps.count - 1 {
                    Rectangle().fill(done ? Color.green.opacity(0.5) : Color.secondary.opacity(0.25)).frame(height: 2)
                        .frame(maxWidth: .infinity).padding(.bottom, 34)
                }
            }
        }
        .padding(.horizontal, 6)
    }

    // MARK: current step

    private var activeCard: some View {
        let s = prep.step
        let msg = prep.progress.message
        return VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: s?.symbol ?? "gearshape.2").font(.title3).foregroundStyle(Color.accentColor)
                VStack(alignment: .leading, spacing: 1) {
                    Text(s?.title ?? "Starting").font(.headline)
                    Label(where_, systemImage: workerName == nil || isLocalStep ? "laptopcomputer" : "server.rack")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if let s, let d = prep.duration(of: s) { Text(formatDuration(d)).font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
            }
            if let c = Self.counter(in: msg) {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(alignment: .firstTextBaseline) {
                        Text(c.done.formatted()).font(.title2.bold().monospacedDigit())
                        Text("of \(c.total.formatted()) \(c.unit)").foregroundStyle(.secondary)
                        Spacer()
                        if let r = rate { Text(r).font(.caption.monospacedDigit()).foregroundStyle(.secondary) }
                    }
                    ProgressView(value: Double(c.done), total: Double(max(c.total, 1)))
                }
            } else if let pct = Self.percent(in: msg) {
                ProgressView(value: pct / 100)
            }
            Text(detail(msg)).font(.callout).foregroundStyle(.secondary)
                .lineLimit(4).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
            if s == .cameras { solveChecklist }
            recentActivity
            Spacer(minLength: 0)
        }
        .padding(12)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.35)))
    }

    /// The last few log lines of the current step.
    @ViewBuilder private var recentActivity: some View {
        let lines = Array(prep.log.suffix(40).filter { !$0.isEmpty }.suffix(6))
        if !lines.isEmpty {
            VStack(alignment: .leading, spacing: 3) {
                Text("Recent activity").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                ForEach(Array(lines.enumerated()), id: \.offset) { _, l in
                    Text(l).font(.caption2.monospaced()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.tail)
                }
            }
            .padding(.top, 4)
        }
    }

    /// "≈ 41 / min" from the step's counter over the last minute.
    private var rate: String? {
        guard let r = prep.counterRate, r > 0 else { return nil }
        return r >= 1 ? String(format: "≈ %.0f / s", r) : String(format: "≈ %.0f / min", r * 60)
    }

    private var isLocalStep: Bool { prep.step == .images && prep.host == nil }

    private var where_: String {
        guard let w = workerName else { return "on this Mac" }
        switch prep.step {
        case .uploading?: return "this Mac → \(w)"
        case .downloading?: return "\(w) → this Mac"
        case .images?: return prep.host == nil ? "on this Mac" : "on \(w)"
        case nil: return "starting"
        default: return "on \(w)"
        }
    }

    private var solveChecklist: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(PrepSession.solveSubsteps.enumerated()), id: \.offset) { i, sub in
                let cur = prep.solveSubstep
                HStack(spacing: 7) {
                    Image(systemName: i < cur ? "checkmark.circle.fill" : i == cur ? "circle.dotted.circle" : "circle")
                        .foregroundStyle(i < cur ? AnyShapeStyle(Color.green) : i == cur ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.tertiary))
                        .symbolEffect(.rotate, isActive: i == cur)
                    Text(sub.title).font(.caption.weight(i == cur ? .semibold : .regular))
                        .foregroundStyle(i <= cur ? .primary : .secondary)
                }
            }
        }
        .padding(.top, 2)
    }

    private func detail(_ msg: String) -> String {
        msg.isEmpty ? (prep.progress.stage.isEmpty ? "Getting ready…" : prep.progress.stage) : msg
    }

    /// "matched 9866/14128 pairs", "registered 941 images" (no total), "sky mask 196/1227" → a counter.
    static func counter(in msg: String) -> (done: Int, total: Int, unit: String)? {
        let pattern = #"(\d[\d,]*)\s*/\s*(\d[\d,]*)\s*([A-Za-z]+)?"#
        guard let re = try? NSRegularExpression(pattern: pattern),
              let m = re.firstMatch(in: msg, range: NSRange(msg.startIndex..., in: msg)),
              let r1 = Range(m.range(at: 1), in: msg), let r2 = Range(m.range(at: 2), in: msg),
              let a = Int(msg[r1].replacingOccurrences(of: ",", with: "")), let b = Int(msg[r2].replacingOccurrences(of: ",", with: "")),
              b > 0, a <= b else { return nil }
        var unit = m.range(at: 3).location != NSNotFound ? Range(m.range(at: 3), in: msg).map { String(msg[$0]) } ?? "" : ""
        if unit.isEmpty {
            let l = msg.lowercased()
            unit = l.contains("pair") ? "pairs" : l.contains("mask") || l.contains("frame") || l.contains("image") || l.contains("photo") ? "photos" : ""
        }
        return (a, b, unit)
    }

    static func percent(in msg: String) -> Double? {
        guard let re = try? NSRegularExpression(pattern: #"(\d{1,3})\s*%"#),
              let m = re.firstMatch(in: msg, range: NSRange(msg.startIndex..., in: msg)),
              let r = Range(m.range(at: 1), in: msg), let v = Double(msg[r]), v <= 100 else { return nil }
        return v
    }

    // MARK: previews

    @ViewBuilder private var previewPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(preview.heading).font(.headline)
                Spacer()
                Text(preview.caption).font(.caption).foregroundStyle(.secondary)
            }
            Group {
                if let plot = preview.plot {
                    CameraPlotView(plot: plot)
                } else if !preview.tiles.isEmpty {
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 6), count: 3), spacing: 6) {
                        ForEach(preview.tiles) { t in
                            Image(decorative: t.image, scale: 1).resizable().aspectRatio(contentMode: .fill)
                                .frame(minWidth: 0, maxWidth: .infinity).frame(height: tileHeight).clipped()
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                                .transition(.opacity)
                        }
                    }
                    .animation(.easeInOut(duration: 0.4), value: preview.tiles.map(\.id))
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: workerName != nil && prep.step != .images ? "server.rack" : "photo.on.rectangle.angled")
                            .font(.system(size: 34)).foregroundStyle(.tertiary)
                        Text(placeholder).font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .frame(maxHeight: .infinity)
            if let legend = preview.legend {
                HStack(spacing: 12) {
                    ForEach(legend, id: \.label) { l in
                        HStack(spacing: 5) { RoundedRectangle(cornerRadius: 2).fill(l.color).frame(width: 12, height: 9); Text(l.label) }
                    }
                }
                .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.35)))
    }

    /// Three rows of tiles that fill the panel at the sheet's running size.
    private var tileHeight: CGFloat { 92 }

    private var placeholder: String {
        if let w = workerName, prep.step != .images, prep.step != .uploading {
            return "This step runs on \(w).\nIts previews appear when the dataset comes back to this Mac."
        }
        return "Previews of each step appear here as it runs."
    }

    // MARK: log

    private var logSection: some View {
        DisclosureGroup(isExpanded: $showLog) {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 1) {
                        ForEach(Array(prep.log.enumerated()), id: \.offset) { i, l in Text(l).id(i) }
                    }
                    .font(.caption.monospaced()).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                }
                .frame(height: 140)
                .onChange(of: prep.log.count) { _, n in if n > 0 { proxy.scrollTo(n - 1, anchor: .bottom) } }
                .onAppear { if !prep.log.isEmpty { proxy.scrollTo(prep.log.count - 1, anchor: .bottom) } }
            }
        } label: {
            HStack {
                Text("Log").font(.callout.weight(.medium))
                Text(prep.log.last ?? "").font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.tail)
                Spacer()
                Button("Copy") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(prep.log.joined(separator: "\n"), forType: .string) }
                    .buttonStyle(.borderless).font(.caption)
            }
        }
    }
}

// MARK: - camera map

/// Top-down map of a solved dataset: sparse points in their colours, each camera as a small wedge pointing where it
/// looks. The view is levelled with the dataset's up vector and turned so the longer side is horizontal.
struct CameraPlot: Sendable {
    var points: [SIMD2<Float>] = []
    var colors: [SIMD3<UInt8>] = []
    var cameras: [(pos: SIMD2<Float>, dir: SIMD2<Float>, holdout: Bool)] = []
    var bounds: (min: SIMD2<Float>, max: SIMD2<Float>) = (.zero, .one)
}

struct CameraPlotView: View {
    let plot: CameraPlot
    var body: some View {
        Canvas { ctx, size in
            let lo = plot.bounds.min, hi = plot.bounds.max
            let span = max(hi.x - lo.x, hi.y - lo.y, 1e-6)
            let scale = CGFloat(min(Float(size.width - 16) / max(hi.x - lo.x, 1e-6), Float(size.height - 16) / max(hi.y - lo.y, 1e-6)))
            let cx = size.width / 2, cy = size.height / 2
            let mid = (lo + hi) / 2
            func pt(_ p: SIMD2<Float>) -> CGPoint { CGPoint(x: cx + CGFloat(p.x - mid.x) * scale, y: cy - CGFloat(p.y - mid.y) * scale) }
            ctx.fill(Path(roundedRect: CGRect(origin: .zero, size: size), cornerRadius: 6), with: .color(.black.opacity(0.55)))
            for (p, c) in zip(plot.points, plot.colors) {
                let q = pt(p)
                ctx.fill(Path(CGRect(x: q.x - 0.6, y: q.y - 0.6, width: 1.2, height: 1.2)),
                         with: .color(Color(red: Double(c.x) / 255, green: Double(c.y) / 255, blue: Double(c.z) / 255).opacity(0.85)))
            }
            let w = min(CGFloat(span) * scale * 0.006 + 1.8, 5)
            for cam in plot.cameras {
                let q = pt(cam.pos)
                var d = CGVector(dx: CGFloat(cam.dir.x), dy: -CGFloat(cam.dir.y))
                let len = max(sqrt(d.dx * d.dx + d.dy * d.dy), 1e-6)
                d = CGVector(dx: d.dx / len, dy: d.dy / len)
                let n = CGVector(dx: -d.dy, dy: d.dx)
                var path = Path()
                path.move(to: q)
                path.addLine(to: CGPoint(x: q.x + (d.dx * 2 + n.dx) * w, y: q.y + (d.dy * 2 + n.dy) * w))
                path.addLine(to: CGPoint(x: q.x + (d.dx * 2 - n.dx) * w, y: q.y + (d.dy * 2 - n.dy) * w))
                path.closeSubpath()
                ctx.fill(path, with: .color(cam.holdout ? .orange.opacity(0.9) : Color(red: 0.38, green: 0.65, blue: 1).opacity(0.85)))
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

// MARK: - preview model

/// Polls the project's dataset folder while it is being prepared and builds the preview for the current step.
@MainActor @Observable
final class PrepPreview {
    struct Tile: Identifiable { let id: String; let image: CGImage }
    struct Legend { let label: String; let color: Color }

    private(set) var heading = "Preview"
    private(set) var caption = ""
    private(set) var tiles: [Tile] = []
    private(set) var plot: CameraPlot? = nil
    private(set) var legend: [Legend]? = nil
    private var cache: [String: CGImage] = [:]
    private var plotStamp: Date? = nil

    func refresh(dataset dir: URL, step: PrepSession.Step?, since: Date?) async {
        let cachedKeys = Set(cache.keys)
        let result = await Task.detached(priority: .utility) { () -> Snapshot in
            Self.build(dataset: dir, step: step, since: since, cached: cachedKeys)
        }.value
        for (k, v) in result.newImages { cache[k] = v }
        if cache.count > 200 { cache = cache.filter { k, _ in result.tileKeys.contains(k) } }
        heading = result.heading; caption = result.caption; legend = result.legend
        tiles = result.tileKeys.compactMap { k in cache[k].map { Tile(id: k, image: $0) } }
        if let p = result.plot { plot = p.plot; plotStamp = p.stamp } else if result.clearPlot { plot = nil; plotStamp = nil }
    }

    struct Snapshot: @unchecked Sendable {
        var heading = "Preview", caption = ""
        var tileKeys: [String] = []
        var newImages: [String: CGImage] = [:]
        var legend: [Legend]? = nil
        var plot: (plot: CameraPlot, stamp: Date)? = nil
        var clearPlot = false
    }

    nonisolated static func build(dataset dir: URL, step: PrepSession.Step?, since: Date?, cached: Set<String>) -> Snapshot {
        var s = Snapshot()
        let fm = FileManager.default
        func newest(_ sub: String, _ n: Int) -> [(URL, Date)] {
            let d = dir.appendingPathComponent(sub)
            guard let items = try? fm.contentsOfDirectory(at: d, includingPropertiesForKeys: [.contentModificationDateKey], options: .skipsHiddenFiles) else { return [] }
            let dated = items.compactMap { u -> (URL, Date)? in
                guard ["jpg", "jpeg", "png", "heic"].contains(u.pathExtension.lowercased()) else { return nil }
                return (u, (try? u.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast)
            }
            return Array(dated.sorted { $0.1 > $1.1 }.prefix(n))
        }
        func count(_ sub: String) -> Int { (try? fm.contentsOfDirectory(atPath: dir.appendingPathComponent(sub).path))?.count ?? 0 }
        func imageFor(stem: String) -> URL? {
            for ext in ["jpg", "jpeg", "png", "heic", "JPG"] {
                let u = dir.appendingPathComponent("images/\(stem).\(ext)")
                if fm.fileExists(atPath: u.path) { return u }
            }
            return nil
        }
        func add(_ key: String, _ make: () -> CGImage?) {
            if !cached.contains(key), let im = make() { s.newImages[key] = im }
            if cached.contains(key) || s.newImages[key] != nil { s.tileKeys.append(key) }
        }

        switch step {
        case .masks?, .sky?:
            let sky = step == .sky
            let subs = sky ? ["splatprep-sky/masks", "sky"] : ["masks", "splatprep-masks/masks"]
            let files = subs.flatMap { newest($0, 9) }.sorted { $0.1 > $1.1 }.prefix(9)
            s.heading = sky ? "Finding the sky" : "Masks"
            let n = subs.map(count).max() ?? 0
            s.caption = n > 0 ? "\(formatCount(n)) done" : ""
            s.legend = sky ? [Legend(label: "sky (left out, learned as a backdrop)", color: .blue.opacity(0.7))]
                           : [Legend(label: "left out of training", color: .red.opacity(0.7))]
            for (u, d) in files {
                let stem = u.deletingPathExtension().lastPathComponent
                guard let img = imageFor(stem: stem) else { continue }
                add("\(sky ? "s" : "m"):\(u.path):\(d.timeIntervalSince1970)") { overlay(image: img, mask: u, sky: sky) }
            }
            s.clearPlot = true
        case .cameras?, .finalising?, .downloading?, nil:
            let ds = dir.appendingPathComponent("dataset.json")
            let stamp = (try? ds.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? nil
            if let stamp, since.map({ stamp >= $0.addingTimeInterval(-5) }) ?? true, let p = loadPlot(dir) {
                s.plot = (p, stamp)
                s.heading = "Solved cameras"
                s.caption = "\(formatCount(p.cameras.count)) cameras · \(formatCount(p.points.count)) points shown"
                s.legend = [Legend(label: "training photo", color: Color(red: 0.38, green: 0.65, blue: 1)),
                            Legend(label: "held out for scoring", color: .orange)]
            } else {
                s.heading = step == .cameras ? "Camera solve" : "Preview"
                s.clearPlot = true
                for (u, d) in newest("images", 9) { add("i:\(u.path):\(d.timeIntervalSince1970)") { thumb(u, 260) } }
                if !s.tileKeys.isEmpty { s.caption = "\(formatCount(count("images"))) photos being placed" }
            }
        case .images?, .uploading?:
            s.heading = step == .uploading ? "Photos being sent" : "Latest photos"
            s.caption = "\(formatCount(count("images"))) in the dataset"
            for (u, d) in newest("images", 9) { add("i:\(u.path):\(d.timeIntervalSince1970)") { thumb(u, 260) } }
            s.clearPlot = true
        }
        return s
    }

    // MARK: image helpers

    nonisolated static func thumb(_ url: URL, _ maxPixels: Int) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: maxPixels,
                                     kCGImageSourceCreateThumbnailWithTransform: true]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
    }

    /// The photo with its mask tinted in: object masks (white = keep) tint what is left out red; sky masks
    /// (white = sky) tint the sky blue.
    nonisolated static func overlay(image: URL, mask: URL, sky: Bool) -> CGImage? {
        guard let im = thumb(image, 260) else { return nil }
        let w = im.width, h = im.height
        guard let mk = thumb(mask, max(w, h)) else { return im }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let mctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
                                   bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return im }
        ctx.draw(im, in: CGRect(x: 0, y: 0, width: w, height: h))
        mctx.interpolationQuality = .none
        mctx.draw(mk, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let px = ctx.data?.bindMemory(to: UInt8.self, capacity: w * h * 4),
              let mp = mctx.data?.bindMemory(to: UInt8.self, capacity: w * h) else { return im }
        let tint: (Float, Float, Float) = sky ? (60, 130, 255) : (255, 60, 60)
        for i in 0..<(w * h) {
            let on = sky ? mp[i] > 127 : mp[i] < 128
            guard on else { continue }
            for (c, t) in [(0, tint.0), (1, tint.1), (2, tint.2)] {
                px[i * 4 + c] = UInt8(Float(px[i * 4 + c]) * 0.45 + t * 0.55)
            }
        }
        return ctx.makeImage()
    }

    // MARK: camera map

    nonisolated static func loadPlot(_ dir: URL) -> CameraPlot? {
        guard let data = try? Data(contentsOf: dir.appendingPathComponent("dataset.json")),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let cams = obj["cameras"] as? [[String: Any]], !cams.isEmpty else { return nil }
        let upArr = (obj["upVector"] as? [Double]) ?? [0, -1, 0]
        var up = SIMD3<Float>(Float(upArr[0]), Float(upArr[1]), Float(upArr.count > 2 ? upArr[2] : 0))
        up = simd_normalize(up)
        var centres: [SIMD3<Float>] = [], fwd: [SIMD3<Float>] = [], hold: [Bool] = []
        for c in cams {
            guard let m = c["cameraToWorld"] as? [[Double]], m.count >= 3 else { continue }
            centres.append(SIMD3(Float(m[0][3]), Float(m[1][3]), Float(m[2][3])))
            fwd.append(SIMD3(Float(m[0][2]), Float(m[1][2]), Float(m[2][2])))   // OpenCV: +Z looks forward
            hold.append((c["holdout"] as? Bool) ?? false)
        }
        guard !centres.isEmpty else { return nil }
        // plane basis ⟂ up, with the cameras' main spread along the first axis
        let mean = centres.reduce(SIMD3<Float>(), +) / Float(centres.count)
        let helper: SIMD3<Float> = abs(up.x) < 0.9 ? [1, 0, 0] : [0, 1, 0]
        var a = simd_normalize(simd_cross(up, helper)), b = simd_normalize(simd_cross(up, a))
        var sxx: Float = 0, sxy: Float = 0, syy: Float = 0
        for p in centres { let d = p - mean; let x = simd_dot(d, a), y = simd_dot(d, b); sxx += x * x; sxy += x * y; syy += y * y }
        let ang = 0.5 * atan2(2 * sxy, sxx - syy)
        let (ca, sa) = (cos(ang), sin(ang))
        (a, b) = (ca * a + sa * b, -sa * a + ca * b)
        func proj(_ p: SIMD3<Float>) -> SIMD2<Float> { let d = p - mean; return SIMD2(simd_dot(d, a), simd_dot(d, b)) }
        var plot = CameraPlot()
        plot.cameras = zip(zip(centres, fwd), hold).map { (cf, h) in (proj(cf.0), SIMD2(simd_dot(cf.1, a), simd_dot(cf.1, b)), h) }
        if let pts = readPoints(dir.appendingPathComponent((obj["points"] as? String) ?? "points3d.ply"), limit: 20_000) {
            plot.points = pts.xyz.map(proj); plot.colors = pts.rgb
        }
        // frame the cameras plus the bulk of the points (2nd–98th percentile), so a few strays don't shrink the map
        var xs = plot.cameras.map(\.pos.x), ys = plot.cameras.map(\.pos.y)
        if !plot.points.isEmpty {
            let px = plot.points.map(\.x).sorted(), py = plot.points.map(\.y).sorted()
            let lo = px.count / 50, hi = px.count - 1 - px.count / 50
            xs += [px[lo], px[hi]]; ys += [py[lo], py[hi]]
        }
        plot.bounds = (SIMD2(xs.min()!, ys.min()!), SIMD2(xs.max()!, ys.max()!))
        return plot
    }

    /// A sample of a PLY point cloud's positions and colours (binary little-endian or ASCII; float/double/uchar).
    nonisolated static func readPoints(_ url: URL, limit: Int) -> (xyz: [SIMD3<Float>], rgb: [SIMD3<UInt8>])? {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe),
              let end = data.range(of: Data("end_header\n".utf8)) else { return nil }
        guard let header = String(data: data[..<end.lowerBound], encoding: .ascii) else { return nil }
        var count = 0, props: [(String, String)] = [], inVertex = false, binary = true
        for line in header.split(separator: "\n") {
            let t = line.split(separator: " ").map(String.init)
            if t.first == "format" { binary = t.count > 1 && t[1] == "binary_little_endian" }
            if t.first == "element" { inVertex = t.count > 2 && t[1] == "vertex"; if inVertex { count = Int(t[2]) ?? 0 } }
            if t.first == "property", inVertex, t.count >= 3 { props.append((t[1], t[2])) }
        }
        guard count > 0 else { return nil }
        let size: [String: Int] = ["float": 4, "float32": 4, "double": 8, "float64": 8, "uchar": 1, "uint8": 1, "char": 1, "int": 4, "uint": 4, "short": 2, "ushort": 2]
        var offsets: [String: (Int, String)] = [:], stride = 0
        for (ty, name) in props { offsets[name] = (stride, ty); stride += size[ty] ?? 4 }
        guard offsets["x"] != nil, offsets["y"] != nil, offsets["z"] != nil else { return nil }
        let step = max(1, count / limit)
        var xyz: [SIMD3<Float>] = [], rgb: [SIMD3<UInt8>] = []
        if binary {
            let base = end.upperBound
            guard data.count >= base + count * stride else { return nil }
            data.withUnsafeBytes { raw in
                func f(_ off: Int, _ ty: String) -> Float {
                    switch ty {
                    case "double", "float64": return Float(raw.loadUnaligned(fromByteOffset: off, as: Double.self))
                    case "uchar", "uint8": return Float(raw.load(fromByteOffset: off, as: UInt8.self))
                    default: return raw.loadUnaligned(fromByteOffset: off, as: Float.self)
                    }
                }
                for i in Swift.stride(from: 0, to: count, by: step) {
                    let o = base + i * stride
                    xyz.append(SIMD3(f(o + offsets["x"]!.0, offsets["x"]!.1), f(o + offsets["y"]!.0, offsets["y"]!.1), f(o + offsets["z"]!.0, offsets["z"]!.1)))
                    if let r = offsets["red"], let g = offsets["green"], let b = offsets["blue"] {
                        rgb.append(SIMD3(UInt8(clamping: Int(f(o + r.0, r.1))), UInt8(clamping: Int(f(o + g.0, g.1))), UInt8(clamping: Int(f(o + b.0, b.1)))))
                    } else { rgb.append(SIMD3(200, 200, 200)) }
                }
            }
        } else {
            guard let body = String(data: data[end.upperBound...], encoding: .ascii) else { return nil }
            let names = props.map(\.1)
            for (i, line) in body.split(separator: "\n").prefix(count).enumerated() where i % step == 0 {
                let v = line.split(separator: " ").compactMap { Float($0) }
                guard v.count >= names.count, let xi = names.firstIndex(of: "x") else { continue }
                xyz.append(SIMD3(v[xi], v[xi + 1], v[xi + 2]))
                if let ri = names.firstIndex(of: "red") { rgb.append(SIMD3(UInt8(clamping: Int(v[ri])), UInt8(clamping: Int(v[ri + 1])), UInt8(clamping: Int(v[ri + 2])))) }
                else { rgb.append(SIMD3(200, 200, 200)) }
            }
        }
        return (xyz, rgb)
    }
}
