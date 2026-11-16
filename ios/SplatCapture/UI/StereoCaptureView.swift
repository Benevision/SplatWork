import AVFoundation
import Observation
import SwiftUI

@MainActor
@Observable
final class StereoCaptureModel {
    let engine = StereoCaptureEngine()
    private(set) var phase: CaptureViewModel.Phase = .idle
    private(set) var stats = StereoStats()
    private(set) var recordingStart: Date?
    var lastSummary: CaptureSummary?
    var showFinished = false
    var errorMessage: String?

    init() {
        engine.onStats = { [weak self] s in Task { @MainActor in self?.stats = s } }
        engine.onError = { [weak self] m in Task { @MainActor in self?.errorMessage = m } }
    }

    func start(settings: CaptureSettings) {
        guard StereoCaptureEngine.isSupported else {
            errorMessage = "Stereo capture needs an iPhone with both a wide and an ultra-wide camera (iPhone 11 or newer)."
            return
        }
        engine.start(settings: settings)
    }

    func stop() { if phase == .idle { engine.stop() } }

    func toggle(settings: CaptureSettings) {
        switch phase {
        case .idle:
            if CaptureStore.freeSpace < 500_000_000 {
                errorMessage = "Less than 500 MB of storage is free. Delete old captures or free up space first."
                return
            }
            stats = StereoStats()
            engine.startRecording(settings: settings) { [weak self] r in
                Task { @MainActor in
                    guard let self else { return }
                    switch r {
                    case .success:
                        self.phase = .recording; self.recordingStart = Date()
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                    case .failure(let e): self.errorMessage = e.localizedDescription
                    }
                }
            }
        case .recording:
            phase = .finishing
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
            engine.stopRecording { [weak self] r in
                Task { @MainActor in
                    guard let self else { return }
                    self.phase = .idle; self.recordingStart = nil
                    switch r {
                    case .success(let s): self.lastSummary = s; self.showFinished = true
                        UINotificationFeedbackGenerator().notificationOccurred(.success)
                    case .failure(let e): self.errorMessage = e.localizedDescription
                    }
                }
            }
        case .finishing: break
        }
    }
}

/// Stereo mode of the Capture tab: a camera preview and a record button. Photo pairs from the wide and ultra-wide
/// cameras are taken at the set interval while recording.
struct StereoCaptureView: View {
    @Environment(AppSettings.self) private var settings
    @Environment(CaptureLibrary.self) private var library
    @Bindable var model: StereoCaptureModel
    let onLibrary: () -> Void
    let onSettings: () -> Void
    let onSend: (CaptureItem) -> Void

    var body: some View {
        ZStack {
            CameraPreview(session: model.engine.session).ignoresSafeArea()
            VStack(spacing: 10) {
                HStack(spacing: 8) {
                    pill(model.stats.hasCalibration || model.phase == .idle ? "Wide + ultra-wide" : "No calibration",
                         color: model.stats.hasCalibration || model.phase == .idle ? .green : .yellow)
                    Spacer()
                    if let start = model.recordingStart {
                        TimelineView(.periodic(from: start, by: 1)) { ctx in
                            let s = Int(ctx.date.timeIntervalSince(start))
                            Label(String(format: "%d:%02d", s / 60, s % 60), systemImage: "record.circle")
                                .font(.footnote.monospacedDigit().weight(.semibold)).foregroundStyle(.red)
                                .padding(.horizontal, 10).padding(.vertical, 6).background(.ultraThinMaterial, in: Capsule())
                        }
                    }
                    pill(model.stats.hasDepth ? "Stereo depth" : "No depth", color: model.stats.hasDepth ? .green : .secondary)
                }
                .padding(.top, 4)
                Spacer()
                if model.phase == .idle {
                    Text("Pairs from both rear cameras every \(String(format: "%.1f", settings.capture.stereoInterval)) s. Walk slowly; no LiDAR or phone tracking — your Mac solves the cameras.")
                        .font(.footnote).multilineTextAlignment(.center)
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 12)).foregroundStyle(.white.opacity(0.9))
                }
                if model.phase != .idle {
                    HStack(spacing: 22) {
                        stat("\(model.stats.pairs)", "pairs")
                        stat(model.stats.bytes.formattedBytes, "stored")
                        if model.stats.droppedBusy > 0 { stat("\(model.stats.droppedBusy)", "skipped (busy)") }
                    }
                    .padding(.horizontal, 16).padding(.vertical, 8).background(.ultraThinMaterial, in: Capsule())
                }
                HStack {
                    HStack { circle("square.stack.3d.up", "Library", onLibrary); Spacer() }.frame(maxWidth: .infinity)
                        .opacity(model.phase == .idle ? 1 : 0).disabled(model.phase != .idle)
                    RecordButton(phase: model.phase) { model.toggle(settings: settings.capture) }
                    HStack { Spacer(); circle("gearshape", "Settings", onSettings) }.frame(maxWidth: .infinity)
                        .opacity(model.phase == .idle ? 1 : 0).disabled(model.phase != .idle)
                }
            }
            .padding(.horizontal, 16).padding(.bottom, 8)
        }
        .onAppear { model.start(settings: settings.capture) }
        .onDisappear { model.stop() }
        .sheet(isPresented: $model.showFinished) {
            if let s = model.lastSummary {
                NavigationStack {
                    List {
                        Section {
                            LabeledContent("Name", value: s.name)
                            LabeledContent("Photo pairs", value: "\(s.frameCount)")
                            LabeledContent("Size", value: s.bytes.formattedBytes)
                        }
                        if s.frameCount < 30 {
                            Label("That's only a few pairs. Most scenes need 80–300, covering every side.", systemImage: "exclamationmark.triangle")
                                .font(.footnote).foregroundStyle(.secondary)
                        }
                        Section {
                            Button {
                                model.showFinished = false
                                Task {
                                    await library.reload()
                                    try? await Task.sleep(for: .milliseconds(500))
                                    if let item = library.item(at: s.url) { onSend(item) }
                                }
                            } label: { Label("Send to Mac", systemImage: "desktopcomputer.and.arrow.down") }
                            Button { model.showFinished = false; onLibrary() } label: { Label("Open Library", systemImage: "square.stack.3d.up") }
                        }
                    }
                    .navigationTitle("Capture saved").navigationBarTitleDisplayMode(.inline)
                    .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { model.showFinished = false } } }
                }
                .presentationDetents([.medium])
            }
        }
        .alert("SplatWork Capture", isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(model.errorMessage ?? "") }
    }

    private func pill(_ text: String, color: Color) -> some View {
        Text(text).font(.caption.weight(.semibold)).foregroundStyle(color)
            .padding(.horizontal, 10).padding(.vertical, 6).background(.ultraThinMaterial, in: Capsule())
    }

    private func stat(_ value: String, _ label: String) -> some View {
        VStack(spacing: 0) {
            Text(value).font(.headline.monospacedDigit())
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func circle(_ icon: String, _ label: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon).font(.title3.weight(.semibold)).frame(width: 54, height: 54).background(.ultraThinMaterial, in: Circle())
        }
        .buttonStyle(.plain).accessibilityLabel(label)
    }
}

/// AVCaptureVideoPreviewLayer host.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.preview.session = session
        v.preview.videoGravity = .resizeAspectFill
        return v
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {}
}
