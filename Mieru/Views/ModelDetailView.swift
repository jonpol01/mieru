//
//  ModelDetailView.swift
//  Mieru
//
//  Detail page for a single model — stats, info, configuration, activate/unload action.
//

import SwiftUI

struct ModelDetailView: View {

    let model: ModelInfo
    let vlmService: VLMService
    let speechService: SpeechService
    let irodoriService: IrodoriTTSService
    let onActivateVLM: (String) async -> Void
    let onActivateTTS: (String) async -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var catalog = ModelCatalog.shared
    @State private var isActivating = false

    private let irodoriId = "mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit"
    private var isIrodori: Bool { model.id == irodoriId }

    private var isActive: Bool {
        switch model.kind {
        case .vlm:
            return vlmService.currentModelId == model.id && vlmService.isLoaded
        case .tts:
            return isIrodori ? irodoriService.isLoaded : speechService.isLoaded
        case .stt:
            return false
        }
    }

    private var isDownloaded: Bool {
        catalog.isDownloaded(model)
    }

    private var isLoading: Bool {
        switch model.kind {
        case .vlm:
            return vlmService.currentModelId == model.id && vlmService.isDownloading
        case .tts:
            return isIrodori ? irodoriService.isLoading : speechService.isLoading
        case .stt:
            return false
        }
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            ScrollView {
                VStack(spacing: 20) {
                    closeButton
                    headerIcon
                    titleSection
                    statsGrid
                    if let warning = model.compatibilityWarning {
                        compatibilityWarningView(warning)
                    }
                    modelInfoCard
                    configCard
                    if isIrodori { voicePicker }
                    actionButton
                    Spacer(minLength: 40)
                }
                .padding(.top, 8)
            }
        }
    }

    // MARK: - Close Button

    private var closeButton: some View {
        HStack {
            Spacer()
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 28))
                    .foregroundColor(.gray)
            }
        }
        .padding(.horizontal, 16)
    }

    // MARK: - Header Icon

    private var headerIcon: some View {
        ZStack {
            Circle()
                .fill(model.family.color.opacity(0.18))
                .frame(width: 80, height: 80)
            Image(systemName: model.kind.icon)
                .font(.system(size: 36))
                .foregroundColor(model.family.color)
        }
    }

    // MARK: - Title Section

    private var titleSection: some View {
        VStack(spacing: 10) {
            Text(model.displayName)
                .font(.system(size: 22, weight: .bold))
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)

            Text(model.family.displayName)
                .font(.system(size: 12, weight: .heavy))
                .foregroundColor(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 5)
                .background(model.family.color)
                .clipShape(Capsule())

            HStack(spacing: 6) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(statusText)
                    .font(.system(size: 13))
                    .foregroundColor(statusColor)
            }
        }
    }

    private var statusText: String {
        if isLoading { return "Loading…" }
        if isActive { return "Loaded & Active" }
        if isDownloaded { return "Downloaded" }
        return "Not downloaded"
    }

    private var statusColor: Color {
        if isActive { return .green }
        if isDownloaded { return .green.opacity(0.7) }
        if isLoading { return .orange }
        return .gray
    }

    // MARK: - Compatibility Warning

    private func compatibilityWarningView(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.yellow)
                .font(.system(size: 16))
            VStack(alignment: .leading, spacing: 4) {
                Text("Compatibility Warning")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundColor(.yellow)
                Text(message)
                    .font(.system(size: 12))
                    .foregroundColor(.white.opacity(0.9))
            }
            Spacer()
        }
        .padding(14)
        .background(Color.yellow.opacity(0.12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(Color.yellow.opacity(0.4), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 16)
    }

    // MARK: - Stats Grid

    private var statsGrid: some View {
        HStack(spacing: 0) {
            statTile(label: "Context", value: contextLabel, icon: "doc.text")
            Divider().background(Color.white.opacity(0.1)).frame(maxHeight: 60)
            statTile(label: "Batch", value: "\(model.batch)", icon: "rectangle.stack")
            Divider().background(Color.white.opacity(0.1)).frame(maxHeight: 60)
            statTile(label: "Size", value: model.sizeLabel, icon: "externaldrive")
            Divider().background(Color.white.opacity(0.1)).frame(maxHeight: 60)
            statTile(label: "Chunks", value: "\(model.chunks)", icon: "square.grid.2x2")
        }
        .padding(.vertical, 8)
        .background(Color.white.opacity(0.04))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 16)
    }

    private var contextLabel: String {
        if model.contextLength == 0 { return "—" }
        if model.contextLength >= 1000 {
            return "\(model.contextLength / 1000)K"
        }
        return "\(model.contextLength)"
    }

    private func statTile(label: String, value: String, icon: String) -> some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 13))
                .foregroundColor(.gray)
            Text(value)
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
            Text(label)
                .font(.system(size: 10))
                .foregroundColor(.gray)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
    }

    // MARK: - Model Information

    private var modelInfoCard: some View {
        infoSection(title: "Model Information", icon: "info.circle.fill", color: .orange) {
            infoRow(label: "Model ID", value: model.shortId)
            infoRow(label: "Size", value: model.sizeLabel)
            infoRow(label: "Architecture", value: model.architecture)
        }
    }

    // MARK: - Configuration

    private var configCard: some View {
        infoSection(title: "Configuration", icon: "gearshape.fill", color: .orange) {
            infoRowWithBadge(label: "Version", badge: model.format.displayName, badgeColor: model.format.color)
            infoRow(label: "Mode Type", value: model.modelType)
        }
    }

    // MARK: - Irodori Voice Picker

    private var voicePicker: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "person.wave.2.fill")
                    .foregroundColor(.orange)
                Text("Voice (VoiceDesign)")
                    .foregroundColor(.white)
                    .font(.system(size: 15, weight: .semibold))
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color.white.opacity(0.06))

            VStack(spacing: 0) {
                ForEach(IrodoriTTSService.voices, id: \.self) { caption in
                    Button {
                        irodoriService.selectedVoice = caption
                    } label: {
                        HStack {
                            Text(caption)
                                .font(.system(size: 14))
                                .foregroundColor(.white)
                                .multilineTextAlignment(.leading)
                            Spacer()
                            if (irodoriService.selectedVoice ?? IrodoriTTSService.voices.first) == caption {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                            } else {
                                Image(systemName: "circle")
                                    .foregroundColor(.gray.opacity(0.5))
                            }
                        }
                        .padding(.vertical, 10)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
            .background(Color.white.opacity(0.03))
        }
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 16)
    }

    // MARK: - Action Button

    @ViewBuilder
    private var actionButton: some View {
        if isLoading {
            loadingButton
        } else if isActive {
            // VLM: offer Unload. TTS: the active engine just shows an Active badge —
            // switching engines is done by activating the other one.
            if model.kind == .vlm {
                unloadButton
            } else {
                activeBadgeButton
            }
        } else if isDownloaded {
            activateButton(label: "Activate", color: .green)
        } else {
            activateButton(label: "Download & Activate", color: .blue)
        }
    }

    private var activeBadgeButton: some View {
        Text("Active")
            .font(.system(size: 16, weight: .bold))
            .foregroundColor(.green)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(Color.green.opacity(0.15))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal, 16)
    }

    private var loadingButton: some View {
        VStack(spacing: 8) {
            HStack {
                ProgressView().tint(.white)
                Text(loadingStatusText)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.white)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(Color.orange.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 12))

            if model.kind == .vlm, vlmService.downloadProgress > 0 {
                ProgressView(value: Double(vlmService.downloadProgress))
                    .tint(.orange)
            }
        }
        .padding(.horizontal, 16)
    }

    private var loadingStatusText: String {
        switch model.kind {
        case .vlm: return vlmService.statusMessage.isEmpty ? "Loading…" : vlmService.statusMessage
        case .tts:
            let msg = isIrodori ? irodoriService.statusMessage : speechService.statusMessage
            return msg.isEmpty ? "Loading…" : msg
        case .stt: return "Loading…"
        }
    }

    private var unloadButton: some View {
        Button {
            Task {
                if model.kind == .vlm {
                    vlmService.unload()
                }
            }
        } label: {
            Text("Unload")
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Color.orange)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .padding(.horizontal, 16)
        // TTS doesn't support unload from UI — Active state shown as badge instead
        .disabled(model.kind == .tts)
        .opacity(model.kind == .tts ? 0.4 : 1.0)
    }

    private func activateButton(label: String, color: Color) -> some View {
        Button {
            activate()
        } label: {
            HStack {
                if isActivating {
                    ProgressView().tint(.white)
                } else {
                    Image(systemName: "play.fill")
                }
                Text(isActivating ? "Activating…" : label)
                    .font(.system(size: 16, weight: .bold))
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .background(color)
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .disabled(isActivating || model.kind == .stt)
        .padding(.horizontal, 16)
    }

    private func activate() {
        guard !isActivating else { return }
        isActivating = true

        Task {
            switch model.kind {
            case .vlm:
                // ContentView handles catalog updates on success/failure
                await onActivateVLM(model.id)
            case .tts:
                // ContentView switches engines (stops both, unloads the other, loads this)
                await onActivateTTS(model.id)
            case .stt:
                break
            }
            isActivating = false
        }
    }

    // MARK: - Section Builders

    private func infoSection<Content: View>(
        title: String,
        icon: String,
        color: Color,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .foregroundColor(color)
                Text(title)
                    .foregroundColor(.white)
                    .font(.system(size: 15, weight: .semibold))
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color.white.opacity(0.06))

            VStack(spacing: 0) {
                content()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color.white.opacity(0.03))
        }
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 16)
    }

    private func infoRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 14))
                .foregroundColor(.gray)
            Spacer()
            Text(value)
                .font(.system(size: 14, design: .monospaced))
                .foregroundColor(.white)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(.vertical, 8)
    }

    private func infoRowWithBadge(label: String, badge: String, badgeColor: Color) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 14))
                .foregroundColor(.gray)
            Spacer()
            Text(badge)
                .font(.system(size: 11, weight: .heavy))
                .foregroundColor(.white)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(badgeColor)
                .clipShape(Capsule())
        }
        .padding(.vertical, 8)
    }
}
