//
//  ModelsView.swift
//  Mieru
//
//  Models list page — VLMs, TTS, storage.
//  Uses ModelCatalog as the single source of truth.
//

import SwiftUI

struct ModelsView: View {

    let vlmService: VLMService
    let speechService: SpeechService
    let irodoriService: IrodoriTTSService
    let onActivateVLM: (String) async -> Void
    let onActivateTTS: (String) async -> Void

    private let irodoriId = "mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit"

    @Environment(\.dismiss) private var dismiss
    @State private var catalog = ModelCatalog.shared
    @State private var selectedModel: ModelInfo?
    @State private var showAddModel = false

    private var vlmModels: [ModelInfo] {
        catalog.models.filter { $0.kind == .vlm }
    }

    private var ttsModels: [ModelInfo] {
        catalog.models.filter { $0.kind == .tts }
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 24) {
                        visionSection
                        voiceSection
                        storageSection
                        Spacer(minLength: 40)
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
                }
            }
            .navigationTitle("Models")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(Color.black, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { dismiss() }
                        .foregroundColor(.blue)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showAddModel = true
                    } label: {
                        Image(systemName: "plus")
                            .foregroundColor(.blue)
                    }
                }
            }
            .sheet(item: $selectedModel) { model in
                ModelDetailView(
                    model: model,
                    vlmService: vlmService,
                    speechService: speechService,
                    irodoriService: irodoriService,
                    onActivateVLM: onActivateVLM,
                    onActivateTTS: onActivateTTS
                )
            }
            .sheet(isPresented: $showAddModel) {
                AddModelView()
            }
            .task {
                await catalog.scanCache()
            }
        }
    }

    // MARK: - Vision Section

    private var visionSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader(
                title: "Vision",
                subtitle: "VLM",
                icon: "eye.fill",
                color: .orange
            )

            ForEach(vlmModels) { model in
                modelCard(model)
            }
        }
    }

    // MARK: - Voice Section

    private var voiceSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader(
                title: "Voice",
                subtitle: "TTS",
                icon: "speaker.wave.2.fill",
                color: .green
            )

            ForEach(ttsModels) { model in
                modelCard(model)
            }
        }
    }

    // MARK: - Storage Section

    private var storageSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader(
                title: "Storage",
                subtitle: nil,
                icon: "internaldrive.fill",
                color: .blue
            )

            VStack(spacing: 12) {
                HStack {
                    Image(systemName: "arrow.down.circle.fill")
                        .foregroundColor(.blue)
                    Text("Downloaded Models")
                        .foregroundColor(.white)
                        .font(.system(size: 14))
                    Spacer()
                    Text(catalog.totalDiskLabel)
                        .foregroundColor(.gray)
                        .font(.system(size: 14, design: .monospaced))
                }

                Divider().background(Color.white.opacity(0.1))

                HStack(spacing: 20) {
                    storageStat(
                        label: "Total",
                        value: "\(catalog.models.count)"
                    )
                    storageStat(
                        label: "Downloaded",
                        value: "\(catalog.models.filter { catalog.isDownloaded($0) }.count)"
                    )
                    storageStat(
                        label: "Available",
                        value: "\(catalog.models.filter { !catalog.isDownloaded($0) }.count)"
                    )
                }
            }
            .padding(14)
            .background(Color.white.opacity(0.04))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }

    private func storageStat(label: String, value: String) -> some View {
        VStack(spacing: 4) {
            Text(value)
                .foregroundColor(.white)
                .font(.system(size: 18, weight: .bold))
            Text(label)
                .foregroundColor(.gray)
                .font(.system(size: 11))
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Helpers

    private func sectionHeader(title: String, subtitle: String?, icon: String, color: Color) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundColor(color)
                .font(.system(size: 16, weight: .semibold))
            Text(title)
                .foregroundColor(.white)
                .font(.system(size: 18, weight: .bold))
            if let subtitle {
                Text(subtitle)
                    .foregroundColor(color)
                    .font(.system(size: 11, weight: .heavy))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(color.opacity(0.15))
                    .clipShape(Capsule())
            }
            Spacer()
        }
    }

    private func modelCard(_ model: ModelInfo) -> some View {
        let isActive = isModelActive(model)
        let isDownloaded = catalog.isDownloaded(model)
        let isLoading = isModelLoading(model)

        return Button {
            selectedModel = model
        } label: {
            HStack(spacing: 14) {
                // Icon
                ZStack {
                    Circle()
                        .fill(model.family.color.opacity(0.18))
                        .frame(width: 44, height: 44)
                    Image(systemName: model.kind.icon)
                        .foregroundColor(model.family.color)
                        .font(.system(size: 18))
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(model.displayName)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.white)
                        .lineLimit(1)

                    HStack(spacing: 6) {
                        Text(model.family.displayName)
                            .font(.system(size: 9, weight: .heavy))
                            .foregroundColor(.white)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(model.family.color)
                            .clipShape(Capsule())

                        Text(model.sizeLabel)
                            .font(.system(size: 12))
                            .foregroundColor(.gray)
                    }
                }

                Spacer()

                statusBadge(isActive: isActive, isDownloaded: isDownloaded, isLoading: isLoading, model: model)
            }
            .padding(14)
            .background(Color.white.opacity(0.04))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(
                        isActive ? model.family.color.opacity(0.5) : Color.clear,
                        lineWidth: 1.5
                    )
            )
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private func statusBadge(isActive: Bool, isDownloaded: Bool, isLoading: Bool, model: ModelInfo) -> some View {
        if isLoading {
            VStack(alignment: .trailing, spacing: 4) {
                ProgressView()
                    .tint(.orange)
                    .scaleEffect(0.8)
                if !loadingStatus(for: model).isEmpty {
                    Text(loadingStatus(for: model))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.gray)
                        .lineLimit(1)
                }
            }
        } else if isActive {
            Text("Active")
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(.orange)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Color.orange.opacity(0.18))
                .clipShape(Capsule())
        } else if isDownloaded {
            Image(systemName: "arrow.down.circle.fill")
                .foregroundColor(.green.opacity(0.7))
                .font(.system(size: 18))
        } else {
            Image(systemName: "chevron.right")
                .foregroundColor(.gray)
                .font(.system(size: 14, weight: .semibold))
        }
    }

    // MARK: - State Helpers

    private func isModelActive(_ model: ModelInfo) -> Bool {
        switch model.kind {
        case .vlm:
            return vlmService.currentModelId == model.id && vlmService.isLoaded
        case .tts:
            return model.id == irodoriId ? irodoriService.isLoaded : speechService.isLoaded
        case .stt:
            return false
        }
    }

    private func isModelLoading(_ model: ModelInfo) -> Bool {
        switch model.kind {
        case .vlm:
            return vlmService.currentModelId == model.id && vlmService.isDownloading
        case .tts:
            return model.id == irodoriId ? irodoriService.isLoading : speechService.isLoading
        case .stt:
            return false
        }
    }

    private func loadingStatus(for model: ModelInfo) -> String {
        switch model.kind {
        case .vlm:
            return vlmService.statusMessage
        case .tts:
            return model.id == irodoriId ? irodoriService.statusMessage : speechService.statusMessage
        case .stt:
            return ""
        }
    }
}
