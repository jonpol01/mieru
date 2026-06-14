//
//  ModelCatalog.swift
//  Mieru
//
//  Registry of available models, active selection, and persistence.
//

import Foundation
import SwiftUI

@MainActor
@Observable
class ModelCatalog {

    static let shared = ModelCatalog()

    /// All known models (built-in + custom).
    var models: [ModelInfo] = []

    /// Currently active VLM (vision) model ID.
    var activeVLMId: String?

    /// Currently active TTS model ID.
    var activeTTSId: String?

    /// IDs of models that have been successfully downloaded.
    private(set) var downloadedIds: Set<String> = []

    /// Total disk usage across all downloaded models (bytes).
    var totalDiskBytes: Int64 = 0

    private let customModelsKey = "mieru.customModels"
    private let downloadedIdsKey = "mieru.downloadedIds"
    private let activeVLMIdKey = "mieru.activeVLMId"
    private let activeTTSIdKey = "mieru.activeTTSId"

    private init() {
        loadFromUserDefaults()
        rebuildModels()
    }

    // MARK: - Built-in Models

    static let builtIn: [ModelInfo] = [
        // VLMs — Gemma 4
        ModelInfo(
            id: "mlx-community/gemma-4-e2b-it-4bit",
            displayName: "Gemma 4 E2B 4-bit",
            family: .gemma, kind: .vlm,
            architecture: "Gemma 4", modelType: "Gemma4_VLM",
            format: .mlx,
            contextLength: 131072, batch: 1, chunks: 1,
            estimatedSizeGB: 3.6
        ),
        // QAT variant — Google's quantization-aware training, slightly larger but better quality.
        // KV-sharing (layers 15+) is supported by the official mlx-swift-lm @ a47894a.
        ModelInfo(
            id: "mlx-community/gemma-4-E2B-it-qat-4bit",
            displayName: "Gemma 4 E2B QAT 4-bit",
            family: .gemma, kind: .vlm,
            architecture: "Gemma 4", modelType: "Gemma4_QAT",
            format: .mlx,
            contextLength: 131072, batch: 1, chunks: 1,
            estimatedSizeGB: 4.36
        ),

        // TTS — Kokoro (lightweight, CoreML / Neural Engine, EN + JA)
        ModelInfo(
            id: "aufklarer/Kokoro-82M-CoreML",
            displayName: "Kokoro-82M CoreML",
            family: .kokoro, kind: .tts,
            architecture: "Kokoro", modelType: "Kokoro_TTS",
            format: .coreml,
            contextLength: 128, batch: 1, chunks: 1,
            estimatedSizeGB: 0.16
        ),

        // TTS — Irodori (Japanese flow-matching DiT, MLX, VoiceDesign captions, 48 kHz)
        ModelInfo(
            id: "mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit",
            displayName: "Irodori-TTS 600M (JP)",
            family: .other, kind: .tts,
            architecture: "Irodori-TTS", modelType: "Irodori_MLXAudio",
            format: .mlx,
            contextLength: 0, batch: 1, chunks: 1,
            estimatedSizeGB: 1.3
        )
    ]

    private var customModels: [ModelInfo] = []

    // MARK: - Catalog Operations

    private func rebuildModels() {
        models = Self.builtIn + customModels
    }

    func addCustomModel(_ model: ModelInfo) {
        guard !models.contains(where: { $0.id == model.id }) else { return }
        customModels.append(model)
        rebuildModels()
        persistCustomModels()
    }

    func removeCustomModel(_ model: ModelInfo) {
        customModels.removeAll { $0.id == model.id }
        rebuildModels()
        persistCustomModels()
    }

    func model(id: String) -> ModelInfo? {
        models.first(where: { $0.id == id })
    }

    // MARK: - Active Selection

    func setActiveVLM(_ model: ModelInfo) {
        activeVLMId = model.id
        UserDefaults.standard.set(model.id, forKey: activeVLMIdKey)
    }

    func setActiveTTS(_ model: ModelInfo) {
        activeTTSId = model.id
        UserDefaults.standard.set(model.id, forKey: activeTTSIdKey)
    }

    var defaultVLMId: String {
        activeVLMId ?? "mlx-community/gemma-4-e2b-it-4bit"
    }

    var defaultTTSId: String {
        activeTTSId ?? "aufklarer/Kokoro-82M-CoreML"
    }

    // MARK: - Downloaded Tracking

    func isDownloaded(_ model: ModelInfo) -> Bool {
        downloadedIds.contains(model.id) || cacheExists(for: model)
    }

    func markDownloaded(_ id: String) {
        downloadedIds.insert(id)
        persistDownloadedIds()
        Task { await scanCache() }
    }

    // MARK: - Persistence

    private func loadFromUserDefaults() {
        if let data = UserDefaults.standard.data(forKey: customModelsKey),
           let decoded = try? JSONDecoder().decode([ModelInfo].self, from: data) {
            customModels = decoded
        }
        if let ids = UserDefaults.standard.array(forKey: downloadedIdsKey) as? [String] {
            downloadedIds = Set(ids)
        }
        activeVLMId = UserDefaults.standard.string(forKey: activeVLMIdKey)
        activeTTSId = UserDefaults.standard.string(forKey: activeTTSIdKey)
    }

    private func persistCustomModels() {
        if let data = try? JSONEncoder().encode(customModels) {
            UserDefaults.standard.set(data, forKey: customModelsKey)
        }
    }

    private func persistDownloadedIds() {
        UserDefaults.standard.set(Array(downloadedIds), forKey: downloadedIdsKey)
    }

    // MARK: - Cache Scanning

    /// Walks HuggingFace cache directories and computes disk usage per model.
    func scanCache() async {
        let fm = FileManager.default
        var totalBytes: Int64 = 0
        var updatedModels = models

        for (idx, model) in updatedModels.enumerated() {
            if let bytes = cacheSizeBytes(for: model) {
                updatedModels[idx].actualSizeBytes = bytes
                totalBytes += bytes
                downloadedIds.insert(model.id)
            } else {
                updatedModels[idx].actualSizeBytes = nil
            }
        }

        _ = fm  // silence unused warning if any
        self.models = updatedModels
        self.totalDiskBytes = totalBytes
        persistDownloadedIds()
    }

    private func cacheExists(for model: ModelInfo) -> Bool {
        cacheSizeBytes(for: model) != nil
    }

    /// Computes total bytes of cached model files. Returns nil if not present.
    private func cacheSizeBytes(for model: ModelInfo) -> Int64? {
        let fm = FileManager.default
        let candidates = cacheCandidatePaths(for: model)

        for path in candidates where fm.fileExists(atPath: path.path) {
            if let size = directorySize(at: path) {
                return size
            }
        }
        return nil
    }

    /// Possible HuggingFace cache paths for a model.
    private func cacheCandidatePaths(for model: ModelInfo) -> [URL] {
        let orgName = model.id.replacingOccurrences(of: "/", with: "--")
        let dirName = "models--\(orgName)"

        var paths: [URL] = []
        let fm = FileManager.default

        if let caches = try? fm.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
            paths.append(caches.appendingPathComponent("huggingface/hub/\(dirName)"))
            paths.append(caches.appendingPathComponent("qwen3-speech/\(dirName)"))
        }
        if let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
            paths.append(docs.appendingPathComponent("huggingface/hub/\(dirName)"))
        }
        if let app = try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: false) {
            paths.append(app.appendingPathComponent("huggingface/hub/\(dirName)"))
        }
        return paths
    }

    private func directorySize(at url: URL) -> Int64? {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) else {
            return nil
        }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let resourceValues = try? fileURL.resourceValues(forKeys: [.fileSizeKey]),
               let size = resourceValues.fileSize {
                total += Int64(size)
            }
        }
        return total > 0 ? total : nil
    }

    var totalDiskLabel: String {
        let gb = Double(totalDiskBytes) / 1_073_741_824
        if gb < 1.0 {
            let mb = Double(totalDiskBytes) / 1_048_576
            return String(format: "%.1f MB", mb)
        }
        return String(format: "%.2f GB", gb)
    }
}
