//
//  VLMService.swift
//  Mieru
//
//  On-device vision-language model service (Gemma 4 family).
//  Supports switching between multiple Gemma 4 model variants.
//

import CoreImage
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXRandom
import MLXVLM
import Tokenizers

@Observable
@MainActor
class VLMService {

    public var output = ""
    public var isLoaded = false
    public var running = false
    public var statusMessage = ""
    public var downloadProgress: Float = 0
    public var downloadedBytes: Int64 = 0
    public var totalBytes: Int64 = 0
    public var isDownloading = false

    /// The currently loaded/loading model ID.
    public var currentModelId: String = "mlx-community/gemma-4-e2b-it-4bit"

    private let generateParameters = GenerateParameters(temperature: 0.6)
    private let maxTokens = 150

    public var isReady: Bool { isLoaded && !running }

    private enum LoadState {
        case idle
        case loaded(modelId: String, container: ModelContainer)
    }
    private var loadState = LoadState.idle

    // MARK: - Load

    /// Load the current model (no-op if already loaded).
    public func load() async {
        do { _ = try await _load(modelId: currentModelId) }
        catch {
            statusMessage = "Load error: \(error.localizedDescription)"
            NSLog("[VLM] %@", statusMessage)
        }
    }

    /// Switch to a different model ID — unloads current, loads new.
    /// Returns true on success, false on failure (in which case previous model is restored).
    @discardableResult
    public func switchModel(to modelId: String) async -> Bool {
        // No-op if already loaded with same id
        if case .loaded(let loadedId, _) = loadState, loadedId == modelId {
            return true
        }

        let previousId = currentModelId
        unload()
        currentModelId = modelId

        Self.log("switch → \(modelId) (from \(previousId))")
        do {
            _ = try await _load(modelId: modelId)
            Self.log("switch OK \(modelId)")
            return true
        } catch {
            // localizedDescription is terse for Swift error enums — log the full value.
            let errMsg = error.localizedDescription
            Self.log("switch FAILED \(modelId): \(String(describing: error))")
            NSLog("[VLM] Load failed for %@: %@", modelId, errMsg)

            // Restore previous model if switching from a different one
            if previousId != modelId {
                NSLog("[VLM] Restoring previous model: %@", previousId)
                unload()
                currentModelId = previousId
                statusMessage = "Failed: \(errMsg). Restoring previous model…"
                do {
                    _ = try await _load(modelId: previousId)
                    statusMessage = "Reverted — \(errMsg)"
                } catch {
                    statusMessage = "Failed to restore: \(error.localizedDescription)"
                }
            } else {
                statusMessage = "Load error: \(errMsg)"
            }
            return false
        }
    }

    private func _load(modelId: String) async throws -> ModelContainer {
        // If a different model is loaded, unload first
        if case .loaded(let loadedId, let container) = loadState {
            if loadedId == modelId { return container }
            unload()
        }

        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
        statusMessage = "Downloading…"
        let config = ModelConfiguration(id: modelId)

        isDownloading = true
        downloadProgress = 0
        downloadedBytes = 0
        totalBytes = 0

        let container = try await #huggingFaceLoadModelContainer(
            configuration: config
        ) { [weak self] progress in
            Task { @MainActor in
                let frac = Float(progress.fractionCompleted)
                self?.downloadProgress = frac
                self?.downloadedBytes = progress.completedUnitCount
                self?.totalBytes = progress.totalUnitCount
                let pct = Int(frac * 100)
                if progress.totalUnitCount > 0 {
                    let dlGB = Double(progress.completedUnitCount) / 1_073_741_824
                    let totGB = Double(progress.totalUnitCount) / 1_073_741_824
                    self?.statusMessage = String(format: "%.2f / %.2f GB  (%d%%)", dlGB, totGB, pct)
                } else {
                    self?.statusMessage = "Loading: \(pct)%"
                }
            }
        }

        isDownloading = false
        statusMessage = "Ready"
        isLoaded = true
        loadState = .loaded(modelId: modelId, container: container)
        ModelCatalog.shared.markDownloaded(modelId)
        return container
    }

    // MARK: - Inference

    public func describe(pixelBuffer: CVPixelBuffer, language: String = "ja") async -> String {
        guard !running else { return output }
        running = true
        defer { running = false }

        let (systemPrompt, userPrompt) = language == "ja"
            ? ("画像に映っているものを具体的に識別してください。ブランド名、商品名、人物、場所など、わかるものはそのまま名前で答えてください。簡潔に1〜3文で。日本語で答えてください。",
               "これは何？")
            : ("Identify what you see in the image. Name specific brands, products, people, places, or objects directly. Be concise, 1-3 sentences.",
               "What is this?")

        do {
            let container = try await _load(modelId: currentModelId)
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let targetSize: CGFloat = 512
            let scale = min(targetSize / ciImage.extent.width, targetSize / ciImage.extent.height)
            let downscaled = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

            let userInput = UserInput(
                chat: [
                    .system(systemPrompt),
                    .user(userPrompt, images: [.ciImage(downscaled)])
                ],
                additionalContext: ["enable_thinking": false]
            )

            let tokenLimit = self.maxTokens
            let result = try await container.perform { context in
                let input = try await context.processor.prepare(input: userInput)
                return try MLXLMCommon.generate(
                    input: input, parameters: self.generateParameters, context: context
                ) { tokens in tokens.count >= tokenLimit ? .stop : .more }
            }

            self.output = result.output
            return result.output
        } catch {
            let msg = "VLM error: \(error.localizedDescription)"
            self.output = msg; NSLog("[VLM] %@", msg)
            return msg
        }
    }

    public func unload() {
        loadState = .idle
        isLoaded = false
        running = false
        output = ""
        statusMessage = ""
        downloadProgress = 0
        MLX.GPU.clearCache()
    }

    // MARK: - Diagnostics

    /// File-based log (device console is flaky). Pull via:
    /// xcrun devicectl device copy from … Documents/vlm.log
    nonisolated static func log(_ line: String) {
        let mem = Int(MLX.GPU.activeMemory / (1024 * 1024))
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let url = docs.appendingPathComponent("vlm.log")
        let entry = "[\(ISO8601DateFormatter().string(from: Date()))] [\(mem)MB] \(line)\n"
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(Data(entry.utf8)); try? h.close()
        } else {
            try? entry.data(using: .utf8)?.write(to: url)
        }
        NSLog("[VLM] %@", line)
    }
}
