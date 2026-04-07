//
//  VLMService.swift
//  Mieru
//
//  On-device VLM service using MLX Swift.
//

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import MLXVLM

// MARK: - VLMService

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

    private let generateParameters = GenerateParameters(temperature: 0.6)
    private let maxTokens = 150

    public var isReady: Bool { isLoaded && !running }

    private enum LoadState { case idle, loaded(ModelContainer) }
    private var loadState = LoadState.idle

    public func load() async {
        do { _ = try await _load() }
        catch { statusMessage = "Load error: \(error.localizedDescription)"; NSLog("[VLM] %@", statusMessage) }
    }

    private func _load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
            statusMessage = "Downloading Gemma 4 E2B…"
            isDownloading = true; downloadProgress = 0; downloadedBytes = 0; totalBytes = 0

            let config = ModelConfiguration(id: "mlx-community/gemma-4-e2b-it-4bit")
            let container = try await VLMModelFactory.shared.loadContainer(
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

            isDownloading = false; statusMessage = "Ready"; isLoaded = true
            loadState = .loaded(container)
            return container
        case .loaded(let container):
            return container
        }
    }

    public func describe(pixelBuffer: CVPixelBuffer) async -> String {
        guard !running else { return output }
        running = true
        defer { running = false }

        do {
            let container = try await _load()
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let targetSize: CGFloat = 512
            let scale = min(targetSize / ciImage.extent.width, targetSize / ciImage.extent.height)
            let downscaled = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

            let userInput = UserInput(
                chat: [
                    .system("あなたはファンタジーRPGのナレーターです。画像に映っているものを、勇者が遭遇した場面として生き生きと、しかし簡潔に描写してください。1〜3文で。現在形を使い、日本語で答えてください。"),
                    .user("何が見える？", images: [.ciImage(downscaled)])
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
        loadState = .idle; isLoaded = false; running = false; output = ""
        MLX.GPU.clearCache()
    }
}
