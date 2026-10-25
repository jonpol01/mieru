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
import os
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

    /// Last generation throughput (tokens/sec) and token count — for the perf HUD.
    public var lastTokensPerSecond: Double = 0
    public var lastTokenCount: Int = 0

    private let generateParameters = GenerateParameters(maxTokens: 150, temperature: 0.6)

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
            Self.log("load FAILED \(currentModelId): \(String(describing: error))")
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

        MLX.Memory.cacheLimit = 20 * 1024 * 1024
        statusMessage = "Downloading…"
        let config = Self.localModelDirectory(modelId).map { ModelConfiguration(directory: $0) }
            ?? ModelConfiguration(id: modelId)

        isDownloading = true
        downloadProgress = 0
        downloadedBytes = 0
        totalBytes = 0

        // Load through the VLM factory explicitly. The generic loader tries VLM then LLM
        // and keeps the first success; MLXLLM (linked via mlx-audio-swift) also accepts
        // "gemma4" as text-only, so a VLM failure would silently drop the vision tower.
        let container = try await VLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
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

            let input = try await container.prepare(input: userInput)
            // Diagnostic: confirm the image actually reached the prompt (Gemma 4 <|image|> = 258880).
            let imageTokens = (input.text.tokens .== 258_880).sum().item(Int.self)
            let firstToken = input.text.tokens.flattened()[0].item(Int.self)   // 2 = <bos>
            Self.log("prepare: promptTokens=\(input.text.tokens.size) firstToken=\(firstToken) imageTokens=\(imageTokens) pixels=\(input.image?.pixels.shape.description ?? "nil")")
            let stream = try await container.generate(input: input, parameters: generateParameters)

            var text = ""
            for await event in stream {
                switch event {
                case .chunk(let piece):
                    text += piece
                case .info(let info):
                    self.lastTokensPerSecond = info.tokensPerSecond
                    self.lastTokenCount = info.generationTokenCount
                default:
                    break   // tool calls / other events: unused here
                }
            }

            self.output = text
            // The description reflects whatever the camera saw — keep its text out of
            // persistent logs in release builds.
            #if DEBUG
            Self.log("describe OK \(lastTokenCount) tok @ \(String(format: "%.1f", lastTokensPerSecond)) tok/s: \(text.prefix(120))")
            Self.debugAppendDescription(text, language: language, modelId: currentModelId)
            #else
            Self.log("describe OK \(lastTokenCount) tok @ \(String(format: "%.1f", lastTokensPerSecond)) tok/s")
            #endif
            return text
        } catch {
            let msg = "VLM error: \(error.localizedDescription)"
            self.output = msg; NSLog("[VLM] %@", msg)
            return msg
        }
    }

    // MARK: - D1A decisions

    /// "local:<name>" is a side-loaded checkpoint in Documents/models/<name>.
    static func localModelDirectory(_ modelId: String) -> URL? {
        guard modelId.hasPrefix("local:") else { return nil }
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("models/\(modelId.dropFirst("local:".count))")
    }

    /// D1A's adapter folder; the question set sits next to it as describe_flow.json. An adapter is
    /// trained for one trunk, so a side-loaded checkpoint carries its own in d1a/.
    static func d1aDirectory(for modelId: String) -> URL {
        localModelDirectory(modelId)?.appendingPathComponent("d1a")
            ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("d1a/adapter")
    }

    var d1aDirectory: URL { Self.d1aDirectory(for: currentModelId) }

    var d1aAvailable: Bool {
        FileManager.default.fileExists(atPath: d1aDirectory.appendingPathComponent("adapters.safetensors").path)
    }

    private var d1a: D1ADecider?
    private var d1aQuestions: [D1ADecider.Question] = []

    /// p(yes) per describe-flow question ("redundant", "vague") for `new`, given the last spoken
    /// description. nil when D1A isn't installed or fails — the caller then simply speaks.
    func decide(previous: String?, new: String) async -> [String: Float]? {
        guard d1aAvailable, !running else { return nil }
        running = true
        defer { running = false }
        do {
            if d1a == nil {
                d1a = try D1ADecider(directory: d1aDirectory)
                let flow = try JSONSerialization.jsonObject(
                    with: Data(contentsOf: d1aDirectory.appendingPathComponent("describe_flow.json"))) as! [String: Any]
                d1aQuestions = D1ADecider.questions(from: flow["questions"] as! [String: Any], order: ["redundant", "vague"])
            }
            let decider = d1a!, questions = d1aQuestions
            let state = D1ADecider.render([("previous description", previous ?? "(none)"), ("new description", new)])
            let container = try await _load(modelId: currentModelId)
            let probs = try await container.perform { context -> [[Float]] in
                guard let gemma = context.model as? Gemma4 else { return [] }
                return try decider.decide(state: state, questions: questions, gemma: gemma, tokenizer: context.tokenizer)
            }
            guard probs.count == questions.count else { return nil }
            return Dictionary(uniqueKeysWithValues: zip(questions.map(\.id), probs.map { $0[1] }))
        } catch {
            Self.log("d1a decide FAILED: \(String(describing: error))")
            return nil
        }
    }

    public func unload() {
        loadState = .idle
        d1a = nil   // the next model may carry a different adapter
        isLoaded = false
        running = false
        output = ""
        statusMessage = ""
        downloadProgress = 0
        MLX.Memory.clearCache()
    }

    // MARK: - Diagnostics

    /// File-based log (device console is flaky). Pull via:
    /// xcrun devicectl device copy from … Documents/vlm.log
    nonisolated static func log(_ line: String) {
        NSLog("[VLM] %@", line)
        #if DEBUG
        // File log only in Debug: it never truncates and Documents is iCloud-backed.
        let mem = Int(MLX.Memory.activeMemory / (1024 * 1024))
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let url = docs.appendingPathComponent("vlm.log")
        let entry = "[\(ISO8601DateFormatter().string(from: Date()))] [\(mem)MB fp=\(footprintMB)MB avail=\(os_proc_available_memory() / (1024 * 1024))MB] \(line)\n"
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(Data(entry.utf8)); try? h.close()
        } else {
            try? entry.data(using: .utf8)?.write(to: url)
        }
        debugExcludeFromBackup(url)
        #endif
    }

    #if DEBUG
    /// Runs `action` on the loaded model (loading it first) — for DEBUG probes that need the model itself.
    func debugPerform<R: Sendable>(_ action: @Sendable (ModelContext) async throws -> sending R) async throws -> R {
        let container = try await _load(modelId: currentModelId)
        return try await container.perform(action)
    }

    /// Process physical footprint (what jetsam counts), in MB.
    nonisolated static var footprintMB: Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint / (1024 * 1024)) : -1
    }

    /// Peak physical footprint of the process so far (what jetsam counts), in MB.
    nonisolated static var peakFootprintMB: Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.ledger_phys_footprint_peak / (1024 * 1024)) : -1
    }

    /// Full description text, one JSON object per line, for building an offline eval set.
    nonisolated static func debugAppendDescription(_ text: String, language: String, modelId: String) {
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
              var line = try? JSONSerialization.data(withJSONObject: [
                  "ts": ISO8601DateFormatter().string(from: Date()),
                  "lang": language, "model": modelId, "text": text,
              ])
        else { return }
        line.append(0x0A)
        let url = docs.appendingPathComponent("descriptions.jsonl")
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(line); try? h.close()
        } else {
            try? line.write(to: url)
        }
        debugExcludeFromBackup(url)
    }

    /// Every D1A decision with its inputs, for calibrating the thresholds on real captures.
    nonisolated static func debugAppendDecision(previous: String?, new: String, redundant: Float, vague: Float, action: String) {
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
              var line = try? JSONSerialization.data(withJSONObject: [
                  "ts": ISO8601DateFormatter().string(from: Date()), "previous": previous ?? NSNull(), "new": new,
                  "p_redundant": redundant, "p_vague": vague, "action": action,
              ])
        else { return }
        line.append(0x0A)
        let url = docs.appendingPathComponent("d1a_decisions.jsonl")
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(line); try? h.close()
        } else {
            try? line.write(to: url)
        }
        debugExcludeFromBackup(url)
    }

    /// Debug artifacts hold what the camera saw; keep them out of iCloud / Finder backups.
    nonisolated static func debugExcludeFromBackup(_ url: URL) {
        var url = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
    }
    #endif
}
