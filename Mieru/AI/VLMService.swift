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

    /// D1A's adapter folder, with the question set beside it as describe_flow.json. An adapter is trained
    /// for one trunk, so only a side-loaded checkpoint that carries its own in d1a/ has one.
    var d1aDirectory: URL? { Self.localModelDirectory(currentModelId)?.appendingPathComponent("d1a") }

    var d1aAvailable: Bool {
        guard let d1aDirectory else { return false }
        return FileManager.default.fileExists(atPath: d1aDirectory.appendingPathComponent("adapters.safetensors").path)
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
                d1a = try D1ADecider(directory: d1aDirectory!)
                let flow = try JSONSerialization.jsonObject(
                    with: Data(contentsOf: d1aDirectory!.appendingPathComponent("describe_flow.json"))) as! [String: Any]
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

/// Bicubic resize that reproduces PIL's `Image.resize(..., BICUBIC)` on 8-bit RGB, which the Hugging Face
/// Gemma 4 processor uses: Keys cubic (a = -0.5), support widened when shrinking (antialiasing), pixel
/// centres, horizontal then vertical pass, fixed-point weights and 8-bit rounding after each pass.
enum PILResize {
    private static let precisionBits = 32 - 8 - 2

    /// RGB bytes (`width * height * 3`, row-major) of `image` resized to `width` x `height`.
    static func bicubicRGB(_ image: CGImage, width: Int, height: Int) -> [UInt8] {
        let w = image.width, h = image.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        rgba.withUnsafeMutableBytes { buffer in
            let context = CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                    space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) { for c in 0..<3 { rgb[i * 3 + c] = rgba[i * 4 + c] } }
        // Rows first (w -> width), then columns (h -> height), as PIL does.
        let wide = width == w ? rgb
            : pass(rgb, lines: h, inSize: w, outSize: width, srcLine: w * 3, srcStep: 3, dstLine: width * 3, dstStep: 3)
        return height == h ? wide
            : pass(wide, lines: width, inSize: h, outSize: height, srcLine: 3, srcStep: width * 3, dstLine: 3, dstStep: width * 3)
    }

    /// Resamples `lines` independent lines of `inSize` RGB pixels to `outSize`; strides are in bytes.
    private static func pass(_ src: [UInt8], lines: Int, inSize: Int, outSize: Int,
                             srcLine: Int, srcStep: Int, dstLine: Int, dstStep: Int) -> [UInt8] {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        let taps: [(start: Int, weights: [Int64])] = (0..<outSize).map { x in
            let center = (Double(x) + 0.5) * scale
            let lo = max(Int(center - support + 0.5), 0), hi = min(Int(center + support + 0.5), inSize)
            var k = (lo..<hi).map { cubic((Double($0) - center + 0.5) / filterScale) }
            let total = k.reduce(0, +)
            if total != 0 { k = k.map { $0 / total } }
            let one = Double(1 << precisionBits)
            return (lo, k.map { Int64($0 < 0 ? -0.5 + $0 * one : 0.5 + $0 * one) })
        }
        var dst = [UInt8](repeating: 0, count: lines * outSize * 3)
        for line in 0..<lines {
            for x in 0..<outSize {
                let (start, weights) = taps[x]
                for c in 0..<3 {
                    var acc = Int64(1) << (precisionBits - 1)
                    for (i, wgt) in weights.enumerated() {
                        acc += Int64(src[line * srcLine + (start + i) * srcStep + c]) * wgt
                    }
                    dst[line * dstLine + x * dstStep + c] = UInt8(clamping: acc >> Int64(precisionBits))
                }
            }
        }
        return dst
    }

    private static func cubic(_ v: Double) -> Double {
        let a = -0.5, x = abs(v)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }
}
