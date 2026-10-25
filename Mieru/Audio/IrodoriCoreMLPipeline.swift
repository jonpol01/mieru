//
//  IrodoriCoreMLPipeline.swift
//  Mieru
//
//  Host loop for Irodori-TTS v4.1-Small-MF on Core ML (caption-only VoiceDesign, no reference
//  audio): tokenize → encoder → duration → 4 MeanFlow DiT steps → chunked DACVAE decode.
//  Mirrors coreml_tts() / flattening_point() in parity_v4_coreml.py and bucket() /
//  decoder_chunk_plan() in export_v4_coreml.py (jonpol01/Irodori-TTS, coreml-export-v4, d6aa1522).
//
//  Device layout (packages compiled on a Mac, pushed with devicectl):
//    Documents/irodori-coreml/
//      irodori4_encoder.mlmodelc     functions L32 L64 L128 L256 (only L128 is loaded)
//      irodori4_duration.mlmodelc
//      irodori4_dit.mlmodelc         functions s64 s128 s192 s256 s384 (only s384 is loaded)
//      irodori4_decoder_w128.mlmodelc   static width 128 (the enumerated build crashed on the iPhone GPU)
//      tokenizer/tokenizer.json
//      tokenizer/tokenizer_config.json   (ModernBERT-ja, from mlx-community/Irodori-TTS-v4.1-Small-MF-8bit)
//  A .mlpackage pushed in place of a .mlmodelc is compiled on the device once and kept as .mlmodelc.
//  TODO: download from Hugging Face instead of a devicectl push.
//

import CoreML
import Foundation
import Tokenizers

enum IrodoriCoreMLError: LocalizedError {
    case missing(String)
    case output(String)
    case badDuration(Float)

    var errorDescription: String? {
        switch self {
        case .missing(let f): return "missing \(f) in Documents/irodori-coreml"
        case .output(let n): return "Core ML output \(n) missing"
        case .badDuration(let v): return "duration model returned \(v)"
        }
    }
}

/// Wall-clock phase timings of one utterance (summed over its segments), in seconds.
struct IrodoriCoreMLTimings: Sendable {
    var encode = 0.0, duration = 0.0, sampler = 0.0
    var decodeFirst = 0.0   // first decoder chunk
    var decode = 0.0        // every decoder chunk
    var firstAudio = 0.0    // synth start → first chunk handed to playback
    var frames = 0
    var segments = 0
    var tokens = 0          // text tokens synthesized (BOS included per segment)
}

final class IrodoriCoreMLPipeline: @unchecked Sendable {

    // Shapes baked into the packages (export_v4_coreml.py).
    // One function per multifunction package: each function loaded as its own MLModel holds its own
    // copy of the weights (~0.6 GB per encoder or DiT function on iPhone 17 Pro), and loading all of
    // them got the app killed. Inputs are padded up to the single bucket; the models mask by length.
    static let textBuckets = [128]   // DiT text is fixed at 128 tokens, so the encoder's L256 is never used
    static let ditBuckets = [384]
    static let ditTextLen = 128
    static let captionLen = 64
    static let dim = 512
    static let latentDim = 32
    static let decoderWidths = [128]   // static-width decoder; a short last chunk is zero-padded
    static let decoderOverlap = 8
    static let hop = 1920
    static let sampleRate = 48000.0
    static let numSteps = 4
    static let bos: Int32 = 1
    /// Checkpoint max_text_len: only scales the first duration feature.
    static let durationMaxTextLen = 256

    static var maxTokens: Int { ditTextLen }
    static var maxFrames: Int { ditBuckets.last! }   // 15.4 s

    private let encoders: [Int: MLModel]
    private let duration: MLModel
    private let dits: [Int: MLModel]
    private let decoder: MLModel
    private let tokenizer: any Tokenizer

    private init(encoders: [Int: MLModel], duration: MLModel, dits: [Int: MLModel], decoder: MLModel,
                 tokenizer: any Tokenizer) {
        self.encoders = encoders
        self.duration = duration
        self.dits = dits
        self.decoder = decoder
        self.tokenizer = tokenizer
    }

    // MARK: - Load

    /// `units` drives the encoder, duration and decoder (.cpuAndGPU or .cpuAndNeuralEngine, never .cpuOnly:
    /// the encoder returns NaN at L=256 there). The DiT always runs on .cpuAndGPU: its ANE output is wrong.
    static func load(from dir: URL, units: MLComputeUnits) async throws -> IrodoriCoreMLPipeline {
        var folder = dir
        var noBackup = URLResourceValues()
        noBackup.isExcludedFromBackup = true
        try? folder.setResourceValues(noBackup)

        func model(_ name: String, _ function: String?, _ units: MLComputeUnits) async throws -> MLModel {
            let t = CFAbsoluteTimeGetCurrent()
            let config = MLModelConfiguration()
            config.computeUnits = units
            config.functionName = function
            let m = try await MLModel.load(contentsOf: try await compiledModel(name, in: dir), configuration: config)
            IrodoriTTSService.log(String(format: "coreml loaded %@%@ in %.2fs fp=%dMB", name,
                                         function.map { ":" + $0 } ?? "", CFAbsoluteTimeGetCurrent() - t,
                                         VLMService.footprintMB))
            return m
        }

        let tokenizer = try await loadTokenizer(dir.appendingPathComponent("tokenizer"))
        var encoders: [Int: MLModel] = [:]
        for L in textBuckets { encoders[L] = try await model("irodori4_encoder", "L\(L)", units) }
        let duration = try await model("irodori4_duration", nil, units)
        var dits: [Int: MLModel] = [:]
        for s in ditBuckets { dits[s] = try await model("irodori4_dit", "s\(s)", .cpuAndGPU) }
        let decoder = try await model("irodori4_decoder_w128", nil, units)

        let p = IrodoriCoreMLPipeline(encoders: encoders, duration: duration, dits: dits, decoder: decoder,
                                      tokenizer: tokenizer)
        try await p.prewarm()
        return p
    }

    private static func compiledModel(_ name: String, in dir: URL) async throws -> URL {
        let fm = FileManager.default
        let compiled = dir.appendingPathComponent("\(name).mlmodelc")
        if fm.fileExists(atPath: compiled.path) { return compiled }
        let package = dir.appendingPathComponent("\(name).mlpackage")
        guard fm.fileExists(atPath: package.path) else { throw IrodoriCoreMLError.missing("\(name).mlmodelc") }
        let t = CFAbsoluteTimeGetCurrent()
        let tmp = try await MLModel.compileModel(at: package)
        try fm.moveItem(at: tmp, to: compiled)
        IrodoriTTSService.log(String(format: "coreml compiled %@ on device in %.1fs", name, CFAbsoluteTimeGetCurrent() - t))
        return compiled
    }

    /// swift-transformers 1.3.5 cannot load ModernBERT-ja's files as shipped. Sanitize a copy, as
    /// irodoriSanitizeTokenizerFiles in jonpol01/mlx-audio-swift (irodori-v4) does: drop the unused
    /// post_processor/decoder (BOS is added by hand) and force the Unigram tokenizer class.
    private static func loadTokenizer(_ source: URL) async throws -> any Tokenizer {
        let fm = FileManager.default
        let dest = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("irodori-coreml-tokenizer", isDirectory: true)
        try? fm.removeItem(at: dest)
        try fm.createDirectory(at: dest, withIntermediateDirectories: true)
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            guard let data = try? Data(contentsOf: source.appendingPathComponent(name)),
                  var obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { throw IrodoriCoreMLError.missing("tokenizer/\(name)") }
            if name == "tokenizer.json" {
                obj["post_processor"] = NSNull()
                obj["decoder"] = NSNull()
            } else {
                obj["tokenizer_class"] = "XLMRobertaTokenizer"
                if let mml = obj["model_max_length"] as? NSNumber, mml.doubleValue > 1_000_000 {
                    obj["model_max_length"] = 4096
                }
            }
            try JSONSerialization.data(withJSONObject: obj).write(to: dest.appendingPathComponent(name))
        }
        return try await AutoTokenizer.from(modelFolder: dest)
    }

    /// One prediction per function and enumerated shape, so GPU/ANE plans are built at load, not on first speech.
    private func prewarm() async throws {
        let t = CFAbsoluteTimeGetCurrent()
        let one = try Self.i32([1], [1])
        let caption = try Self.f16([Float](repeating: 0, count: Self.captionLen * Self.dim), [1, Self.captionLen, Self.dim])
        for L in Self.textBuckets {
            let ids = try Self.i32([Self.bos] + [Int32](repeating: 0, count: L - 1), [1, L])
            IrodoriTTSService.log("coreml pre-warm encoder L\(L) fp=\(VLMService.footprintMB)MB")
            _ = try await predict(encoders[L]!, ["input_ids": ids, "length": one], "text_state")
            let ts = try Self.f16([Float](repeating: 0, count: L * Self.dim), [1, L, Self.dim])
            IrodoriTTSService.log("coreml pre-warm duration fp=\(VLMService.footprintMB)MB")
            _ = try await predict(duration, ["text_state": ts, "text_len": one, "caption_state": caption,
                                             "caption_len": one, "duration_features": try Self.f16([Float](repeating: 0, count: 14), [1, 14])],
                                  "log_frames")
        }
        let text = try Self.f16([Float](repeating: 0, count: Self.ditTextLen * Self.dim), [1, Self.ditTextLen, Self.dim])
        for s in Self.ditBuckets {
            IrodoriTTSService.log("coreml pre-warm dit s\(s) fp=\(VLMService.footprintMB)MB")
            _ = try await predict(dits[s]!, [
                "x_t": try Self.f16([Float](repeating: 0, count: s * Self.latentDim), [1, s, Self.latentDim]),
                "t": try Self.f16([1], [1]), "delta_t": try Self.f16([0.25], [1]), "latent_len": try Self.i32([Int32(s)], [1]),
                "text_state": text, "text_len": one, "caption_state": caption, "caption_len": one,
            ], "v")
        }
        // No decoder pre-warm: its first GPU run at 48 kHz needs more working memory than is free while
        // Gemma is still resident (load happens before Gemma is unloaded for synthesis), and iOS killed
        // the app there. It warms on the first real chunk instead.
        IrodoriTTSService.log(String(format: "coreml pre-warm %.2fs fp=%dMB", CFAbsoluteTimeGetCurrent() - t,
                                     VLMService.footprintMB))
    }

    // MARK: - Synthesize

    /// Synthesizes `text` (already trimmed to whole sentences) with a VoiceDesign `caption`. Calls `emit` with each
    /// decoded audio chunk (48 kHz mono) as soon as it exists; `emit` returning false stops the work.
    func synthesize(_ text: String, caption: String,
                    emit: @Sendable ([Float]) async -> Bool) async throws -> IrodoriCoreMLTimings {
        var T = IrodoriCoreMLTimings()
        let start = CFAbsoluteTimeGetCurrent()
        var clock = start
        func lap() -> Double {
            let now = CFAbsoluteTimeGetCurrent()
            defer { clock = now }
            return now - clock
        }

        // Caption: stripped, not normalized; BOS + tokens, at most 64, through the encoder's L64 caption head.
        var captionIds = [Self.bos] + tokenizer.encode(text: caption.trimmingCharacters(in: .whitespacesAndNewlines),
                                                       addSpecialTokens: false).map { Int32($0) }
        captionIds = Array(captionIds.prefix(Self.captionLen))
        let nc = captionIds.count
        // Encoded at the loaded text bucket; positions past `length` are masked, so the first 64 rows are exact.
        let LC = Self.textBuckets.last!
        let captionIn = try Self.i32(captionIds + [Int32](repeating: 0, count: LC - nc), [1, LC])
        let captionOut = try await predict(encoders[LC]!, ["input_ids": captionIn, "length": try Self.i32([Int32(nc)], [1])],
                                           "caption_state")
        let captionState = try Self.f16(Array(Self.floats(captionOut).prefix(Self.captionLen * Self.dim)),
                                        [1, Self.captionLen, Self.dim])
        let captionLen = try Self.i32([Int32(nc)], [1])
        T.encode += lap()

        var queue = plan(text)
        var firstChunk = true
        while !queue.isEmpty {
            try Task.checkCancellation()
            let seg = queue.removeFirst()
            let n = seg.ids.count
            let textLen = try Self.i32([Int32(n)], [1])
            clock = CFAbsoluteTimeGetCurrent()

            // Encoder: smallest bucket ≥ n (BOS included), zero-padded.
            let L = Self.bucket(Self.textBuckets, n)
            let ids = try Self.i32(seg.ids + [Int32](repeating: 0, count: L - n), [1, L])
            let ts = Self.floats(try await predict(encoders[L]!, ["input_ids": ids, "length": textLen], "text_state"))
            T.encode += lap()

            // Duration → frames = round(expm1(log_frames)), clamped like frames_from_log, then to the DiT max.
            let feats = Self.durationFeatures(seg.text, tokenCount: n)
            let logf = Self.floats(try await predict(duration, [
                "text_state": try Self.f16(ts, [1, L, Self.dim]), "text_len": textLen,
                "caption_state": captionState, "caption_len": captionLen,
                "duration_features": try Self.f16(feats, [1, 14]),
            ], "log_frames"))[0]
            guard logf.isFinite else { throw IrodoriCoreMLError.badDuration(logf) }
            var frames = Self.framesFromLog(Double(logf))
            T.duration += lap()
            if frames > Self.maxFrames {
                if seg.sentences.count > 1 {
                    let half = seg.sentences.count / 2
                    let parts = [Array(seg.sentences[..<half]), Array(seg.sentences[half...])].compactMap(makeSegment)
                    IrodoriTTSService.log("coreml segment \(frames) frames > \(Self.maxFrames): split into \(parts.count)")
                    queue.insert(contentsOf: parts, at: 0)
                    continue
                }
                IrodoriTTSService.log("coreml single sentence \(frames) frames > \(Self.maxFrames): clamped")
                frames = Self.maxFrames
            }
            T.frames += frames
            T.segments += 1
            T.tokens += n

            // MeanFlow sampler: text state sliced to n tokens, zero-padded to 128; noise for `frames`, zeros after.
            let S = Self.bucket(Self.ditBuckets, frames)
            var textD = [Float](repeating: 0, count: Self.ditTextLen * Self.dim)
            textD.replaceSubrange(0..<(min(n, Self.ditTextLen) * Self.dim), with: ts[0..<(min(n, Self.ditTextLen) * Self.dim)])
            let textState = try Self.f16(textD, [1, Self.ditTextLen, Self.dim])
            let latentLen = try Self.i32([Int32(frames)], [1])
            var x = Self.gaussian(frames * Self.latentDim, seed: IrodoriTTSService.speakerSeed(for: caption)) + [Float](repeating: 0, count: (S - frames) * Self.latentDim)
            for i in 0..<Self.numSteps {
                try Task.checkCancellation()
                let t = 1 - Float(i) / Float(Self.numSteps)
                let tNext = 1 - Float(i + 1) / Float(Self.numSteps)
                let v = Self.floats(try await predict(dits[S]!, [
                    "x_t": try Self.f16(x, [1, S, Self.latentDim]), "t": try Self.f16([t], [1]),
                    "delta_t": try Self.f16([t - tNext], [1]), "latent_len": latentLen,
                    "text_state": textState, "text_len": textLen,
                    "caption_state": captionState, "caption_len": captionLen,
                ], "v"))
                for j in 0..<x.count { x[j] += v[j] * (tNext - t) }
            }
            T.sampler += lap()

            // Tail trim (runtime trim_tail defaults), decided on the latent so chunks past it are never decoded.
            let flat = Self.flatteningPoint(x, frames: frames)
            let keepFrames = flat > 0 ? min(frames, flat) : frames

            // Decoder: 128-frame windows with 8 frames of context, each chunk played as soon as it is decoded.
            for (ws, w, ks, ke) in Self.decoderChunkPlan(frames) where ks < keepFrames {
                try Task.checkCancellation()
                var z = [Float](repeating: 0, count: Self.latentDim * w)   // (1, 32, w), zeros past `frames`
                for f in ws..<min(ws + w, frames) {
                    for c in 0..<Self.latentDim { z[c * w + (f - ws)] = x[f * Self.latentDim + c] }
                }
                let tc = CFAbsoluteTimeGetCurrent()
                IrodoriTTSService.log("coreml decode W\(w) fp=\(VLMService.footprintMB)MB")
                let audio = Self.floats(try await predict(decoder, ["z": try Self.f16(z, [1, Self.latentDim, w])], "audio"))
                let dt = CFAbsoluteTimeGetCurrent() - tc
                T.decode += dt
                let part = Array(audio[((ks - ws) * Self.hop)..<((min(ke, keepFrames) - ws) * Self.hop)])
                if firstChunk {
                    firstChunk = false
                    T.decodeFirst = dt
                    T.firstAudio = CFAbsoluteTimeGetCurrent() - start
                }
                guard await emit(part) else { return T }
            }
        }
        return T
    }

    private func predict(_ model: MLModel, _ inputs: [String: MLMultiArray], _ output: String) async throws -> MLMultiArray {
        let features = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
        guard let out = try await model.prediction(from: features).featureValue(for: output)?.multiArrayValue
        else { throw IrodoriCoreMLError.output(output) }
        return out
    }

    // MARK: - Text → segments

    /// A run of whole sentences that fits the DiT: ≤ 128 tokens (BOS included).
    struct Segment {
        let sentences: [String]   // raw text, for re-splitting when the predicted duration is too long
        let text: String          // v4-normalized, stripped
        let ids: [Int32]          // BOS + tokens
    }

    private func makeSegment(_ sentences: [String]) -> Segment? {
        let text = irodoriV4Normalize(sentences.joined()).pythonStripped()
        guard !text.isEmpty else { return nil }
        let ids = [Self.bos] + tokenizer.encode(text: text, addSpecialTokens: false).map { Int32($0) }
        return Segment(sentences: sentences, text: text, ids: ids)
    }

    /// Greedy packing of sentences (。！？) into segments of ≤ 128 tokens. A sentence that alone is too long is
    /// split at 、, and a clause that is still too long is truncated to 128 tokens.
    func plan(_ text: String) -> [Segment] {
        func pack(_ pieces: [String], fallback: Set<Character>?) -> [Segment] {
            var out: [Segment] = []
            var current: [String] = []
            func flush() {
                defer { current = [] }
                guard let seg = makeSegment(current) else { return }
                if seg.ids.count <= Self.maxTokens {
                    out.append(seg)
                } else if let fallback, current.count == 1 {
                    out += pack(Self.split(current[0], at: fallback), fallback: nil)
                } else {
                    IrodoriTTSService.log("coreml segment \(seg.ids.count) tokens > \(Self.maxTokens): truncated")
                    out.append(Segment(sentences: current, text: seg.text, ids: Array(seg.ids.prefix(Self.maxTokens))))
                }
            }
            for piece in pieces {
                if !current.isEmpty, (makeSegment(current + [piece])?.ids.count ?? 0) > Self.maxTokens { flush() }
                current.append(piece)
            }
            if !current.isEmpty { flush() }
            return out
        }
        return pack(Self.split(text, at: ["。", "！", "？", "!", "?"]), fallback: ["、"])
    }

    /// Pieces that each end with one of `terminators` (the last may not).
    private static func split(_ text: String, at terminators: Set<Character>) -> [String] {
        var out: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if terminators.contains(ch) { out.append(current); current = "" }
        }
        if !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { out.append(current) }
        return out
    }

    // MARK: - Host math (ports of the Python reference)

    static func bucket(_ buckets: [Int], _ n: Int) -> Int {
        buckets.first { n <= $0 } ?? buckets.last!
    }

    /// frames_from_log: round(expm1(log_frames)) (Python rounds half to even), clamped to [0.5 s, 30 s] of frames.
    static func framesFromLog(_ logFrames: Double) -> Int {
        let lo = Double(Int((0.5 * sampleRate / Double(hop)).rounded(.up)))
        let hi = Double(Int((30 * sampleRate / Double(hop)).rounded(.down)))
        return Int(max(lo, min(hi, expm1(logFrames).rounded(.toNearestOrEven))))
    }

    /// decoder_chunk_plan: [(window start, width, keep start, keep end)] in latent frames.
    static func decoderChunkPlan(_ S: Int) -> [(Int, Int, Int, Int)] {
        var plan: [(Int, Int, Int, Int)] = []
        var a = 0
        while a < S {
            var ws = a - min(decoderOverlap, a)
            let w = decoderWidths.first { ws + $0 >= S } ?? decoderWidths.last!
            if ws + w >= S {   // final chunk: end the window at S when possible (no right padding)
                ws = max(0, S - w)
                plan.append((ws, w, a, S))
                break
            }
            plan.append((ws, w, a, ws + w - decoderOverlap))
            a = ws + w - decoderOverlap
        }
        return plan
    }

    /// flattening_point: first frame whose next 20 frames (zero-padded past the end) have std < 0.05 and
    /// |mean| < 0.1 over all values. `x` is (≥ frames, 32) row-major.
    static func flatteningPoint(_ x: [Float], frames: Int, window: Int = 20) -> Int {
        let count = Double(window * latentDim)
        for i in 0..<frames {
            var sum = 0.0, sumSq = 0.0
            for f in i..<min(i + window, frames) {
                for c in 0..<latentDim {
                    let v = Double(x[f * latentDim + c])
                    sum += v
                    sumSq += v * v
                }
            }
            let mean = sum / count
            let std = (max(0, sumSq / count - mean * mean)).squareRoot()
            if std < 0.05 && abs(mean) < 0.1 { return i }
        }
        return frames
    }

    /// Standard normal noise (Box–Muller). Restarting from the same seed for every segment gives
    /// each frame the same noise, so a voice keeps its speaker across segments and utterances.
    static func gaussian(_ count: Int, seed: UInt64) -> [Float] {
        var rng = SplitMix64(state: seed)
        var out: [Float] = []
        out.reserveCapacity(count + 1)
        while out.count < count {
            let r = (-2 * log(Float.random(in: Float.ulpOfOne..<1, using: &rng))).squareRoot()
            let theta = 2 * Float.pi * Float.random(in: 0..<1, using: &rng)
            out.append(r * cos(theta))
            out.append(r * sin(theta))
        }
        return Array(out.prefix(count))
    }

    private struct SplitMix64: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9e37_79b9_7f4a_7c15
            var z = state
            z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
            z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
            return z ^ (z >> 31)
        }
    }

    // MARK: - MLMultiArray helpers (stride-aware: Core ML outputs may be padded)

    private static func forEachOffset(_ shape: [Int], _ strides: [Int], _ body: (Int, Int) -> Void) {
        let count = shape.reduce(1, *)
        var contiguous = [Int](repeating: 1, count: shape.count)
        for d in stride(from: shape.count - 2, through: 0, by: -1) { contiguous[d] = contiguous[d + 1] * shape[d + 1] }
        if strides == contiguous {
            for i in 0..<count { body(i, i) }
            return
        }
        var index = [Int](repeating: 0, count: shape.count)
        var offset = 0
        for i in 0..<count {
            body(i, offset)
            var d = shape.count - 1
            while d >= 0 {
                index[d] += 1
                offset += strides[d]
                if index[d] < shape[d] { break }
                offset -= strides[d] * shape[d]
                index[d] = 0
                d -= 1
            }
        }
    }

    static func f16(_ values: [Float], _ shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        a.withUnsafeMutableBufferPointer(ofType: Float16.self) { p, strides in
            forEachOffset(shape, strides) { i, o in p[o] = Float16(values[i]) }
        }
        return a
    }

    static func i32(_ values: [Int32], _ shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        a.withUnsafeMutableBufferPointer(ofType: Int32.self) { p, strides in
            forEachOffset(shape, strides) { i, o in p[o] = values[i] }
        }
        return a
    }

    /// Logical (row-major) Float values of a float16 / float32 output.
    static func floats(_ a: MLMultiArray) -> [Float] {
        let shape = a.shape.map(\.intValue), strides = a.strides.map(\.intValue)
        var out = [Float](repeating: 0, count: a.count)
        switch a.dataType {
        case .float16:
            a.withUnsafeBufferPointer(ofType: Float16.self) { p in forEachOffset(shape, strides) { i, o in out[i] = Float(p[o]) } }
        case .float32:
            a.withUnsafeBufferPointer(ofType: Float.self) { p in forEachOffset(shape, strides) { i, o in out[i] = p[o] } }
        default:
            for i in 0..<a.count { out[i] = a[i].floatValue }
        }
        return out
    }

    // MARK: - Duration features (irodori_tts/duration.py build_duration_features, no speaker)

    private static let annotationEmojis: [[Unicode.Scalar]] = {
        let list: [String] = [
        "⏩", "⏱️", "⏸️", "🌬️", "🍭", "🎛️", "🎭", "🎵", "🐢", "🐱", "👂", "👃", "👅",
        "👌", "👏", "💋", "💥", "💦", "💪", "📄", "📞", "📢", "📣", "😆", "😊", "😌",
        "😎", "😏", "😒", "😖", "😟", "😠", "😪", "😭", "😮", "😮\u{200D}💨", "😰",
        "😱", "😲", "😴", "🙄", "🙏", "🤐", "🤔", "🤢", "🤧", "🤭", "🥤", "🥱", "🥴",
        "🥵", "🥹", "🥺", "🫣", "🫶", "📖",
        ]
        return list.map { Array($0.unicodeScalars) }.sorted { $0.count > $1.count }   // longest first, like the regex
    }()

    /// 14 features of the normalized text; Python's len() and str.count() work on code points, so scalars here.
    static func durationFeatures(_ text: String, tokenCount: Int) -> [Float] {
        let s = Array(text.unicodeScalars)
        let chars = Float(max(s.count, 1))
        func occurrences(_ targets: Unicode.Scalar...) -> Int { s.reduce(0) { $0 + (targets.contains($1) ? 1 : 0) } }
        func capped(_ n: Int, _ cap: Int) -> Float { log1p(Float(min(max(n, 0), cap))) / log1p(Float(cap)) }
        var emojis = 0, i = 0
        while i < s.count {
            if let e = annotationEmojis.first(where: { i + $0.count <= s.count && Array(s[i..<(i + $0.count)]) == $0 }) {
                emojis += 1
                i += e.count
            } else {
                i += 1
            }
        }
        let kana = s.filter { (0x3040...0x309F).contains($0.value) || (0x30A0...0x30FF).contains($0.value) }.count
        let kanji = s.filter {
            (0x3400...0x4DBF).contains($0.value) || (0x4E00...0x9FFF).contains($0.value)
                || (0xF900...0xFAFF).contains($0.value) || (0x20000...0x2FA1F).contains($0.value)
        }.count
        let alnum = s.filter { (0x30...0x39).contains($0.value) || (0x41...0x5A).contains($0.value) || (0x61...0x7A).contains($0.value) }.count
        return [
            Float(min(max(tokenCount, 0), durationMaxTextLen)) / Float(durationMaxTextLen),
            log1p(min(chars, 512)) / log1p(Float(512)),
            Float(tokenCount) / chars,
            capped(occurrences("。", "."), 8),
            capped(occurrences("、", ","), 16),
            capped(occurrences("ー"), 8),
            capped(occurrences("…"), 8),
            capped(occurrences("！", "!"), 8),
            capped(occurrences("？", "?"), 8),
            capped(emojis, 8),
            Float(kana) / chars,
            Float(kanji) / chars,
            Float(alnum) / chars,
            0,   // has_speaker: caption-only
        ]
    }
}

// MARK: - v4 text normalization
// Copied from irodoriNormalizeTextUpstream / irodoriStripOuterBrackets / irodoriPythonStripped in
// jonpol01/mlx-audio-swift (irodori-v4, Sources/MLXAudioTTS/Models/IrodoriTTS/IrodoriTTSText.swift), which are
// internal to MLXAudioTTS. That port mirrors Irodori-TTS/irodori_tts/text_normalization.py normalize_text.

private let v4SimpleReplacements: [(String, String)] = [
    ("\t", ""), ("[n]", ""), ("\\[n\\]", ""), ("\u{3000}", ""),
    ("？", "?"), ("！", "!"), ("♥", "♡"), ("●", "○"), ("◯", "○"), ("〇", "○"),
]

private let v4RegexReplacements: [(NSRegularExpression, String)] = [
    ("[;▼♀♂《》≪≫①②③④⑤⑥]", ""),
    ("[\u{02d7}\u{2010}-\u{2015}\u{2043}\u{2212}\u{23af}\u{23e4}\u{2500}\u{2501}\u{2e3a}\u{2e3b}]", ""),
    ("[\u{ff5e}\u{301C}]", "ー"),
    ("…{3,}", "……"),
].map { (try! NSRegularExpression(pattern: $0.0), $0.1) }

private let v4BracketPairs: [Unicode.Scalar: Unicode.Scalar] = [
    "「": "」", "『": "』", "（": "）", "【": "】", "(": ")",
]

/// Removes bracket pairs that enclose the whole string, repeatedly (depth-aware: 「前半」と「後半」 is kept).
private func v4StripOuterBrackets(_ input: String) -> String {
    var text = Array(input.unicodeScalars)
    while text.count >= 2, let close = v4BracketPairs[text[0]], close == text[text.count - 1] {
        let open = text[0]
        var depth = 0
        var enclosesAll = true
        for (i, c) in text.enumerated() {
            if c == open { depth += 1 } else if c == close { depth -= 1 }
            if depth == 0 && i < text.count - 1 {
                enclosesAll = false
                break
            }
        }
        guard enclosesAll && depth == 0 else { break }
        text = Array(text[1 ..< text.count - 1])
    }
    var view = String.UnicodeScalarView()
    view.append(contentsOf: text)
    return String(view)
}

/// normalize_text (v4). Callers strip the result, as upstream does.
func irodoriV4Normalize(_ input: String) -> String {
    var text = input
    for (old, new) in v4SimpleReplacements {
        text = text.replacingOccurrences(of: old, with: new, options: .literal)
    }
    for (regex, replacement) in v4RegexReplacements {
        text = regex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: replacement)
    }
    text = v4StripOuterBrackets(text)
    // NFKC as NFKD + NFC: Foundation's one-step NFKC leaves halfwidth voiced kana (ｷﾞ) decomposed, Python composes them.
    text = text.decomposedStringWithCompatibilityMapping.precomposedStringWithCanonicalMapping
    text = text.replacingOccurrences(of: "...", with: "…", options: .literal)
    text = text.replacingOccurrences(of: "..", with: "…", options: .literal)
    return text
}

/// Python's str.isspace() set (Foundation's whitespace set lacks U+001C–U+001F).
private let pythonWhitespace: Set<Unicode.Scalar> = {
    var set = Set<Unicode.Scalar>()
    for v: UInt32 in [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0, 0x1680,
                      0x2028, 0x2029, 0x202F, 0x205F, 0x3000] {
        set.insert(Unicode.Scalar(v)!)
    }
    for v: UInt32 in 0x2000 ... 0x200A { set.insert(Unicode.Scalar(v)!) }
    return set
}()

private extension String {
    /// Python str.strip() with no arguments.
    func pythonStripped() -> String {
        let scalars = Array(unicodeScalars)
        guard let first = scalars.firstIndex(where: { !pythonWhitespace.contains($0) }),
              let last = scalars.lastIndex(where: { !pythonWhitespace.contains($0) })
        else { return "" }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars[first ... last])
        return String(view)
    }
}
