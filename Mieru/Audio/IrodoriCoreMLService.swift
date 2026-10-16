//
//  IrodoriCoreMLService.swift
//  Mieru
//
//  Irodori-TTS v4.1-Small-MF on Core ML (no MLX): same VoiceDesign captions and surface as
//  IrodoriTTSService, but audio streams. Each decoded chunk is scheduled on the player as soon as
//  it exists, and speak() returns when the first chunk starts playing.
//  Model files and folder layout: see IrodoriCoreMLPipeline.swift.
//

import AVFoundation
import CoreML
import Foundation

@Observable
@MainActor
class IrodoriCoreMLService {

    nonisolated static let catalogId = "coreml:irodori-v4.1-mf"

    static var modelDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("irodori-coreml", isDirectory: true)
    }

    var isSpeaking = false
    var isEnabled = true
    var isLoaded = false
    var isLoading = false
    var statusMessage = ""

    /// Last synthesis perf — for the perf HUD.
    var lastSynthSeconds: Double = 0
    var lastAudioSeconds: Double = 0
    var lastRTF: Double = 0   // realtime factor = audio / synth (higher = faster)

    /// Same captions as the MLX engine.
    static var voices: [String] { IrodoriTTSService.voices }
    var availableVoices: [String] { Self.voices }

    /// Shares the MLX engine's persisted pick, so the voice picker and -irodoriVoice drive both.
    var selectedVoice: String? {
        get { UserDefaults.standard.string(forKey: "irodoriVoice") }
        set { UserDefaults.standard.set(newValue, forKey: "irodoriVoice") }
    }

    /// Compute units for the encoder, duration and decoder. The DiT is pinned to CPU+GPU (wrong on the ANE),
    /// and CPU-only is not offered (the encoder returns NaN there at L=256).
    enum Units: String {
        case gpu, ane
        var mlComputeUnits: MLComputeUnits { self == .ane ? .cpuAndNeuralEngine : .cpuAndGPU }
    }

    var units: Units {
        didSet {
            UserDefaults.standard.set(units.rawValue, forKey: "irodoriCoreMLUnits")
            if units != oldValue { unload() }   // the next load() uses the new units
        }
    }

    private var pipeline: IrodoriCoreMLPipeline?
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var synthTask: Task<Void, Never>?
    /// Resumed when the first chunk starts playing, or when synthesis ends without audio.
    private var firstAudio: CheckedContinuation<Void, Never>?
    private var pendingBuffers = 0
    private var synthDone = false
    private var emittedSamples = 0
    #if DEBUG
    private var dumpSamples: [Float] = []
    #endif

    /// Monotonic token — bumped on every speak()/stop(); stale work bails.
    private var playToken = 0

    private static let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: IrodoriCoreMLPipeline.sampleRate, channels: 1, interleaved: false)

    init() {
        // ANE by default: on the iPhone the decoder's enumerated-width build crashes on the GPU, and the
        // decoder on the ANE next to an encoder on the GPU never finished loading. The DiT is always GPU.
        var units = Units(rawValue: UserDefaults.standard.string(forKey: "irodoriCoreMLUnits") ?? "") ?? .ane
        #if DEBUG
        // -irodoriCoreMLUnits gpu|ane, to compare the two on the phone.
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-irodoriCoreMLUnits"), i + 1 < args.count,
           let u = Units(rawValue: args[i + 1]) {
            units = u
        }
        #endif
        self.units = units
    }

    // MARK: - Load

    func load() async {
        guard !isLoaded, !isLoading else { return }
        isLoading = true
        statusMessage = "Loading Irodori TTS (Core ML)…"
        IrodoriTTSService.log("coreml load start units=\(units.rawValue) fp=\(VLMService.footprintMB)MB")

        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            pipeline = try await IrodoriCoreMLPipeline.load(from: Self.modelDirectory, units: units.mlComputeUnits)
            isLoaded = true
            statusMessage = ""
            IrodoriTTSService.log(String(format: "coreml load OK in %.1fs (incl. pre-warm) units=%@ fp=%dMB",
                                         CFAbsoluteTimeGetCurrent() - t0, units.rawValue, VLMService.footprintMB))
            ModelCatalog.shared.markDownloaded(Self.catalogId)
        } catch {
            statusMessage = "Irodori Core ML error: \(error.localizedDescription)"
            IrodoriTTSService.log("coreml load ERROR: \(String(describing: error))")
        }

        isLoading = false
    }

    func unload() {
        stop()
        pipeline = nil
        isLoaded = false
    }

    // MARK: - Speak

    /// Synthesize and stream. **Returns the instant the first chunk starts playing** (or on
    /// failure) — so the caller can reveal the on-screen text in sync with the voice.
    func speak(_ text: String, language: String = "ja") async {
        guard isEnabled, !text.isEmpty, let pipeline else { return }

        stop()
        let myToken = playToken
        isSpeaking = true

        let cleanText = trimToCompleteSentences(text)
        guard !cleanText.isEmpty else { isSpeaking = false; return }
        // Irodori only reads Japanese; leave English answers on screen unspoken.
        guard containsJapanese(cleanText) else {
            IrodoriTTSService.log("coreml synth skipped: no Japanese script in text")
            isSpeaking = false
            return
        }

        let caption = selectedVoice ?? Self.voices.first ?? ""
        synthDone = false
        pendingBuffers = 0
        emittedSamples = 0
        #if DEBUG
        dumpSamples = []
        #endif
        IrodoriTTSService.log("coreml synth start len=\(cleanText.count) voice=\(caption) units=\(units.rawValue) fp=\(VLMService.footprintMB)MB")

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            firstAudio = continuation
            synthTask = Task { await self.synthesize(cleanText, caption: caption, pipeline: pipeline, token: myToken) }
        }
    }

    private func synthesize(_ text: String, caption: String, pipeline: IrodoriCoreMLPipeline, token: Int) async {
        let t0 = CFAbsoluteTimeGetCurrent()
        do {
            let t = try await pipeline.synthesize(text, caption: caption) { [weak self] samples in
                await self?.enqueue(samples, token: token) ?? false
            }
            guard token == playToken else { return }   // stopped mid-synth
            let synth = CFAbsoluteTimeGetCurrent() - t0
            lastSynthSeconds = synth
            lastAudioSeconds = Double(emittedSamples) / IrodoriCoreMLPipeline.sampleRate
            lastRTF = synth > 0 ? lastAudioSeconds / synth : 0
            IrodoriTTSService.log("coreml synth OK \(emittedSamples) samples in \(String(format: "%.1f", synth))s (RTF \(String(format: "%.2f", lastRTF))) — fp=\(VLMService.footprintMB)MB")
            // Same leading fields as IrodoriTTSService's "phases" line, so phone logs compare directly.
            IrodoriTTSService.log(String(format: "phases encode=%.2fs duration=%.2fs sampler=%.2fs decode=%.2fs frames=%d decode_first=%.2fs first_audio=%.2fs segments=%d engine=coreml units=%@",
                                         t.encode, t.duration, t.sampler, t.decode, t.frames, t.decodeFirst,
                                         t.firstAudio, t.segments, units.rawValue))
            #if DEBUG
            IrodoriTTSService.debugDumpWAV(samples: dumpSamples, rate: IrodoriCoreMLPipeline.sampleRate, tag: "irodori_coreml")
            dumpSamples = []
            #endif
        } catch {
            guard token == playToken else { return }
            IrodoriTTSService.log("coreml synth ERROR: \(String(describing: error))")
        }
        synthDone = true
        resumeFirstAudio()   // nothing was emitted (or it failed): let the caller go on
        if pendingBuffers == 0 { finishPlayback(token: token) }
    }

    private func resumeFirstAudio() {
        firstAudio?.resume()
        firstAudio = nil
    }

    // MARK: - Stop

    func stop() {
        playToken &+= 1   // any in-flight synth bails when it sees a changed token
        synthTask?.cancel()
        synthTask = nil
        playerNode?.stop()
        if let engine = audioEngine, engine.isRunning { engine.stop() }
        audioEngine = nil
        playerNode = nil
        pendingBuffers = 0
        isSpeaking = false
        resumeFirstAudio()
    }

    // MARK: - Playback

    /// Schedules one decoded chunk. Returns false once this utterance is stale, which stops the synthesis.
    private func enqueue(_ samples: [Float], token: Int) -> Bool {
        guard token == playToken, let format = Self.format else { return false }
        guard !samples.isEmpty else { return true }
        if playerNode == nil {
            guard startEngine(format: format) else { return false }
        }
        guard let player = playerNode,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else { return false }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count) }

        pendingBuffers += 1
        emittedSamples += samples.count
        #if DEBUG
        dumpSamples += samples
        #endif
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in self?.bufferPlayed(token: token) }
        }
        if !player.isPlaying {
            player.play()
            IrodoriTTSService.log("coreml playback started (first chunk \(String(format: "%.2f", Double(samples.count) / IrodoriCoreMLPipeline.sampleRate))s)")
        }
        resumeFirstAudio()
        return true
    }

    private func startEngine(format: AVAudioFormat) -> Bool {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            let engine = AVAudioEngine()
            let player = AVAudioPlayerNode()
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
            try engine.start()
            audioEngine = engine
            playerNode = player
            return true
        } catch {
            IrodoriTTSService.log("coreml playback ERROR: \(error.localizedDescription)")
            return false
        }
    }

    private func bufferPlayed(token: Int) {
        guard token == playToken else { return }
        pendingBuffers -= 1
        if synthDone && pendingBuffers == 0 { finishPlayback(token: token) }
    }

    private func finishPlayback(token: Int) {
        guard token == playToken else { return }
        playerNode?.stop()
        audioEngine?.stop()
        audioEngine = nil
        playerNode = nil
        isSpeaking = false
    }
}

// MARK: - Text Processing (same rules as IrodoriTTSService)

/// True if the text has any Hiragana, Katakana or CJK ideograph.
private func containsJapanese(_ text: String) -> Bool {
    text.unicodeScalars.contains {
        (0x3040...0x30FF).contains($0.value) || (0x4E00...0x9FFF).contains($0.value)
    }
}

private func trimToCompleteSentences(_ text: String) -> String {
    let endings: [Character] = ["。", "！", "？", ".", "!", "?"]
    if let lastIdx = text.lastIndex(where: { endings.contains($0) }) {
        return String(text[...lastIdx])
    }
    return text
}
