//
//  IrodoriTTSService.swift
//  Mieru
//
//  On-device Japanese neural TTS using Irodori-TTS (600M, flow-matching DiT)
//  via mlx-audio-swift. VoiceDesign: each "voice" is a Japanese caption that
//  describes the speaker; the model renders a voice from it.
//
//  Mirrors SpeechService (Kokoro) so ContentView can switch engines.
//

import AVFoundation
import Foundation
import MLX
import MLXAudioTTS

@Observable
@MainActor
class IrodoriTTSService {

    var isSpeaking = false
    var isEnabled = true
    var isLoaded = false
    var isLoading = false
    var statusMessage = ""

    /// Last synthesis perf — for the perf HUD.
    var lastSynthSeconds: Double = 0
    var lastAudioSeconds: Double = 0
    var lastRTF: Double = 0   // realtime factor = audio / synth (higher = faster)

    /// VoiceDesign captions surfaced in the voice picker. The selected caption is
    /// passed straight through as the `voice` parameter to the model.
    static let voices: [String] = [
        "落ち着いた自然な女性の声",
        "明るく親しみやすい女性の声",
        "優しく穏やかな女性の声",
        "落ち着いた自然な男性の声",
        "ハキハキした若い男性の声",
        "低く落ち着いた渋い男性の声",
    ]

    var availableVoices: [String] { Self.voices }

    /// Selected VoiceDesign caption (persisted).
    var selectedVoice: String? {
        didSet { UserDefaults.standard.set(selectedVoice, forKey: "irodoriVoice") }
    }

    let modelId = "mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit"

    private var model: SpeechGenerationModel?
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?

    /// Discovered from the loaded model (Irodori is 48 kHz).
    private var sampleRate: Double = 48000

    /// Monotonic token — bumped on every speak()/stop(); stale work bails.
    private var playToken = 0

    init() {
        selectedVoice = UserDefaults.standard.string(forKey: "irodoriVoice")
    }

    // MARK: - Load

    func load() async {
        guard !isLoaded, !isLoading else { return }
        isLoading = true
        statusMessage = "Loading Irodori TTS…"

        do {
            // Keep MLX's buffer cache small — Irodori (~1.3 GB) may load alongside the
            // resident Gemma 4 brain (~1.5 GB) + camera; a low cache limit returns
            // transient load buffers to the OS instead of caching them.
            MLX.Memory.cacheLimit = 32 * 1024 * 1024
            MLX.Memory.clearCache()

            Self.log("load start — active=\(Self.activeMB)MB")
            let m = try await TTS.loadModel(modelRepo: modelId)
            model = m
            sampleRate = Double(m.sampleRate)
            isLoaded = true
            statusMessage = ""
            Self.log("load OK sr=\(Int(sampleRate)) — active=\(Self.activeMB)MB peak=\(Self.peakMB)MB")
            NSLog("[Irodori] Loaded %@ (%.0f Hz)", modelId, sampleRate)
            ModelCatalog.shared.markDownloaded(modelId)
        } catch {
            statusMessage = "Irodori error: \(error.localizedDescription)"
            Self.log("load ERROR: \(String(describing: error))")
            NSLog("[Irodori] Load error: %@", String(describing: error))
        }

        isLoading = false
    }

    func unload() {
        stop()
        model = nil
        isLoaded = false
        MLX.Memory.clearCache()
    }

    // MARK: - Speak

    /// Synthesize, then start playback. **Returns the instant audio begins** (or on
    /// failure) — so the caller can reveal the on-screen text in sync with the voice.
    func speak(_ text: String, language: String = "ja") async {
        guard isEnabled, !text.isEmpty, let model else { return }

        stop()
        let myToken = playToken
        isSpeaking = true

        let cleanText = trimToCompleteSentences(text)
        guard !cleanText.isEmpty else { isSpeaking = false; return }
        // Gemma sometimes answers in English despite a Japanese prompt; Irodori can only
        // read Japanese, so leave such text on screen unspoken instead of voicing gibberish.
        guard containsJapanese(cleanText) else {
            Self.log("synth skipped: no Japanese script in text")
            isSpeaking = false
            return
        }

        // Irodori is a Japanese-only model: no language routing, and the "voice" is a
        // VoiceDesign caption. Default to the first preset so it matches the picker top.
        let voice = selectedVoice ?? Self.voices.first
        let capturedModel = model
        MLX.Memory.clearCache()
        Self.log("synth start len=\(cleanText.count) voice=\(voice ?? "nil") — active=\(Self.activeMB)MB")

        do {
            // Whole-utterance generate (flow-matching) on a background task.
            let t0 = CFAbsoluteTimeGetCurrent()
            let samples: [Float] = try await Task.detached {
                let audio = try await capturedModel.generate(
                    text: cleanText, voice: voice, refAudio: nil, refText: nil, language: nil)
                return audio.asArray(Float.self)
            }.value
            let synth = CFAbsoluteTimeGetCurrent() - t0
            lastSynthSeconds = synth
            lastAudioSeconds = Double(samples.count) / sampleRate
            lastRTF = synth > 0 ? lastAudioSeconds / synth : 0
            Self.log("synth OK \(samples.count) samples in \(String(format: "%.1f", synth))s (RTF \(String(format: "%.2f", lastRTF))) — active=\(Self.activeMB)MB peak=\(Self.peakMB)MB")
            #if DEBUG
            Self.debugDumpWAV(samples: samples, rate: sampleRate, tag: "irodori")
            #endif

            guard myToken == playToken else { return }   // cancelled mid-synth
            playAudio(samples: samples, token: myToken)   // starts playback, returns immediately
        } catch {
            Self.log("synth ERROR: \(String(describing: error))")
            NSLog("[Irodori] Synthesis error: %@", String(describing: error))
            if myToken == playToken { isSpeaking = false }
        }
    }

    private func finishIfCurrent(_ token: Int) {
        guard token == playToken else { return }
        isSpeaking = false
    }

    // MARK: - Stop

    func stop() {
        playToken &+= 1   // any in-flight synth bails when it sees a changed token
        playerNode?.stop()
        if let engine = audioEngine, engine.isRunning { engine.stop() }
        audioEngine = nil
        playerNode = nil
        isSpeaking = false
    }

    // MARK: - Playback

    private func playAudio(samples: [Float], token: Int) {
        guard token == playToken else { return }
        guard !samples.isEmpty else { finishIfCurrent(token); return }

        playerNode?.stop()
        audioEngine?.stop()
        audioEngine = nil
        playerNode = nil

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
            channels: 1, interleaved: false
        ) else { finishIfCurrent(token); return }

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)
        ) else { finishIfCurrent(token); return }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        let channelData = buffer.floatChannelData![0]
        for i in 0..<samples.count { channelData[i] = samples[i] }

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)

            let engine = AVAudioEngine()
            let player = AVAudioPlayerNode()
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
            try engine.start()

            self.audioEngine = engine
            self.playerNode = player

            player.scheduleBuffer(buffer) { [weak self] in
                Task { @MainActor in self?.finishPlayback(token: token) }
            }
            player.play()
            Self.log("playback started \(String(format: "%.1f", Double(samples.count) / sampleRate))s audio")
        } catch {
            NSLog("[Irodori] Playback error: %@", error.localizedDescription)
            finishIfCurrent(token)
        }
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

// MARK: - Diagnostics

extension IrodoriTTSService {
    /// Active MLX memory in MB.
    nonisolated static var activeMB: Int { Int(MLX.Memory.activeMemory / (1024 * 1024)) }
    /// Peak MLX memory in MB.
    nonisolated static var peakMB: Int { Int(MLX.Memory.peakMemory / (1024 * 1024)) }

    /// File-based log (device console is flaky). Pull via:
    /// xcrun devicectl device copy from … Documents/irodori.log
    nonisolated static func log(_ line: String) {
        NSLog("[Irodori] %@", line)
        #if DEBUG
        // File log only in Debug: it never truncates and Documents is iCloud-backed.
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let url = docs.appendingPathComponent("irodori.log")
        let entry = "[\(ISO8601DateFormatter().string(from: Date()))] \(line)\n"
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(Data(entry.utf8)); try? h.close()
        } else {
            try? entry.data(using: .utf8)?.write(to: url)
        }
        VLMService.debugExcludeFromBackup(url)
        #endif
    }
}

#if DEBUG
extension IrodoriTTSService {
    /// Writes synthesized audio to Documents/tts_dumps/<tag>_<time>.wav (16-bit PCM mono)
    /// so a remote test can pull it off the device and check it isn't silent or truncated.
    nonisolated static func debugDumpWAV(samples: [Float], rate: Double, tag: String) {
        guard !samples.isEmpty,
              let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return }
        let dir = docs.appendingPathComponent("tts_dumps", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        VLMService.debugExcludeFromBackup(dir)

        let pcm = samples.map { Int16(max(-1, min(1, $0)) * Float(Int16.max)) }
        let sr = UInt32(rate)
        var data = Data()
        func le<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: "RIFF".utf8); le(UInt32(36 + pcm.count * 2)); data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8); le(UInt32(16)); le(UInt16(1)); le(UInt16(1))
        le(sr); le(sr * 2); le(UInt16(2)); le(UInt16(16))
        data.append(contentsOf: "data".utf8); le(UInt32(pcm.count * 2))
        pcm.forEach { le($0) }
        try? data.write(to: dir.appendingPathComponent("\(tag)_\(Int(Date().timeIntervalSince1970)).wav"))
    }
}
#endif

// MARK: - Text Processing

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
