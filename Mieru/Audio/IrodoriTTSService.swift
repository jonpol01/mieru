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
    private var generateTask: Task<Void, Never>?

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
            MLX.GPU.set(cacheLimit: 32 * 1024 * 1024)
            MLX.GPU.clearCache()

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
        MLX.GPU.clearCache()
    }

    // MARK: - Speak

    func speak(_ text: String, language: String = "ja") {
        guard isEnabled, !text.isEmpty, let model else { return }

        stop()
        let myToken = playToken
        isSpeaking = true

        let cleanText = trimToCompleteSentences(text)
        guard !cleanText.isEmpty else { isSpeaking = false; return }

        // Irodori is a Japanese-only model: no language routing, and the "voice" is a
        // VoiceDesign caption. Default to the first preset so it matches the picker top.
        let voice = selectedVoice ?? Self.voices.first
        let capturedModel = model

        generateTask = Task.detached {
            do {
                if await self.isStale(myToken) { return }
                await MainActor.run { MLX.GPU.clearCache() }

                Self.log("synth start len=\(cleanText.count) voice=\(voice ?? "nil") — active=\(Self.activeMB)MB")
                // Whole-utterance generate (flow-matching — no per-sentence chunking).
                let audio = try await capturedModel.generate(
                    text: cleanText, voice: voice, refAudio: nil, refText: nil, language: nil)
                let samples = audio.asArray(Float.self)
                Self.log("synth OK \(samples.count) samples — active=\(Self.activeMB)MB peak=\(Self.peakMB)MB")
                NSLog("[Irodori] Generated %d samples", samples.count)

                if await self.isStale(myToken) { return }
                await self.playAudio(samples: samples, token: myToken)
            } catch {
                Self.log("synth ERROR: \(String(describing: error))")
                NSLog("[Irodori] Synthesis error: %@", String(describing: error))
                await MainActor.run { self.finishIfCurrent(myToken) }
            }
        }
    }

    private func isStale(_ token: Int) -> Bool { token != playToken }

    private func finishIfCurrent(_ token: Int) {
        guard token == playToken else { return }
        isSpeaking = false
    }

    // MARK: - Stop

    func stop() {
        playToken &+= 1
        generateTask?.cancel()
        generateTask = nil
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
    nonisolated static var activeMB: Int { Int(MLX.GPU.activeMemory / (1024 * 1024)) }
    /// Peak MLX memory in MB.
    nonisolated static var peakMB: Int { Int(MLX.GPU.peakMemory / (1024 * 1024)) }

    /// File-based log (device console is flaky). Pull via:
    /// xcrun devicectl device copy from … Documents/irodori.log
    nonisolated static func log(_ line: String) {
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let url = docs.appendingPathComponent("irodori.log")
        let entry = "[\(ISO8601DateFormatter().string(from: Date()))] \(line)\n"
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(Data(entry.utf8)); try? h.close()
        } else {
            try? entry.data(using: .utf8)?.write(to: url)
        }
        NSLog("[Irodori] %@", line)
    }
}

// MARK: - Text Processing

private func trimToCompleteSentences(_ text: String) -> String {
    let endings: [Character] = ["。", "！", "？", ".", "!", "?"]
    if let lastIdx = text.lastIndex(where: { endings.contains($0) }) {
        return String(text[...lastIdx])
    }
    return text
}
