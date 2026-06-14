//
//  SpeechService.swift
//  Mieru
//
//  On-device neural TTS using Kokoro-82M via CoreML/Neural Engine.
//

import AVFoundation
import KokoroTTS
import AudioCommon

@Observable
@MainActor
class SpeechService {

    var isSpeaking = false
    var isEnabled = true
    var isLoaded = false
    var isLoading = false
    var statusMessage = ""

    private var model: KokoroTTSModel?
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var synthesisTask: Task<Void, Never>?

    private let sampleRate: Double = 24000

    func load() async {
        guard !isLoaded, !isLoading else { return }
        isLoading = true
        statusMessage = "Loading Kokoro TTS…"

        do {
            let cacheDir = try HuggingFaceDownloader.getCacheDirectory(
                for: KokoroTTSModel.defaultModelId)
            await predownloadVoice("jf_alpha", to: cacheDir)

            let tts = try await KokoroTTSModel.fromPretrained(
                voice: "af_heart"
            ) { [weak self] progress, status in
                Task { @MainActor in
                    self?.statusMessage = "TTS: \(status)"
                }
            }

            NSLog("[TTS] Available voices: %@", tts.availableVoices.joined(separator: ", "))
            model = tts
            isLoaded = true
            statusMessage = ""
        } catch {
            statusMessage = "TTS error: \(error.localizedDescription)"
            NSLog("[TTS] Load error: %@", error.localizedDescription)
        }
        isLoading = false
    }

    private func predownloadVoice(_ voiceName: String, to cacheDir: URL) async {
        let voiceFile = cacheDir.appendingPathComponent("voices/\(voiceName).json")
        if FileManager.default.fileExists(atPath: voiceFile.path) { return }

        let urlString = "https://huggingface.co/\(KokoroTTSModel.defaultModelId)/resolve/main/voices/\(voiceName).json"
        guard let url = URL(string: urlString) else { return }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            try FileManager.default.createDirectory(
                at: voiceFile.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: voiceFile)
            NSLog("[TTS] Downloaded voice: %@", voiceName)
        } catch {
            NSLog("[TTS] Failed to download voice %@: %@", voiceName, error.localizedDescription)
        }
    }

    func speak(_ text: String, language: String = "ja") {
        guard isEnabled, !text.isEmpty, let model else { return }

        stop()
        isSpeaking = true

        let voice = language == "ja"
            ? (model.availableVoices.contains("jf_alpha") ? "jf_alpha" : "af_heart")
            : "af_heart"
        let lang = language == "ja" ? "ja" : "en"
        let capturedModel = model

        synthesisTask = Task.detached {
            do {
                // Only speak complete sentences — drop trailing incomplete text
                let cleanText = trimToCompleteSentences(text)
                guard !cleanText.isEmpty else {
                    await MainActor.run { self.isSpeaking = false }
                    return
                }

                let sentences = splitForKokoro(cleanText)
                var allSamples = [Float]()

                for sentence in sentences {
                    if Task.isCancelled { return }
                    let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { continue }

                    let samples = try capturedModel.synthesize(
                        text: trimmed, voice: voice, language: lang, speed: 1.0)
                    allSamples.append(contentsOf: samples)
                }

                if Task.isCancelled { return }

                // Play the full audio as one smooth buffer
                await self.playAudio(samples: allSamples)
            } catch {
                NSLog("[TTS] Synthesis error: %@", error.localizedDescription)
                await MainActor.run { self.isSpeaking = false }
            }
        }
    }

    func stop() {
        synthesisTask?.cancel()
        synthesisTask = nil
        playerNode?.stop()
        if let engine = audioEngine, engine.isRunning {
            engine.stop()
        }
        audioEngine = nil
        playerNode = nil
        isSpeaking = false
    }

    private func playAudio(samples: [Float]) {
        guard !samples.isEmpty else {
            isSpeaking = false
            return
        }

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else { isSpeaking = false; return }

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else { isSpeaking = false; return }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        let channelData = buffer.floatChannelData![0]
        for i in 0..<samples.count {
            channelData[i] = samples[i]
        }

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)

            let engine = AVAudioEngine()
            let player = AVAudioPlayerNode()
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
            try engine.start()

            player.scheduleBuffer(buffer) { [weak self] in
                Task { @MainActor in
                    self?.isSpeaking = false
                }
            }
            player.play()

            self.audioEngine = engine
            self.playerNode = player
        } catch {
            NSLog("[TTS] Playback error: %@", error.localizedDescription)
            isSpeaking = false
        }
    }

    func unload() {
        stop()
        model = nil
        isLoaded = false
    }
}

// MARK: - Text Processing

/// Remove trailing incomplete sentence (no ending punctuation).
private func trimToCompleteSentences(_ text: String) -> String {
    let endings: [Character] = ["。", "！", "？", ".", "!", "?"]
    // Find the last sentence-ending punctuation
    if let lastIdx = text.lastIndex(where: { endings.contains($0) }) {
        return String(text[...lastIdx])
    }
    // No sentence endings found — return full text as-is
    return text
}

/// Split text into short chunks that stay under Kokoro's 128 phoneme token limit.
/// Japanese chars map to ~3 phoneme tokens each, so max ~35 chars per chunk.
private func splitForKokoro(_ text: String, maxChars: Int = 35) -> [String] {
    // Step 1: Split on sentence endings and clause markers
    let pattern = "(?<=[。！？、，\\.!?,;:])\\s*"
    guard let regex = try? NSRegularExpression(pattern: pattern) else {
        return breakLongChunk(text, maxChars: maxChars)
    }

    let range = NSRange(text.startIndex..., in: text)
    var rawChunks = [String]()
    var lastEnd = text.startIndex

    regex.enumerateMatches(in: text, range: range) { match, _, _ in
        guard let matchRange = match?.range, let swiftRange = Range(matchRange, in: text) else { return }
        rawChunks.append(String(text[lastEnd..<swiftRange.upperBound]))
        lastEnd = swiftRange.upperBound
    }
    if lastEnd < text.endIndex {
        rawChunks.append(String(text[lastEnd...]))
    }

    // Step 2: Break any chunk that's still too long
    var result = [String]()
    for chunk in rawChunks {
        let trimmed = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count <= maxChars {
            result.append(trimmed)
        } else {
            result.append(contentsOf: breakLongChunk(trimmed, maxChars: maxChars))
        }
    }

    return result.filter { !$0.isEmpty }
}

/// Break a long chunk into pieces at particle/space boundaries.
private func breakLongChunk(_ text: String, maxChars: Int) -> [String] {
    guard text.count > maxChars else { return [text] }

    // Japanese particles and natural break points
    let breakChars: Set<Character> = ["は", "が", "を", "に", "で", "と", "の", "へ", "も", "か", "ら", " ", "　"]
    var chunks = [String]()
    var current = ""

    for char in text {
        current.append(char)
        if current.count >= maxChars, breakChars.contains(char) {
            chunks.append(current)
            current = ""
        }
    }
    if !current.isEmpty {
        chunks.append(current)
    }

    return chunks
}
