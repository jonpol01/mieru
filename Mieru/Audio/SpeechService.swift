//
//  SpeechService.swift
//  Mieru
//
//  Speaks AI-generated text using iOS AVSpeechSynthesizer.
//

import AVFoundation

@Observable
class SpeechService: NSObject, AVSpeechSynthesizerDelegate {

    var isSpeaking = false
    var isEnabled = true

    private let synthesizer = AVSpeechSynthesizer()

    override init() {
        super.init()
        synthesizer.delegate = self

        // Ensure audio session allows playback with the blip SFX
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
    }

    /// Speak text with the appropriate language voice.
    func speak(_ text: String, language: String = "ja") {
        guard isEnabled, !text.isEmpty else { return }

        // Stop any current speech
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }

        let utterance = AVSpeechUtterance(string: text)

        if language == "ja" {
            utterance.voice = AVSpeechSynthesisVoice(language: "ja-JP")
            utterance.rate = 0.48
            utterance.pitchMultiplier = 1.15
        } else {
            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
            utterance.rate = 0.45
            utterance.pitchMultiplier = 1.1
        }

        utterance.volume = 0.9
        utterance.preUtteranceDelay = 0.3

        isSpeaking = true
        synthesizer.speak(utterance)
    }

    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
    }

    // MARK: - AVSpeechSynthesizerDelegate

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
        }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.isSpeaking = false
        }
    }
}
