//
//  TypewriterSFX.swift
//  Mieru
//
//  Dragon Quest-style typewriter blip sound effect.
//  Generates an 8-bit style blip programmatically — no asset file needed.
//

import AVFoundation

class TypewriterSFX {

    private var audioPlayer: AVAudioPlayer?
    private var blipData: Data?

    init() {
        blipData = Self.generateBlip()
        preparePlayer()
    }

    /// Play one blip. Fast enough for per-character calls.
    func playBlip() {
        // Reset to start and play — overlapping is fine for short blips
        audioPlayer?.currentTime = 0
        audioPlayer?.play()
    }

    // MARK: - Private

    private func preparePlayer() {
        guard let data = blipData else { return }
        do {
            audioPlayer = try AVAudioPlayer(data: data)
            audioPlayer?.prepareToPlay()
            audioPlayer?.volume = 0.4
        } catch {
            NSLog("[SFX] Failed to create player: %@", error.localizedDescription)
        }
    }

    /// Generate a short DQ-style blip as a WAV in memory.
    /// Square wave at ~660Hz, ~60ms duration, 8-bit mono.
    private static func generateBlip() -> Data {
        let sampleRate: Double = 22050
        let duration: Double = 0.06
        let frequency: Double = 660 // DQ-like pitch
        let numSamples = Int(sampleRate * duration)

        // Generate square wave samples (8-bit unsigned)
        var samples = [UInt8](repeating: 128, count: numSamples)
        for i in 0..<numSamples {
            let t = Double(i) / sampleRate
            let phase = t * frequency
            let square: Double = (phase - floor(phase)) < 0.5 ? 1.0 : -1.0

            // Quick fade-out envelope for the last 30% to avoid click
            let fadeStart = Double(numSamples) * 0.7
            let envelope: Double
            if Double(i) > fadeStart {
                envelope = 1.0 - (Double(i) - fadeStart) / (Double(numSamples) - fadeStart)
            } else {
                envelope = 1.0
            }

            let amplitude = 0.6 * envelope
            samples[i] = UInt8(128.0 + square * amplitude * 127.0)
        }

        // Build WAV file in memory
        return buildWAV(samples: samples, sampleRate: Int(sampleRate))
    }

    private static func buildWAV(samples: [UInt8], sampleRate: Int) -> Data {
        var data = Data()

        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 8
        let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = UInt32(samples.count)
        let fileSize = 36 + dataSize

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) }) // chunk size
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM
        data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })
        data.append(contentsOf: samples)

        return data
    }
}
