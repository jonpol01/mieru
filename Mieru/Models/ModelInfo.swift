//
//  ModelInfo.swift
//  Mieru
//
//  Model registry data model and enums.
//

import SwiftUI

// MARK: - ModelInfo

struct ModelInfo: Identifiable, Codable, Hashable {
    let id: String                  // HF id: "mlx-community/gemma-4-e2b-it-4bit"
    let displayName: String         // "Gemma 4 E2B 4-bit"
    let family: ModelFamily
    let kind: ModelKind
    let architecture: String        // "Gemma 4", "Kokoro"
    let modelType: String           // "Gemma4_VLM", "Kokoro_TTS"
    let format: ModelFormat
    let contextLength: Int          // 131072 for Gemma 4
    let batch: Int
    let chunks: Int
    let estimatedSizeGB: Double

    /// Optional warning label shown on detail page (e.g. "KV-sharing not supported")
    var compatibilityWarning: String?

    // Runtime — set after scanning cache
    var actualSizeBytes: Int64?

    var sizeGB: Double {
        if let bytes = actualSizeBytes {
            return Double(bytes) / 1_073_741_824
        }
        return estimatedSizeGB
    }

    var sizeLabel: String {
        let gb = sizeGB
        if gb < 1.0 {
            let mb = gb * 1024
            return String(format: "%.0f MB", mb)
        }
        return String(format: "%.1f GB", gb)
    }

    var shortId: String {
        let parts = id.split(separator: "/")
        guard parts.count == 2 else { return id }
        let name = String(parts[1])
        if name.count > 22 {
            let prefix = name.prefix(10)
            let suffix = name.suffix(10)
            return "\(prefix)..\(suffix)"
        }
        return name
    }
}

// MARK: - Model Family

enum ModelFamily: String, Codable, CaseIterable {
    case gemma, qwen, llama, phi, kokoro, whisper, other

    var displayName: String {
        switch self {
        case .gemma: return "GEMMA"
        case .qwen: return "QWEN"
        case .llama: return "LLAMA"
        case .phi: return "PHI"
        case .kokoro: return "KOKORO"
        case .whisper: return "WHISPER"
        case .other: return "OTHER"
        }
    }

    var color: Color {
        switch self {
        case .gemma: return .orange
        case .qwen: return .purple
        case .llama: return .blue
        case .phi: return .pink
        case .kokoro: return .green
        case .whisper: return .cyan
        case .other: return .gray
        }
    }
}

// MARK: - Model Kind

enum ModelKind: String, Codable {
    case vlm        // Vision-language model (Gemma 4)
    case tts        // Text-to-speech
    case stt        // Speech-to-text

    var displayName: String {
        switch self {
        case .vlm: return "Vision"
        case .tts: return "Voice"
        case .stt: return "Listen"
        }
    }

    var icon: String {
        switch self {
        case .vlm: return "brain.head.profile"
        case .tts: return "waveform"
        case .stt: return "mic.fill"
        }
    }

    var sectionIcon: String {
        switch self {
        case .vlm: return "eye.fill"
        case .tts: return "speaker.wave.2.fill"
        case .stt: return "mic.fill"
        }
    }
}

// MARK: - Model Format

enum ModelFormat: String, Codable {
    case mlx
    case coreml
    case gguf

    var displayName: String {
        switch self {
        case .mlx: return "MLX"
        case .coreml: return "CoreML"
        case .gguf: return "GGUF"
        }
    }

    var color: Color {
        switch self {
        case .mlx: return .orange
        case .coreml: return .blue
        case .gguf: return .green
        }
    }
}
