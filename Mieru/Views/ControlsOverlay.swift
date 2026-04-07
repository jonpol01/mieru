//
//  ControlsOverlay.swift
//  Mieru
//
//  DQ-style しらべる / キャンセル button and model status.
//

import SwiftUI

struct ControlsOverlay: View {
    let isModelReady: Bool
    let isProcessing: Bool
    let isAutoMode: Bool
    let statusMessage: String
    let downloadProgress: Float

    var onCapture: () -> Void
    var onCancel: (() -> Void)?
    var onToggleAutoMode: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            if isProcessing {
                cancelButton
            } else {
                captureButton
            }
        }
    }

    // MARK: - DQ Capture Button

    private var captureButton: some View {
        Button(action: onCapture) {
            HStack(spacing: 10) {
                Text("▶")
                    .font(.system(size: 18))
                    .foregroundColor(.white)
                    .modifier(BlinkModifier())

                Text("しらべる")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundColor(.white)
            }
            .padding(.horizontal, 36)
            .padding(.vertical, 14)
            .background(Color.black.opacity(0.9))
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(Color.white, lineWidth: 3)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(Color.white.opacity(0.4), lineWidth: 1)
                    .padding(5)
            )
        }
        .opacity(isModelReady ? 1.0 : 0.7)
    }

    // MARK: - DQ Cancel Button

    private var cancelButton: some View {
        Button(action: { onCancel?() }) {
            Text("キャンセル")
                .font(.system(size: 22, weight: .bold))
                .foregroundColor(Color(red: 1.0, green: 0.4, blue: 0.4))
                .padding(.horizontal, 36)
                .padding(.vertical, 14)
                .background(Color.black.opacity(0.9))
                .clipShape(RoundedRectangle(cornerRadius: 14))
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .strokeBorder(Color(red: 1.0, green: 0.4, blue: 0.4), lineWidth: 3)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(Color(red: 1.0, green: 0.4, blue: 0.4).opacity(0.4), lineWidth: 1)
                        .padding(5)
                )
        }
    }
}

// MARK: - Blink Modifier

private struct BlinkModifier: ViewModifier {
    @State private var visible = true

    func body(content: Content) -> some View {
        content
            .opacity(visible ? 1 : 0)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.4).repeatForever(autoreverses: true)) {
                    visible = false
                }
            }
    }
}
