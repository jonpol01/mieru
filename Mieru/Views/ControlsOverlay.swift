//
//  ControlsOverlay.swift
//  Mieru
//
//  Capture button, auto-mode toggle, and model status indicator.
//

import SwiftUI

struct ControlsOverlay: View {
    let isModelReady: Bool
    let isProcessing: Bool
    let isAutoMode: Bool
    let statusMessage: String
    let downloadProgress: Float

    var onCapture: () -> Void
    var onToggleAutoMode: () -> Void

    var body: some View {
        VStack {
            // Status bar at top
            statusBar
                .padding(.top, 60)

            Spacer()

            // Controls at bottom-right, above the text box
            HStack {
                Spacer()

                VStack(spacing: 16) {
                    // Auto-mode toggle
                    Button(action: onToggleAutoMode) {
                        Image(systemName: isAutoMode ? "arrow.triangle.2.circlepath.circle.fill" : "arrow.triangle.2.circlepath.circle")
                            .font(.system(size: 32))
                            .foregroundColor(isAutoMode ? .green : .white)
                            .shadow(color: .black.opacity(0.6), radius: 4)
                    }

                    // Capture button
                    Button(action: onCapture) {
                        ZStack {
                            Circle()
                                .fill(Color.white.opacity(0.3))
                                .frame(width: 72, height: 72)

                            Circle()
                                .fill(isProcessing ? Color.gray : Color.white)
                                .frame(width: 60, height: 60)

                            if isProcessing {
                                ProgressView()
                                    .tint(.black)
                            }
                        }
                    }
                    .disabled(isProcessing || !isModelReady)
                }
                .padding(.trailing, 24)
                .padding(.bottom, 220) // above the DQ text box
            }
        }
    }

    // MARK: - Status Bar

    private var statusBar: some View {
        HStack(spacing: 8) {
            // Model status dot
            Circle()
                .fill(isModelReady ? Color.green : Color.orange)
                .frame(width: 8, height: 8)

            if downloadProgress > 0 && downloadProgress < 1 {
                ProgressView(value: Double(downloadProgress))
                    .frame(width: 100)
                    .tint(.white)
            }

            Text(statusMessage)
                .font(.system(size: 13, weight: .medium, design: .monospaced))
                .foregroundColor(.white)
                .shadow(color: .black.opacity(0.8), radius: 3)

            Spacer()
        }
        .padding(.horizontal, 20)
    }
}
