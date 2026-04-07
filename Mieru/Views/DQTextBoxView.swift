//
//  DQTextBoxView.swift
//  Mieru
//
//  Dragon Quest-style text box with typewriter animation and blip SFX.
//

import SwiftUI

struct DQTextBoxView: View {

    /// The full text to display (typewriter reveals it over time).
    let text: String

    /// Whether we're waiting for the AI to respond.
    let isThinking: Bool

    /// Called when the typewriter finishes revealing all text.
    var onComplete: (() -> Void)?

    // MARK: - State

    @State private var revealedCount = 0
    @State private var typingTimer: Timer?
    @State private var cursorVisible = true
    @State private var cursorTimer: Timer?
    @State private var sfx = TypewriterSFX()

    /// Characters per second for the typewriter effect.
    private let charsPerSecond: Double = 20

    // MARK: - Body

    var body: some View {
        VStack {
            Spacer()

            if !text.isEmpty || isThinking {
                ZStack {
                    // DQ double-border box
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.black.opacity(0.88))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .strokeBorder(Color.white, lineWidth: 3)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .strokeBorder(Color.white.opacity(0.5), lineWidth: 1)
                                .padding(6)
                        )

                    // Text content
                    VStack(alignment: .leading, spacing: 0) {
                        if isThinking {
                            thinkingView
                        } else {
                            typewriterText
                        }
                    }
                    .padding(.horizontal, 24)
                    .padding(.vertical, 20)
                }
                .frame(maxHeight: 180)
                .padding(.horizontal, 16)
                .padding(.bottom, 24)
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .animation(.easeOut(duration: 0.3), value: text.isEmpty && !isThinking)
            }
        }
        .onChange(of: text) { _, newText in
            startTypewriter(for: newText)
        }
    }

    // MARK: - Subviews

    private var thinkingView: some View {
        HStack(spacing: 4) {
            Text("Thinking")
                .font(.system(size: 20, weight: .medium, design: .monospaced))
                .foregroundColor(.white)
            ThinkingDots()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var typewriterText: some View {
        HStack(alignment: .bottom) {
            Text(String(text.prefix(revealedCount)))
                .font(.system(size: 20, weight: .medium, design: .monospaced))
                .foregroundColor(.white)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Blinking triangle cursor (DQ style)
            if revealedCount >= text.count && !text.isEmpty {
                Triangle()
                    .fill(Color.white)
                    .frame(width: 12, height: 10)
                    .opacity(cursorVisible ? 1 : 0)
                    .onAppear { startCursorBlink() }
                    .onDisappear { stopCursorBlink() }
            }
        }
    }

    // MARK: - Typewriter Logic

    private func startTypewriter(for newText: String) {
        stopTypewriter()
        guard !newText.isEmpty else { return }

        revealedCount = 0
        let interval = 1.0 / charsPerSecond

        typingTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { timer in
            guard revealedCount < newText.count else {
                timer.invalidate()
                typingTimer = nil
                onComplete?()
                return
            }

            revealedCount += 1

            // Play blip for visible characters (skip spaces)
            let idx = newText.index(newText.startIndex, offsetBy: revealedCount - 1)
            let char = newText[idx]
            if !char.isWhitespace {
                sfx.playBlip()
            }
        }
    }

    private func stopTypewriter() {
        typingTimer?.invalidate()
        typingTimer = nil
    }

    private func startCursorBlink() {
        cursorTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            cursorVisible.toggle()
        }
    }

    private func stopCursorBlink() {
        cursorTimer?.invalidate()
        cursorTimer = nil
        cursorVisible = true
    }
}

// MARK: - Thinking Dots Animation

private struct ThinkingDots: View {
    @State private var dotCount = 0
    @State private var timer: Timer?

    var body: some View {
        Text(String(repeating: ".", count: dotCount))
            .font(.system(size: 20, weight: .medium, design: .monospaced))
            .foregroundColor(.white)
            .frame(width: 40, alignment: .leading)
            .onAppear {
                timer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { _ in
                    dotCount = (dotCount % 3) + 1
                }
            }
            .onDisappear {
                timer?.invalidate()
            }
    }
}

// MARK: - DQ Triangle Cursor

private struct Triangle: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.midX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
        path.closeSubpath()
        return path
    }
}

#Preview {
    ZStack {
        Color.gray
        DQTextBoxView(
            text: "A vast meadow stretches before the hero. The wind carries the scent of adventure.",
            isThinking: false
        )
    }
}
