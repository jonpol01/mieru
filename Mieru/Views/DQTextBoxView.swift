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

    /// Whether typewriter is still revealing text.
    @Binding var isTyping: Bool

    /// Called when the typewriter finishes revealing all text.
    var onComplete: (() -> Void)?

    // MARK: - State

    @State private var revealedCount = 0
    @State private var typingTimer: Timer?
    @State private var cursorVisible = true
    @State private var cursorTimer: Timer?
    @State private var sfx = TypewriterSFX()
    @State private var slimeFrame = 0
    @State private var autoScroll = true
    @State private var scrollViewHeight: CGFloat = 0

    /// Characters per second for the typewriter effect.
    private let charsPerSecond: Double = 20

    // MARK: - Body

    var body: some View {
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
                GeometryReader { outerGeo in
                    ScrollViewReader { proxy in
                        ScrollView {
                            VStack(alignment: .leading, spacing: 0) {
                                if isThinking {
                                    thinkingView
                                } else {
                                    typewriterText
                                }

                                // Bottom anchor
                                Color.clear.frame(height: 1).id("bottom")
                                    .onAppear { autoScroll = true }
                                    .onDisappear { autoScroll = false }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .onChange(of: revealedCount) { _, _ in
                            if autoScroll {
                                proxy.scrollTo("bottom", anchor: .bottom)
                            }
                        }
                    }
                    .onAppear { scrollViewHeight = outerGeo.size.height }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
            }
            .frame(maxHeight: 180)
            .padding(.horizontal, 16)
            .transition(.move(edge: .bottom).combined(with: .opacity))
            .animation(.easeOut(duration: 0.3), value: text.isEmpty && !isThinking)
            .onChange(of: text) { _, newText in
                startTypewriter(for: newText)
            }
            .onDisappear {
                stopTypewriter()
                stopCursorBlink()
            }
        }
    }

    // MARK: - Subviews

    /// Bouncing dots thinking indicator.
    private var thinkingView: some View {
        HStack(spacing: 6) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .fill(Color.white)
                    .frame(width: 8, height: 8)
                    .offset(y: slimeFrame == i ? -6 : 0)
            }
        }
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .onAppear {
            Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { _ in
                withAnimation(.easeInOut(duration: 0.2)) {
                    slimeFrame = (slimeFrame + 1) % 3
                }
            }
        }
    }

    private var typewriterText: some View {
        HStack(alignment: .bottom) {
            Text(String(text.prefix(revealedCount)))
                .font(.system(size: 20, weight: .medium))
                .foregroundColor(.white)
                .frame(maxWidth: .infinity, alignment: .leading)

            // Blinking triangle cursor (DQ style) when done
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

    func startTypewriter(for newText: String) {
        stopTypewriter()
        guard !newText.isEmpty else {
            isTyping = false
            return
        }

        revealedCount = 0
        isTyping = true
        let interval = 1.0 / charsPerSecond

        typingTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { timer in
            guard revealedCount < newText.count else {
                timer.invalidate()
                typingTimer = nil
                isTyping = false
                onComplete?()
                return
            }

            revealedCount += 1

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

// MARK: - Blinking Cursor

private struct BlinkingCursor: ViewModifier {
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
