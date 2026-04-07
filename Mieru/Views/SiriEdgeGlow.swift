//
//  SiriEdgeGlow.swift
//  Mieru
//
//  Animated edge glow effect inspired by Apple's Siri UI.
//  Colors flow around the screen border while AI is active.
//

import SwiftUI

struct SiriEdgeGlow: View {
    let isActive: Bool

    @State private var rotation: Double = 0
    @State private var pulse: CGFloat = 1.0

    // Siri-like colors
    private let colors: [Color] = [
        .purple,
        Color(red: 0.4, green: 0.2, blue: 1.0),  // indigo
        .cyan,
        Color(red: 0.0, green: 0.8, blue: 0.6),   // teal
        .cyan,
        Color(red: 0.4, green: 0.2, blue: 1.0),   // indigo
        .purple,
        Color(red: 0.8, green: 0.2, blue: 0.6),   // magenta
        .purple,
    ]

    var body: some View {
        GeometryReader { geo in
            if isActive {
                ZStack {
                    // Outer soft glow (wide blur)
                    RoundedRectangle(cornerRadius: 44)
                        .strokeBorder(
                            AngularGradient(
                                colors: colors,
                                center: .center,
                                angle: .degrees(rotation)
                            ),
                            lineWidth: 8
                        )
                        .blur(radius: 20)
                        .opacity(0.6 * pulse)

                    // Mid glow
                    RoundedRectangle(cornerRadius: 44)
                        .strokeBorder(
                            AngularGradient(
                                colors: colors,
                                center: .center,
                                angle: .degrees(rotation + 30)
                            ),
                            lineWidth: 4
                        )
                        .blur(radius: 8)
                        .opacity(0.8 * pulse)

                    // Sharp inner edge
                    RoundedRectangle(cornerRadius: 44)
                        .strokeBorder(
                            AngularGradient(
                                colors: colors,
                                center: .center,
                                angle: .degrees(rotation)
                            ),
                            lineWidth: 2
                        )
                        .opacity(pulse)
                }
                .ignoresSafeArea()
                .allowsHitTesting(false)
                .transition(.opacity)
                .onAppear {
                    startAnimations()
                }
            }
        }
        .animation(.easeInOut(duration: 0.4), value: isActive)
    }

    private func startAnimations() {
        // Continuous rotation
        withAnimation(
            .linear(duration: 4)
            .repeatForever(autoreverses: false)
        ) {
            rotation = 360
        }

        // Breathing pulse
        withAnimation(
            .easeInOut(duration: 1.5)
            .repeatForever(autoreverses: true)
        ) {
            pulse = 0.6
        }
    }
}

#Preview {
    ZStack {
        Color.black.ignoresSafeArea()
        SiriEdgeGlow(isActive: true)
    }
}
