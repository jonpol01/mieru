//
//  VoiceToggle.swift
//  Mieru
//
//  DQ-style voice on/off toggle.
//

import SwiftUI

struct VoiceToggle: View {

    @Binding var isEnabled: Bool

    var body: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.25)) {
                isEnabled.toggle()
            }
        } label: {
            Image(systemName: isEnabled ? "speaker.wave.2.fill" : "speaker.slash.fill")
                .font(.system(size: 14, weight: .bold))
                .foregroundColor(isEnabled ? .black : .white.opacity(0.4))
                .frame(width: 36, height: 28)
                .background(isEnabled ? Color.white : Color.black.opacity(0.7))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(Color.white, lineWidth: 2)
                )
        }
    }
}
