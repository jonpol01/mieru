//
//  LanguageToggle.swift
//  Mieru
//
//  DQ-style language toggle button: EN ⇄ JA
//

import SwiftUI

struct LanguageToggle: View {

    @Binding var language: String

    private var isJapanese: Bool { language == "ja" }

    var body: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.25)) {
                language = isJapanese ? "en" : "ja"
            }
        } label: {
            HStack(spacing: 0) {
                // EN side
                Text("EN")
                    .font(.system(size: 12, weight: .heavy, design: .monospaced))
                    .foregroundColor(isJapanese ? .white.opacity(0.4) : .black)
                    .frame(width: 32, height: 28)
                    .background(isJapanese ? Color.clear : Color.white)

                // JA side
                Text("JA")
                    .font(.system(size: 12, weight: .heavy, design: .monospaced))
                    .foregroundColor(isJapanese ? .black : .white.opacity(0.4))
                    .frame(width: 32, height: 28)
                    .background(isJapanese ? Color.white : Color.clear)
            }
            .background(Color.black.opacity(0.7))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color.white, lineWidth: 2)
            )
        }
    }
}
