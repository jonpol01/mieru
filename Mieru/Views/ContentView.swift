//
//  ContentView.swift
//  Mieru
//
//  Main view: full-screen camera with DQ text box overlay.
//

import SwiftUI

struct ContentView: View {

    @State private var cameraManager = CameraManager()
    @State private var vlmService = VLMService()

    @State private var descriptionText = ""
    @State private var isThinking = false
    @State private var isTyping = false
    @State private var isAutoMode = false
    @State private var autoTimer: Timer?
    @State private var language = "ja"

    /// Tracks the generation to discard stale results.
    @State private var generation = 0

    /// Whether to show cancel (thinking OR typing)
    private var showCancel: Bool { isThinking || isTyping }

    var body: some View {
        ZStack {
            // Layer 1: Full-screen camera
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            // Layer 2: Status bar at top + language toggle
            VStack {
                HStack(spacing: 8) {
                    Circle()
                        .fill(vlmService.isReady ? Color.green : Color.orange)
                        .frame(width: 8, height: 8)
                    if vlmService.downloadProgress > 0 && vlmService.downloadProgress < 1 {
                        ProgressView(value: Double(vlmService.downloadProgress))
                            .frame(width: 100)
                            .tint(.white)
                    }
                    Text(vlmService.statusMessage)
                        .font(.system(size: 13, weight: .medium, design: .monospaced))
                        .foregroundColor(.white)
                        .shadow(color: .black.opacity(0.8), radius: 3)
                    Spacer()

                    // Language toggle
                    LanguageToggle(language: $language)
                }
                .padding(.horizontal, 20)
                .padding(.top, 60)
                Spacer()
            }

            // Layer 3: Siri-style edge glow while thinking
            SiriEdgeGlow(isActive: isThinking)

            // Layer 4: Text box + button stacked vertically at bottom
            VStack(spacing: 12) {
                Spacer()

                // DQ Text Box
                DQTextBoxView(
                    text: descriptionText,
                    isThinking: isThinking,
                    isTyping: $isTyping
                )

                // DQ Button — しらべる or キャンセル
                ControlsOverlay(
                    isModelReady: vlmService.isReady,
                    isProcessing: showCancel,
                    isAutoMode: isAutoMode,
                    statusMessage: vlmService.statusMessage,
                    downloadProgress: vlmService.downloadProgress,
                    onCapture: { captureAndDescribe() },
                    onCancel: { cancelDescribe() },
                    onToggleAutoMode: { toggleAutoMode() }
                )
            }
            .padding(.bottom, 32)
        }
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            cameraManager.setup()
            cameraManager.start()
        }
        .onChange(of: cameraManager.isRunning) { _, running in
            if running {
                Task { await vlmService.load() }
            }
        }
        .onChange(of: isAutoMode) { _, auto in
            if auto { startAutoMode() } else { stopAutoMode() }
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)) { _ in
            handleBackground()
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in
            handleForeground()
        }
        .statusBarHidden()
    }

    // MARK: - Capture & Describe

    private func captureAndDescribe() {
        guard let frame = cameraManager.latestFrame else { return }

        // Load model on first use, then auto-capture
        guard vlmService.isReady else {
            Task {
                await vlmService.load()
                captureAndDescribe()
            }
            return
        }

        generation += 1
        let currentGen = generation
        isThinking = true
        isTyping = false
        descriptionText = ""

        Task {
            let result = await vlmService.describe(pixelBuffer: frame, language: language)

            guard currentGen == generation else { return }

            isThinking = false
            descriptionText = result
        }
    }

    private func cancelDescribe() {
        generation += 1
        isThinking = false
        isTyping = false
        descriptionText = ""
    }

    // MARK: - Auto Mode

    private func toggleAutoMode() {
        isAutoMode.toggle()
    }

    private func startAutoMode() {
        captureAndDescribe()
        autoTimer = Timer.scheduledTimer(withTimeInterval: 6.0, repeats: true) { _ in
            if vlmService.isReady { captureAndDescribe() }
        }
    }

    private func stopAutoMode() {
        autoTimer?.invalidate()
        autoTimer = nil
    }

    // MARK: - Lifecycle

    private func handleBackground() {
        UIApplication.shared.isIdleTimerDisabled = false
        stopAutoMode()
        isAutoMode = false
        cameraManager.stop()
        vlmService.unload()
    }

    private func handleForeground() {
        cameraManager.start()
        // Model reload triggers via onChange(cameraManager.isRunning)
    }
}

#Preview {
    ContentView()
}
