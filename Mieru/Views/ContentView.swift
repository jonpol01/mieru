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
    @State private var isAutoMode = false
    @State private var autoTimer: Timer?

    /// Tracks the generation to discard stale results.
    @State private var generation = 0

    var body: some View {
        ZStack {
            // Layer 1: Full-screen camera
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            // Layer 2: DQ Text Box
            DQTextBoxView(
                text: descriptionText,
                isThinking: isThinking
            )

            // Layer 3: Siri-style edge glow while thinking
            SiriEdgeGlow(isActive: isThinking)

            // Layer 4: Controls
            ControlsOverlay(
                isModelReady: vlmService.isReady,
                isProcessing: vlmService.running,
                isAutoMode: isAutoMode,
                statusMessage: vlmService.statusMessage,
                downloadProgress: vlmService.downloadProgress,
                onCapture: { captureAndDescribe() },
                onToggleAutoMode: { toggleAutoMode() }
            )
        }
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            cameraManager.setup()
            cameraManager.start()
            Task { await vlmService.load() }
        }
        .onChange(of: isAutoMode) { _, auto in
            if auto { startAutoMode() } else { stopAutoMode() }
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)) {_ in
            handleBackground()
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in
            handleForeground()
        }
        .statusBarHidden()
    }

    // MARK: - Capture & Describe

    private func captureAndDescribe() {
        guard vlmService.isReady, let frame = cameraManager.latestFrame else { return }

        generation += 1
        let currentGen = generation
        isThinking = true
        descriptionText = ""

        Task {
            let result = await vlmService.describe(pixelBuffer: frame)

            // Discard if a newer request was fired
            guard currentGen == generation else { return }

            isThinking = false
            descriptionText = result
        }
    }

    // MARK: - Auto Mode

    private func toggleAutoMode() {
        isAutoMode.toggle()
    }

    private func startAutoMode() {
        // Immediately capture once
        captureAndDescribe()

        // Then repeat every 6 seconds (enough for inference to finish)
        autoTimer = Timer.scheduledTimer(withTimeInterval: 6.0, repeats: true) { _ in
            if vlmService.isReady {
                captureAndDescribe()
            }
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
        Task { await vlmService.load() }
    }
}

#Preview {
    ContentView()
}
