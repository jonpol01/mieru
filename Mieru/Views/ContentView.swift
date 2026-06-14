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
    @State private var speechService = SpeechService()        // Kokoro (CoreML / ANE)
    @State private var irodoriService = IrodoriTTSService()   // Irodori (MLX, JA VoiceDesign)
    @State private var catalog = ModelCatalog.shared

    private let irodoriId = "mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit"

    /// Whichever TTS engine the catalog has marked active.
    private var isIrodoriActive: Bool { catalog.activeTTSId == irodoriId }

    @State private var descriptionText = ""
    @State private var isThinking = false
    @State private var isTyping = false
    @State private var isAutoMode = false
    @State private var autoTimer: Timer?
    @State private var language = "ja"
    @State private var voiceEnabled = false
    @State private var showModels = false

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

                    // Models info button
                    Button { showModels = true } label: {
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundColor(.white)
                            .frame(width: 36, height: 28)
                            .background(Color.black.opacity(0.7))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .strokeBorder(Color.white, lineWidth: 2)
                            )
                    }

                    // Voice toggle
                    VoiceToggle(isEnabled: $voiceEnabled)

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
                    isTyping: $isTyping,
                    voiceMode: voiceEnabled
                )

                // DQ Button — しらべる or キャンセル
                ControlsOverlay(
                    isModelReady: vlmService.isReady,
                    isModelLoading: vlmService.isDownloading || (vlmService.isLoaded == false && vlmService.statusMessage != ""),
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
                Task {
                    // Use the catalog's active selection (persisted across launches)
                    vlmService.currentModelId = catalog.defaultVLMId
                    await vlmService.load()
                    await loadActiveTTS()
                    await catalog.scanCache()
                }
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
        .sheet(isPresented: $showModels) {
            ModelsView(
                vlmService: vlmService,
                speechService: speechService,
                irodoriService: irodoriService,
                onActivateVLM: { modelId in
                    await activateVLM(modelId: modelId)
                },
                onActivateTTS: { modelId in
                    await activateTTS(modelId: modelId)
                }
            )
        }
    }

    // MARK: - TTS Engine Dispatch

    /// Load only the active TTS engine (one resident at a time to save memory).
    private func loadActiveTTS() async {
        if isIrodoriActive {
            await irodoriService.load()
        } else {
            await speechService.load()
        }
    }

    private func ttsSpeak(_ text: String) {
        if isIrodoriActive {
            irodoriService.speak(text, language: language)
        } else {
            speechService.speak(text, language: language)
        }
    }

    /// Stop both engines — cheap and avoids a stuck stream after an engine switch.
    private func ttsStop() {
        speechService.stop()
        irodoriService.stop()
    }

    // MARK: - Model Activation

    private func activateVLM(modelId: String) async {
        ttsStop()
        let success = await vlmService.switchModel(to: modelId)

        if success {
            // Persist the selection in the catalog
            if let model = catalog.model(id: modelId) {
                catalog.setActiveVLM(model)
            }
        } else {
            // Restore catalog selection to whatever is actually loaded
            if let restored = catalog.model(id: vlmService.currentModelId) {
                catalog.setActiveVLM(restored)
            }
        }
        await catalog.scanCache()
    }

    /// Switch the active TTS engine: stop both, persist selection, unload the other,
    /// load the chosen one. Only one TTS engine is resident at a time.
    private func activateTTS(modelId: String) async {
        ttsStop()
        guard let model = catalog.model(id: modelId) else { return }
        catalog.setActiveTTS(model)

        if modelId == irodoriId {
            speechService.unload()
            await irodoriService.load()
        } else {
            irodoriService.unload()
            await speechService.load()
        }
        await catalog.scanCache()
    }

    // MARK: - Capture & Describe

    private func captureAndDescribe() {
        guard let frame = cameraManager.latestFrame else { return }

        // Load model on first use, then auto-capture
        guard vlmService.isReady else {
            guard !vlmService.isDownloading else { return }
            Task {
                await vlmService.load()
                captureAndDescribe()
            }
            return
        }

        ttsStop()

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

            if voiceEnabled {
                // Irodori is a heavy MLX model (~1.3 GB) whose flow-matching synthesis
                // spikes memory. Keeping Gemma (~1.5 GB) resident through that spike
                // (plus the camera) overflows the budget → Jetsam. The description text
                // is already produced, so free the brain before synthesizing; the next
                // しらべる lazily reloads it. Kokoro (CoreML/ANE) is light — leave Gemma in.
                if isIrodoriActive {
                    vlmService.unload()
                }
                ttsSpeak(result)
            }
        }
    }

    private func cancelDescribe() {
        generation += 1
        isThinking = false
        isTyping = false
        descriptionText = ""
        ttsStop()
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
        ttsStop()
    }

    private func handleForeground() {
        // Re-assert keep-awake — it was cleared on background.
        UIApplication.shared.isIdleTimerDisabled = true
        cameraManager.start()
        // Model reload triggers via onChange(cameraManager.isRunning)
    }
}

#Preview {
    ContentView()
}
