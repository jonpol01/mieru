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
    @State private var isSynthesizing = false   // voice being prepared (after text ready)
    @State private var isTyping = false
    @State private var isAutoMode = false
    @State private var autoTimer: Timer?
    @State private var language = "ja"
    @State private var voiceEnabled = false
    @State private var showModels = false
    @AppStorage("showPerfStats") private var showPerfStats = false

    /// Idle power-saving: unload models after inactivity so the GPU/ANE go cold.
    @State private var idleTimer: Timer?
    @State private var isIdle = false
    private let idleTimeout: TimeInterval = 120   // 2 min of no captures → unload

    /// Tracks the generation to discard stale results.
    @State private var generation = 0

    /// Whether to show cancel (thinking, synthesizing voice, OR typing)
    private var showCancel: Bool { isThinking || isSynthesizing || isTyping }

    var body: some View {
        ZStack {
            // Layer 1: Full-screen camera
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            // Layer 2: Status bar at top + language toggle
            VStack {
                HStack(spacing: 8) {
                    Circle()
                        .fill(isIdle ? Color.gray : (vlmService.isReady ? Color.green : Color.orange))
                        .frame(width: 8, height: 8)
                    if vlmService.downloadProgress > 0 && vlmService.downloadProgress < 1 {
                        ProgressView(value: Double(vlmService.downloadProgress))
                            .frame(width: 100)
                            .tint(.white)
                    }
                    Text(isIdle ? "省電力 — タップで再開" : vlmService.statusMessage)
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

                if showPerfStats {
                    perfHUD
                        .padding(.horizontal, 20)
                        .padding(.top, 6)
                }

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
                    isSynthesizing: isSynthesizing,
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
                    resetIdleTimer()
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

    // MARK: - Perf HUD

    private var perfHUD: some View {
        HStack(spacing: 8) {
            perfPill(
                icon: "eye.fill",
                label: "VLM",
                value: vlmService.lastTokensPerSecond > 0
                    ? String(format: "%.1f tok/s", vlmService.lastTokensPerSecond)
                    : "—"
            )
            perfPill(
                icon: "speaker.wave.2.fill",
                label: "Voice",
                value: voiceRTF > 0
                    ? String(format: "%.2fx RT", voiceRTF)
                    : "—"
            )
            Spacer()
        }
    }

    private var voiceRTF: Double {
        isIrodoriActive ? irodoriService.lastRTF : speechService.lastRTF
    }

    private func perfPill(icon: String, label: String, value: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 10))
                .foregroundColor(.green)
            Text(label)
                .font(.system(size: 11, weight: .heavy))
                .foregroundColor(.white.opacity(0.6))
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .foregroundColor(.white)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Color.black.opacity(0.6))
        .clipShape(Capsule())
        .overlay(Capsule().strokeBorder(Color.white.opacity(0.2), lineWidth: 1))
        .shadow(color: .black.opacity(0.6), radius: 3)
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

    /// Synthesize + start playback on the active engine. Returns when audio begins.
    private func ttsSpeak(_ text: String) async {
        if isIrodoriActive {
            await irodoriService.speak(text, language: language)
        } else {
            await speechService.speak(text, language: language)
        }
    }

    /// Stop both engines — cheap and avoids a stuck stream after an engine switch.
    private func ttsStop() {
        speechService.stop()
        irodoriService.stop()
    }

    // MARK: - Model Activation

    private func activateVLM(modelId: String) async {
        // Free TTS weights before loading the (possibly large, e.g. QAT ~4.4 GB) VLM,
        // so the load peak doesn't collide with a resident TTS model → Jetsam.
        ttsStop()
        speechService.unload()
        irodoriService.unload()

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

        // Reload the active TTS engine now that the VLM is resident.
        await loadActiveTTS()
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

        // Load model on first use (or wake from idle unload), then auto-capture.
        guard vlmService.isReady else {
            guard !vlmService.isDownloading else { return }
            Task {
                isIdle = false
                await vlmService.load()
                await loadActiveTTS()   // idle may have unloaded the voice too
                captureAndDescribe()
            }
            return
        }

        ttsStop()
        isIdle = false
        idleTimer?.invalidate()   // pause idle countdown during active work

        generation += 1
        let currentGen = generation
        isThinking = true
        isSynthesizing = false
        isTyping = false
        descriptionText = ""

        Task {
            let result = await vlmService.describe(pixelBuffer: frame, language: language)

            guard currentGen == generation else { return }
            isThinking = false

            if voiceEnabled {
                // Sync text with voice: show a "preparing voice" state, synthesize, and
                // reveal the text (typewriter) only once audio actually starts.
                //
                // Irodori is a heavy MLX model (~1.3 GB) whose flow-matching synthesis
                // spikes memory; keeping Gemma (~1.5 GB) resident through that spike
                // (plus the camera) overflows the budget → Jetsam. The text is already
                // produced, so free the brain before synthesizing; the next しらべる
                // lazily reloads it. Kokoro (CoreML/ANE) is light — leave Gemma in.
                if isIrodoriActive {
                    vlmService.unload()
                }
                isSynthesizing = true
                await ttsSpeak(result)            // returns when audio starts (or fails)
                guard currentGen == generation else { return }
                isSynthesizing = false
                descriptionText = result          // typewriter + voice in sync
            } else {
                descriptionText = result          // no voice → reveal immediately
            }

            // Work done — start the idle countdown.
            resetIdleTimer()
        }
    }

    private func cancelDescribe() {
        generation += 1
        isThinking = false
        isSynthesizing = false
        isTyping = false
        descriptionText = ""
        ttsStop()
        resetIdleTimer()
    }

    // MARK: - Idle Power Saving

    /// (Re)start the inactivity countdown. After `idleTimeout` with no captures,
    /// unload the models so the GPU/ANE go cold (battery + heat).
    private func resetIdleTimer() {
        idleTimer?.invalidate()
        idleTimer = Timer.scheduledTimer(withTimeInterval: idleTimeout, repeats: false) { _ in
            Task { @MainActor in idleUnload() }
        }
    }

    private func idleUnload() {
        // Never unload mid-activity — reschedule and try later.
        guard !isThinking, !isSynthesizing, !isTyping,
              !speechService.isSpeaking, !irodoriService.isSpeaking else {
            resetIdleTimer()
            return
        }
        guard vlmService.isLoaded || speechService.isLoaded || irodoriService.isLoaded else { return }

        vlmService.unload()
        speechService.unload()
        irodoriService.unload()
        isIdle = true
        VLMService.log("idle unload — models released after \(Int(idleTimeout))s")
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
        idleTimer?.invalidate()
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
