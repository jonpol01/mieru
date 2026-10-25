//
//  ContentView.swift
//  Mieru
//
//  Main view: full-screen camera with DQ text box overlay.
//

import CoreML
import SwiftUI

struct ContentView: View {

    @State private var cameraManager = CameraManager()
    @State private var vlmService = VLMService()
    @State private var speechService = SpeechService()        // Kokoro (CoreML / ANE)
    @State private var irodoriService = IrodoriTTSService()   // Irodori (MLX, JA VoiceDesign)
    @State private var irodoriCoreMLService = IrodoriCoreMLService()   // Irodori v4.1 MF (Core ML, streamed)
    @State private var catalog = ModelCatalog.shared

    /// Whichever TTS engine the catalog has marked active.
    private var isIrodoriActive: Bool { IrodoriTTSService.modelIds.contains(catalog.defaultTTSId) }
    private var isIrodoriCoreMLActive: Bool { catalog.defaultTTSId == IrodoriCoreMLService.catalogId }

    /// Core ML Irodori is selected but its side-loaded files are missing or failed to load: the MLX
    /// v4.1-MF engine speaks instead. The selection is kept, so pushing the files restores Core ML.
    @State private var irodoriCoreMLFallback = false
    /// The engine that actually speaks.
    private var speaksIrodoriCoreML: Bool { isIrodoriCoreMLActive && !irodoriCoreMLFallback }
    private var speaksIrodoriMLX: Bool { isIrodoriActive || (isIrodoriCoreMLActive && irodoriCoreMLFallback) }

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
    /// D1A speak / skip / re-capture after each description (needs the adapter in Documents/d1a/adapter).
    @State private var lastSpokenDescription: String?
    #if DEBUG
    /// The camera restarts on every foreground, re-firing onChange(isRunning); run the harness once.
    @State private var debugHarnessStarted = false
    #endif

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
                    #if DEBUG
                    // -ttsBench times cold loads: start with neither Gemma nor a voice resident.
                    let skipModelLoad = ProcessInfo.processInfo.arguments.contains("-ttsBench")
                    #else
                    let skipModelLoad = false
                    #endif
                    if !skipModelLoad {
                        // Use the catalog's active selection (persisted across launches)
                        vlmService.currentModelId = catalog.defaultVLMId
                        await vlmService.load()
                        await loadActiveTTS()
                    }
                    await catalog.scanCache()
                    resetIdleTimer()
                    #if DEBUG
                    await runDebugLaunchArguments()
                    #endif
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
                irodoriCoreMLService: irodoriCoreMLService,
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
        speaksIrodoriMLX ? irodoriService.lastRTF
            : speaksIrodoriCoreML ? irodoriCoreMLService.lastRTF : speechService.lastRTF
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
            irodoriService.select(modelId: catalog.defaultTTSId)
            await irodoriService.load()
        } else if isIrodoriCoreMLActive {
            // After a failed Core ML load, reloads (wake, Gemma swaps) go straight to the MLX fallback;
            // Core ML is retried on the next launch or when it is picked again in the Models sheet.
            if irodoriCoreMLFallback { await irodoriService.load() } else { await loadIrodoriCoreML() }
        } else {
            await speechService.load()
        }
    }

    /// Core ML Irodori, or the MLX v4.1-MF engine when the Core ML files are missing or fail to load
    /// (there is no in-app download for them).
    private func loadIrodoriCoreML() async {
        await irodoriCoreMLService.load()
        while irodoriCoreMLService.isLoading { try? await Task.sleep(for: .milliseconds(100)) }   // a concurrent load
        irodoriCoreMLFallback = !irodoriCoreMLService.isLoaded
        if irodoriCoreMLFallback {
            IrodoriTTSService.log("coreml unavailable (\(irodoriCoreMLService.statusMessage)) — falling back to MLX v4.1-MF")
            irodoriService.select(modelId: IrodoriTTSService.v41MeanFlowId)
            await irodoriService.load()
        } else {
            irodoriService.unload()
        }
    }

    /// Synthesize + start playback on the active engine. Returns when audio begins.
    private func ttsSpeak(_ text: String) async {
        if speaksIrodoriMLX {
            await irodoriService.speak(text, language: language)
        } else if speaksIrodoriCoreML {
            await irodoriCoreMLService.speak(text, language: language)
        } else {
            await speechService.speak(text, language: language)
        }
    }

    /// Stop every engine — cheap and avoids a stuck stream after an engine switch.
    private func ttsStop() {
        speechService.stop()
        irodoriService.stop()
        irodoriCoreMLService.stop()
    }

    /// Release every TTS engine's weights.
    private func ttsUnloadAll() {
        speechService.unload()
        irodoriService.unload()
        irodoriCoreMLService.unload()
    }

    // MARK: - Model Activation

    private func activateVLM(modelId: String) async {
        // Free TTS weights before loading the (possibly large, e.g. QAT ~4.4 GB) VLM,
        // so the load peak doesn't collide with a resident TTS model → Jetsam.
        ttsStop()
        ttsUnloadAll()

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

    /// Switch the active TTS engine: stop all, persist selection, unload the others,
    /// load the chosen one. Only one TTS engine is resident at a time.
    private func activateTTS(modelId: String) async {
        ttsStop()
        guard let model = catalog.model(id: modelId) else { return }
        catalog.setActiveTTS(model)

        if IrodoriTTSService.modelIds.contains(modelId) {
            speechService.unload()
            irodoriCoreMLService.unload()
            irodoriService.select(modelId: modelId)
            await irodoriService.load()
        } else if modelId == IrodoriCoreMLService.catalogId {
            speechService.unload()
            irodoriService.unload()
            await loadIrodoriCoreML()
        } else {
            irodoriService.unload()
            irodoriCoreMLService.unload()
            await speechService.load()
        }
        await catalog.scanCache()
    }

    // MARK: - Capture & Describe

    private func captureAndDescribe(isRecapture: Bool = false) {
        guard let frame = cameraManager.latestFrame else { return }

        // Load model on first use (or wake from idle unload), then auto-capture.
        guard vlmService.isReady else {
            guard !vlmService.isDownloading else { return }
            Task {
                isIdle = false
                await vlmService.load()
                await loadActiveTTS()   // idle may have unloaded the voice too
                captureAndDescribe(isRecapture: isRecapture)
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
            if result.isEmpty { VLMService.log("describe returned empty text") }

            // D1A, when installed and on: re-capture a vague description once, and stay quiet when
            // it repeats the last spoken one. When unsure, speak — a wrong skip costs the listener
            // information, a wrong speak only a few seconds.
            if vlmService.d1aAvailable, !result.isEmpty, !result.hasPrefix("VLM error"),
               let p = await vlmService.decide(previous: lastSpokenDescription, new: result) {
                guard currentGen == generation else { return }
                let vague = p["vague"] ?? 0
                let redundant = lastSpokenDescription == nil ? 0 : (p["redundant"] ?? 0)
                let action = vague >= 0.6 && !isRecapture ? "recapture" : redundant >= 0.6 ? "skip" : "speak"
                VLMService.log(String(format: "d1a: redundant=%.2f vague=%.2f -> %@", redundant, vague, action))
                #if DEBUG
                VLMService.debugAppendDecision(previous: lastSpokenDescription, new: result,
                                               redundant: redundant, vague: vague, action: action)
                #endif
                if action == "recapture" {
                    try? await Task.sleep(for: .milliseconds(800))
                    guard currentGen == generation else { return }
                    captureAndDescribe(isRecapture: true)
                    return
                }
                if action == "skip" {
                    isThinking = false
                    descriptionText = result
                    resetIdleTimer()
                    return
                }
            }
            isThinking = false
            lastSpokenDescription = result

            if voiceEnabled {
                // Sync text with voice: show a "preparing voice" state, synthesize, and
                // reveal the text (typewriter) only once audio actually starts.
                //
                // Irodori is a heavy MLX model (~1.3 GB) whose flow-matching synthesis
                // spikes memory; keeping Gemma (~1.5 GB) resident through that spike
                // (plus the camera) overflows the budget → Jetsam. The text is already
                // produced, so free the brain before synthesizing; the next しらべる
                // lazily reloads it. Kokoro (CoreML/ANE) is light — leave Gemma in.
                // Irodori on Core ML (~1.4 GB of packages) gets the same treatment.
                if isIrodoriActive || isIrodoriCoreMLActive {
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

    // MARK: - Debug Launch Arguments

    #if DEBUG
    /// Remote device testing without screen taps. App arguments go after `--`:
    /// `devicectl … process launch John-Soliva.Mieru -- -voice irodori -irodoriVoice 2 -autoCapture 3`
    ///   -lang ja|en           response language
    ///   -vlm <hf model id>    switch the vision model first (e.g. the QAT repo)
    ///   -voice kokoro|irodori|irodori4|irodoricoreml  enable voice with that engine (irodori4 = v4.1 MeanFlow,
    ///                         irodoricoreml = the same model on Core ML)
    ///   -irodoriCoreMLUnits gpu|ane  Core ML Irodori encoder/duration/decoder units (read by IrodoriCoreMLService)
    ///   -d1aProbe ane|gpu|cpu|all [-d1aVision]  D1A Core ML trunk (or vision encoder) check, see D1AProbe
    ///   -d1aPhotos [-d1aPhotosMac]  D1A photo requests (photo_rows.json + jpgs in the D1A folder's photo/);
    ///                         -d1aPhotosMac feeds the Mac processor's pixels instead, -d1aPhotosPIL the
    ///                         app's PIL-exact resize, see D1AProbe
    ///   -removeDocuments a,b  delete those paths under Documents (side-loaded data no longer needed)
    ///   -d1aPlan [folder]     MLComputePlan device per op + timed predictions for Documents/d1a/<folder, plan>/*.mlpackage
    ///   -irodoriVoice <0-5>   pick an Irodori VoiceDesign caption
    ///   -autoCapture [N]      run N capture cycles (default 1), each waiting for the previous to finish
    ///   -voiceSeeds N [-voiceSeedVoice I]  WAVs of voice I (default 2) with seeds 1...N, to pick a speaker
    ///   -ttsBench coreml-gpu,coreml-ane,mlx [-benchRuns N] [-benchSustainMinutes M]  synthesis-only Irodori
    ///                         benchmark (camera + Gemma freed) → Documents/tts_bench.jsonl, see TTSBench
    private func runDebugLaunchArguments() async {
        guard !debugHarnessStarted else { return }
        debugHarnessStarted = true
        let args = ProcessInfo.processInfo.arguments
        func value(_ flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count,
                  !args[i + 1].hasPrefix("-") else { return nil }
            return args[i + 1]
        }

        if let lang = value("-lang") { language = lang }
        if let vlm = value("-vlm") { await activateVLM(modelId: vlm) }
        if let voice = value("-voice") {
            let ttsIds = ["irodori": IrodoriTTSService.v3Id, "irodori4": IrodoriTTSService.v41MeanFlowId,
                          "irodoricoreml": IrodoriCoreMLService.catalogId]
            await activateTTS(modelId: ttsIds[voice] ?? "aufklarer/Kokoro-82M-CoreML")
            voiceEnabled = true
        }
        if let v = value("-irodoriVoice"), let i = Int(v), IrodoriTTSService.voices.indices.contains(i) {
            irodoriService.selectedVoice = IrodoriTTSService.voices[i]   // also the Core ML engine's pick
        }
        if args.contains("-d1aFlow") {
            ttsUnloadAll()
            await D1AProbe.runFlow(vlm: vlmService)
            return
        }
        if args.contains("-d1aPhotos") {
            ttsUnloadAll()
            await D1AProbe.runPhotos(vlm: vlmService, macPixels: args.contains("-d1aPhotosMac"),
                                     pilResize: args.contains("-d1aPhotosPIL"))
            return
        }
        if args.contains("-d1aAdapter") {
            // D1A as an unmerged adapter on the resident Gemma: no second model to load.
            ttsUnloadAll()
            VLMService.log("d1a: adapter start fp=\(VLMService.footprintMB)MB")
            await D1AProbe.runAdapter(vlm: vlmService)
            return
        }
        if let paths = value("-removeDocuments") {
            // devicectl can copy into the container but not delete from it.
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            for path in paths.split(separator: ",").map(String.init) where !path.contains("..") {
                let removed = (try? FileManager.default.removeItem(at: docs.appendingPathComponent(path))) != nil
                VLMService.log("debug: remove Documents/\(path) \(removed ? "OK" : "skipped")")
            }
            return
        }
        if args.contains("-d1aPlan") {
            vlmService.unload()
            ttsUnloadAll()
            await D1AProbe.runPlan(folder: value("-d1aPlan") ?? "plan")
            return
        }
        if let units = value("-d1aProbe") {
            // D1A runs as a separate model: free Gemma and the voices first, as the app would.
            vlmService.unload()
            ttsUnloadAll()
            for _ in 0..<50 where VLMService.footprintMB > 1500 { try? await Task.sleep(for: .milliseconds(100)) }
            VLMService.log("d1a: start units=\(units) fp=\(VLMService.footprintMB)MB")
            let map: [String: MLComputeUnits] = ["ane": .cpuAndNeuralEngine, "gpu": .cpuAndGPU, "cpu": .cpuOnly, "all": .all]
            if args.contains("-d1aPhoto") {
                await D1AProbe.run(units: map[units] ?? .cpuAndNeuralEngine, label: units,
                                   feedsDir: "photo/feeds", baseline: "photo/mac_int4.json")
            } else if args.contains("-d1aVision") {
                await D1AProbe.runVision(units: map[units] ?? .cpuAndNeuralEngine, label: units)
            } else {
                await D1AProbe.run(units: map[units] ?? .cpuAndNeuralEngine, label: units)
            }
            return
        }
        if let count = value("-voiceSeeds").flatMap(Int.init) {
            // Speaker sampling for one voice: the same text with seeds 1...count (and the default seed) as
            // Documents/tts_dumps/voice<i>_seed<k>_*.wav, to pick a seed by ear for chosenSeeds.
            idleTimer?.invalidate()
            cameraManager.stop()
            vlmService.unload()
            ttsUnloadAll()
            let index = value("-voiceSeedVoice").flatMap(Int.init) ?? 2
            let caption = IrodoriTTSService.voices[index]
            irodoriService.selectedVoice = caption
            await irodoriCoreMLService.load()
            let text = "画像には、木製のテーブルの上にノートパソコンとコーヒーカップが置かれています。窓から柔らかい光が差し込んでいます。"
            let key = IrodoriTTSService.seedOverrideKey(caption)
            for seed in [0] + Array(1...count) {
                if seed == 0 { UserDefaults.standard.removeObject(forKey: key) } else { UserDefaults.standard.set(String(seed), forKey: key) }
                if let result = try? await irodoriCoreMLService.debugSynthesize(text) {
                    IrodoriTTSService.debugDumpWAV(samples: result.samples, rate: IrodoriCoreMLPipeline.sampleRate,
                                                   tag: "voice\(index)_seed\(seed == 0 ? "default" : String(seed))")
                }
            }
            UserDefaults.standard.removeObject(forKey: key)
            IrodoriTTSService.log("voiceSeeds DONE voice=\(index) count=\(count)")
            return
        }
        if let engines = value("-ttsBench") {
            // Synthesis only: free the camera, Gemma and the voices first, as for -d1aProbe, and keep the
            // idle unload from firing mid-run.
            idleTimer?.invalidate()
            cameraManager.stop()
            vlmService.unload()
            ttsUnloadAll()
            for _ in 0..<50 where VLMService.footprintMB > 1500 { try? await Task.sleep(for: .milliseconds(100)) }
            await TTSBench.run(engines: engines.split(separator: ",").map(String.init),
                               runs: Int(value("-benchRuns") ?? "") ?? 3,
                               sustainMinutes: Double(value("-benchSustainMinutes") ?? "") ?? 0,
                               coreML: irodoriCoreMLService, mlx: irodoriService)
            return
        }
        guard args.contains("-autoCapture") else { return }

        let cycles = Int(value("-autoCapture") ?? "") ?? 1
        VLMService.log("debug: \(cycles) capture cycle(s) lang=\(language) voice=\(voiceEnabled ? (speaksIrodoriMLX ? "irodori" : speaksIrodoriCoreML ? "irodoricoreml" : "kokoro") : "off") vlm=\(vlmService.currentModelId)")
        for cycle in 1...cycles {
            try? await Task.sleep(for: .seconds(3))
            VLMService.log("debug: cycle \(cycle)/\(cycles) start")
            descriptionText = ""
            captureAndDescribe()
            // Done = a new result is on screen and nothing is still thinking, synthesizing,
            // typing or speaking. (With Irodori, Gemma reloads first — text stays empty then.)
            // Require two consecutive idle polls so a result that lands just before the
            // typewriter flips `isTyping` isn't mistaken for a finished cycle.
            var idlePolls = 0
            for poll in 0..<480 {
                try? await Task.sleep(for: .milliseconds(500))
                let busy = isThinking || isSynthesizing || isTyping || vlmService.isDownloading
                    || speechService.isSpeaking || irodoriService.isSpeaking || irodoriCoreMLService.isSpeaking
                idlePolls = (!descriptionText.isEmpty && !busy) ? idlePolls + 1 : 0
                if idlePolls >= 2 { break }
                // A Gemma reload doesn't set isThinking, so only warn once the model is in.
                if poll == 40 && descriptionText.isEmpty && !busy && vlmService.isLoaded {
                    VLMService.log("debug: cycle \(cycle) produced nothing after 20s (no frame or still loading?)")
                }
            }
        }
        VLMService.log("debug: capture cycles done")
    }
    #endif

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
        guard !isThinking, !isSynthesizing, !isTyping, !speechService.isSpeaking,
              !irodoriService.isSpeaking, !irodoriCoreMLService.isSpeaking else {
            resetIdleTimer()
            return
        }
        guard vlmService.isLoaded || speechService.isLoaded || irodoriService.isLoaded
            || irodoriCoreMLService.isLoaded else { return }

        vlmService.unload()
        ttsUnloadAll()
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
            Task { @MainActor in
                if vlmService.isReady { captureAndDescribe() }
            }
        }
    }

    private func stopAutoMode() {
        autoTimer?.invalidate()
        autoTimer = nil
    }

    // MARK: - Lifecycle

    private func handleBackground() {
        VLMService.log("lifecycle: background — models released")
        UIApplication.shared.isIdleTimerDisabled = false
        idleTimer?.invalidate()
        stopAutoMode()
        isAutoMode = false
        cameraManager.stop()
        vlmService.unload()
        ttsStop()
    }

    private func handleForeground() {
        VLMService.log("lifecycle: foreground")
        // Re-assert keep-awake — it was cleared on background.
        UIApplication.shared.isIdleTimerDisabled = true
        cameraManager.start()
        // Model reload triggers via onChange(cameraManager.isRunning)
    }
}

#Preview {
    ContentView()
}
