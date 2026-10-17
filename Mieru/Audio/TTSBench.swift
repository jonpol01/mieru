//
//  TTSBench.swift
//  Mieru
//
//  DEBUG heavy TTS benchmark (-ttsBench, see ContentView.runDebugLaunchArguments): synthesis only, no
//  playback, of a fixed Japanese corpus on each Irodori engine in turn. One JSON object per line in
//  Documents/tts_bench.jsonl ("type": start | load | utt | sustain | unload | done), mirrored to vlm.log.
//  The caller frees the camera, Gemma and every voice first.
//

#if DEBUG
import Foundation

struct TTSBenchError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

@MainActor
enum TTSBench {

    /// Short, Gemma-style camera descriptions, medium, long (> 128 tokens / 15 s: several Core ML segments),
    /// numbers / dates / units, English brand names, full- and half-width text, emoji / symbols, 「」 quotes.
    static let corpus: [(id: String, text: String)] = [
        ("short_hai", "はい。"),
        ("short_thanks", "ありがとうございます。"),
        ("short_hello", "こんにちは。"),
        ("cam_laptop", "画像には、机の上にノートパソコンとコーヒーカップがあります。"),
        ("cam_plant", "白い壁の前に観葉植物が置かれています。窓から柔らかい光が差し込んでいます。"),
        ("cam_bicycle", "道路の脇に赤い自転車が停まっています。後ろには青い看板のコンビニが見えます。"),
        ("cam_curry", "テーブルの上に、お皿に盛られたカレーライスとスプーンがあります。湯気が立っていて、できたてのようです。"),
        ("cam_crossing", "人々が横断歩道を渡っています。信号は青で、背景には高いビルが並んでいます。夕方の時間帯のようです。"),
        ("cam_cat", "ソファの上で、黒い猫が丸くなって眠っています。"),
        ("cam_kitchen", "キッチンのカウンターに、包丁とまな板、切りかけのトマトがあります。"),
        ("cam_books", "本棚には、色とりどりの本がぎっしりと並んでいます。一番上の段には小さな地球儀も置かれています。"),
        ("medium_park", "画像には、晴れた日の公園の様子が写っています。手前には木製のベンチがあり、その上に赤いリュックサックが置かれています。奥では子どもたちがボール遊びをしていて、さらに向こうには大きな桜の木が見えます。空は青く、雲はほとんどありません。"),
        ("medium_station", "駅のホームの写真です。電光掲示板には次の電車の行き先と発車時刻が表示されています。ホームの端には黄色い点字ブロックが続いていて、何人かの人がスマートフォンを見ながら電車を待っています。屋根の隙間から、夕焼けの空が少しだけ見えています。"),
        ("long_room", "これはリビングルームの写真です。部屋の中央には灰色の大きなソファがあり、その上にはクッションが三つ並んでいます。ソファの前には木製の低いテーブルがあって、テーブルの上には雑誌が二冊と、半分ほど水の入ったガラスのコップ、そしてテレビのリモコンが置かれています。左側の壁には大きな窓があり、白いレースのカーテン越しに午後の光が差し込んでいます。右側の壁際には背の高い本棚があり、本のほかに小さな観葉植物や写真立ても飾られています。床にはベージュのラグが敷かれていて、その端で茶色い犬が気持ちよさそうに眠っています。全体的に明るく、落ち着いた雰囲気の部屋です。"),
        ("long_street", "夜の商店街の様子です。通りの両側には小さなお店が並んでいて、どのお店にも明るい看板が灯っています。手前のラーメン屋の前では赤いのれんが風に揺れていて、数人のお客さんが列を作って順番を待っています。その隣の八百屋では、店先にリンゴやミカン、キャベツなどが並べられています。通りの真ん中を自転車に乗った人がゆっくりと通り過ぎていき、遠くには駅の明かりがぼんやりと見えています。雨上がりなのか路面が少し濡れていて、看板の光が反射してきらきらと輝いています。"),
        // One sentence over 128 tokens: Core ML falls back to splitting it at 、.
        ("long_one_sentence", "机の上には、開いたままのノートパソコン、飲みかけのコーヒーが入った白いマグカップ、黒いボールペンが三本、付箋がたくさん貼られた分厚い手帳、充電ケーブルにつながったスマートフォン、小さなサボテンの鉢植え、読みかけの文庫本、角が少し折れた会議の資料、銀色の腕時計、そして食べかけのチョコレートの箱が、少し散らかった様子で置かれていて、持ち主が仕事の途中で急いで席を外したような、どこか慌ただしい雰囲気が伝わってきます。"),
        ("num_date", "会議は2026年10月17日の午後3時45分から、3階の会議室Bで始まります。"),
        ("num_units", "現在の気温は23.5℃、湿度は60%で、風速は毎秒4mです。"),
        ("num_price", "このコーヒーは1杯480円で、容量は350mlです。"),
        ("brand_iphone", "iPhone 17 Proで撮った写真を、MacBookで編集しています。"),
        ("brand_mix", "画面にはApple WatchとAirPodsが写っていて、横にはUSB-Cケーブルがあります。"),
        ("width_mix", "ＡＢＣ１２３とABC123、ｶﾀｶﾅとカタカナを読み比べます。"),
        ("emoji", "今日はいい天気ですね☀️ 散歩に行きましょう♪ 楽しみです😊。"),
        ("quote", "看板には「本日休業」と書かれています。店員さんは「また明日来てください」と言いました。"),
    ]

    private struct Synth {
        let samples: [Float]
        let rate: Double
        let ttfa: Double?          // nil: whole synthesis (MLX does not stream)
        let tokens: Int?
        let segments: Int
        let phases: [String: Double]
    }

    private struct Result { let ttfa, synth, rtf: Double }

    private static func now() -> Double { CFAbsoluteTimeGetCurrent() }

    static func run(engines: [String], runs: Int, sustainMinutes: Double,
                    coreML: IrodoriCoreMLService, mlx: IrodoriTTSService) async {
        let savedUnits = coreML.units
        defer { coreML.units = savedUnits }   // the units setter persists; leave the user's choice as it was
        log(["type": "start", "engines": engines, "runs": runs, "sustain_min": sustainMinutes,
             "items": corpus.count, "voice": mlx.selectedVoice ?? IrodoriTTSService.defaultVoice])

        var summary: [String: Any] = [:]
        for engine in engines {
            switch engine {
            case "coreml-gpu": coreML.units = .gpu
            case "coreml-ane": coreML.units = .ane   // encoder, duration, decoder; the DiT stays on the GPU
            case "mlx": mlx.select(modelId: IrodoriTTSService.v41MeanFlowId)
            default:
                log(["type": "load", "engine": engine, "ok": false, "error": "unknown engine"])
                summary[engine] = ["error": "unknown engine"]
                continue
            }
            let baseline = VLMService.footprintMB
            // Cold load, unload, then warm load (OS file cache and compiled Core ML plans hot).
            if let error = await timedLoad(engine, kind: "cold", coreML: coreML, mlx: mlx) {
                summary[engine] = ["error": "load: \(error)"]
                continue
            }
            unload(engine, coreML: coreML, mlx: mlx)
            _ = await settle(above: baseline)
            if let error = await timedLoad(engine, kind: "warm", coreML: coreML, mlx: mlx) {
                summary[engine] = ["error": "load: \(error)"]
                continue
            }

            var results: [Result] = []
            var errors = 0
            for run in 0..<runs {
                for item in corpus {
                    if let r = await utterance(engine, phase: "corpus", run: run, item: item, coreML: coreML, mlx: mlx) {
                        results.append(r)
                    } else {
                        errors += 1
                    }
                }
            }

            if sustainMinutes > 0 {
                // Loop the corpus to catch thermal throttling: one progress line per minute.
                let start = now()
                var minuteStart = start, minute = 0, i = 0, rtfs: [Double] = [], minuteErrors = 0
                while now() - start < sustainMinutes * 60 {
                    let r = await utterance(engine, phase: "sustain", run: i / corpus.count,
                                            item: corpus[i % corpus.count], coreML: coreML, mlx: mlx)
                    i += 1
                    if let r { rtfs.append(r.rtf) } else { minuteErrors += 1 }
                    if now() - minuteStart >= 60 || now() - start >= sustainMinutes * 60 {
                        minute += 1
                        log(["type": "sustain", "engine": engine, "minute": minute, "utterances": rtfs.count,
                             "errors": minuteErrors, "mean_rtf": r3(rtfs.isEmpty ? 0 : rtfs.reduce(0, +) / Double(rtfs.count)),
                             "fp_mb": VLMService.footprintMB, "fp_peak_mb": VLMService.peakFootprintMB,
                             "thermal": thermal, "ts": timestamp])
                        minuteStart = now()
                        rtfs = []
                        minuteErrors = 0
                    }
                }
            }

            let loaded = VLMService.footprintMB
            unload(engine, coreML: coreML, mlx: mlx)
            let after = await settle(above: baseline)
            log(["type": "unload", "engine": engine, "fp_baseline_mb": baseline, "fp_loaded_mb": loaded,
                 "fp_after_mb": after, "released": after <= baseline + 150, "ts": timestamp])
            summary[engine] = stats(results, errors: errors)
        }
        log(["type": "done", "engines": summary, "ts": timestamp], text: "bench DONE")
    }

    // MARK: - Engines

    /// nil on success, else the load error.
    private static func timedLoad(_ engine: String, kind: String,
                                  coreML: IrodoriCoreMLService, mlx: IrodoriTTSService) async -> String? {
        let before = VLMService.footprintMB
        let t0 = now()
        var error: String?
        if engine == "mlx" {
            await mlx.load()
            if !mlx.isLoaded { error = mlx.statusMessage }
        } else {
            await coreML.load()
            if !coreML.isLoaded { error = coreML.statusMessage }
        }
        log(["type": "load", "engine": engine, "kind": kind, "ok": error == nil, "seconds": r3(now() - t0),
             "units": engine == "mlx" ? "mlx" : coreML.units.rawValue,
             "fp_before_mb": before, "fp_after_mb": VLMService.footprintMB, "fp_peak_mb": VLMService.peakFootprintMB,
             "thermal": thermal, "error": orNull(error), "ts": timestamp])
        return error
    }

    private static func unload(_ engine: String, coreML: IrodoriCoreMLService, mlx: IrodoriTTSService) {
        if engine == "mlx" { mlx.unload() } else { coreML.unload() }
    }

    /// Waits up to 5 s for the footprint to fall back near `baseline`; returns the footprint.
    private static func settle(above baseline: Int) async -> Int {
        for _ in 0..<50 where VLMService.footprintMB > baseline + 150 { try? await Task.sleep(for: .milliseconds(100)) }
        return VLMService.footprintMB
    }

    private static func synthesize(_ engine: String, _ text: String,
                                   coreML: IrodoriCoreMLService, mlx: IrodoriTTSService) async throws -> Synth {
        if engine == "mlx" {
            let (samples, rate, phases) = try await mlx.debugSynthesize(text)
            return Synth(samples: samples, rate: rate, ttfa: nil, tokens: nil, segments: 1, phases: phases)
        }
        let (samples, t) = try await coreML.debugSynthesize(text)
        return Synth(samples: samples, rate: IrodoriCoreMLPipeline.sampleRate, ttfa: t.firstAudio, tokens: t.tokens,
                     segments: t.segments,
                     phases: ["encode": t.encode, "duration": t.duration, "sampler": t.sampler, "decode": t.decode,
                              "decode_first": t.decodeFirst, "frames": Double(t.frames)])
    }

    /// One utterance, logged as a "utt" line. nil when it threw.
    private static func utterance(_ engine: String, phase: String, run: Int, item: (id: String, text: String),
                                  coreML: IrodoriCoreMLService, mlx: IrodoriTTSService) async -> Result? {
        var line: [String: Any] = ["type": "utt", "engine": engine, "phase": phase, "run": run, "item": item.id,
                                   "chars": item.text.count, "fp_before_mb": VLMService.footprintMB]
        var result: Result?
        let t0 = now()
        do {
            let s = try await synthesize(engine, item.text, coreML: coreML, mlx: mlx)
            let synth = now() - t0
            let audio = Double(s.samples.count) / s.rate
            let ttfa = s.ttfa ?? synth
            result = Result(ttfa: ttfa, synth: synth, rtf: synth > 0 ? audio / synth : 0)
            line.merge(["tokens": orNull(s.tokens), "segments": s.segments, "audio_s": r3(audio),
                        "ttfa_s": r3(ttfa), "synth_s": r3(synth), "rtf": r3(result!.rtf),
                        "phases": s.phases.mapValues(r3), "error": NSNull()]) { $1 }
            // WAV dumps for the first corpus run only, so the device does not fill up.
            if phase == "corpus" && run == 0 {
                IrodoriTTSService.debugDumpWAV(samples: s.samples, rate: s.rate, tag: "bench_\(engine)_\(item.id)")
            }
        } catch {
            line["error"] = String(describing: error)
        }
        line.merge(["fp_after_mb": VLMService.footprintMB, "fp_peak_mb": VLMService.peakFootprintMB,
                    "thermal": thermal, "ts": timestamp]) { $1 }
        log(line)
        try? await Task.sleep(for: .milliseconds(300))   // a short pause between utterances
        return result
    }

    // MARK: - Stats and logging

    private static func stats(_ results: [Result], errors: Int) -> [String: Any] {
        func pct(_ values: [Double], _ p: Double) -> Double {
            guard !values.isEmpty else { return 0 }
            let sorted = values.sorted()
            return r3(sorted[Int((Double(sorted.count - 1) * p).rounded())])
        }
        let ttfa = results.map(\.ttfa), synth = results.map(\.synth), rtf = results.map(\.rtf)
        return ["n": results.count, "errors": errors,
                "ttfa_p50": pct(ttfa, 0.5), "ttfa_p95": pct(ttfa, 0.95),
                "synth_p50": pct(synth, 0.5), "synth_p95": pct(synth, 0.95),
                "rtf_p50": pct(rtf, 0.5), "rtf_p95": pct(rtf, 0.95)]
    }

    private static func r3(_ x: Double) -> Double { (x * 1000).rounded() / 1000 }

    private static func orNull(_ value: Any?) -> Any { value ?? NSNull() }

    private static var timestamp: String { ISO8601DateFormatter().string(from: Date()) }

    private static var thermal: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// Appends one JSON line to Documents/tts_bench.jsonl (excluded from backup) and mirrors it to vlm.log.
    private static func log(_ object: [String: Any], text: String? = nil) {
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
              var line = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        else { return }
        VLMService.log("ttsbench: " + (text.map { $0 + " " } ?? "") + String(decoding: line, as: UTF8.self))
        line.append(0x0A)
        let url = docs.appendingPathComponent("tts_bench.jsonl")
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(line); try? h.close()
        } else {
            try? line.write(to: url)
        }
        VLMService.debugExcludeFromBackup(url)
    }
}
#endif
