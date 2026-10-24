//
//  D1AProbe.swift
//  Mieru
//
//  DEBUG proof of concept: runs the D1A-E2B-it Core ML int4 trunk (5 chunks, L=512) on
//  precomputed host-side feeds and compares its answers with the same build on the Mac.
//  Data is pushed with devicectl into Documents/d1a/ (compiled/, feeds/, head_*.bin, mac_int4.json).
//

#if DEBUG
import Accelerate
import CoreML
import Foundation
import MLX
import MLXLMCommon
@_spi(GemmaEncoder) import MLXVLM

enum D1AProbe {

    private static let chunks = ["L512_00-07", "L512_07-15", "L512_15-22", "L512_22-29", "L512_29-35"]

    private static func log(_ line: String) { VLMService.log("d1a: " + line) }

    private static func now() -> Double { CFAbsoluteTimeGetCurrent() }

    private static func array(_ url: URL, _ shape: [Int], _ type: MLMultiArrayDataType) throws -> MLMultiArray {
        let data = try Data(contentsOf: url)
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: type)
        precondition(data.count == a.count * (type == .int32 ? 4 : 2), "\(url.lastPathComponent): size mismatch")
        data.withUnsafeBytes { a.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: data.count) }
        return a
    }

    private static func floats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// Describe-flow smoke test: the cases in describe_flow.json through the in-app decide path.
    static func runFlow(vlm: VLMService) async {
        let url = VLMService.d1aDirectory.appendingPathComponent("describe_flow.json")
        guard let data = try? Data(contentsOf: url),
              let flow = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let cases = flow["cases"] as? [[String: Any]] else { log("flow: no describe_flow.json"); return }
        for c in cases {
            let previous = (c["previous"] as? String).flatMap { $0.isEmpty || $0 == "(none)" ? nil : $0 }
            let t = now()
            guard let p = await vlm.decide(previous: previous, new: c["new"] as! String) else {
                log("flow \(c["name"] ?? "?"): decide returned nil"); continue
            }
            log(String(format: "flow %@: redundant=%.2f vague=%.2f (%.2fs) mac=%@", c["name"] as? String ?? "?",
                       p["redundant"] ?? -1, p["vague"] ?? -1, now() - t, String(describing: c["mac_mlx_q8_p"] ?? "")))
        }
        log("flow DONE")
    }

    /// MLX adapter check on the resident Gemma 4 VLM: Documents/d1a/adapter/{adapters.safetensors,
    /// adapter_config.json, head.safetensors, d1a_config.json, layout.json, requests.jsonl, gate.json}.
    /// Each question runs as one causal row (<bos> <state> state <q> branch <decide>), which equals
    /// D1A's packed block-causal encoding for that question.
    static func runAdapter(vlm: VLMService) async {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("d1a/adapter")
        do {
            let summary = try await vlm.debugPerform { context -> String in
                guard let gemma = context.model as? Gemma4 else { return "model is not MLXVLM.Gemma4" }
                func json(_ name: String) throws -> [String: Any] {
                    try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent(name))) as! [String: Any]
                }
                let layout = try json("layout.json")
                let leading = layout["leading_ids"] as! [Int]
                let escape = try NSRegularExpression(pattern: layout["escape_pattern"] as! String)
                let special = try NSRegularExpression(pattern: #"<\|([A-Za-z0-9_]+)\|>"#)
                func userTokens(_ text: String) -> [Int] {
                    let ns = NSMutableString(string: text)
                    for m in escape.matches(in: text, range: NSRange(location: 0, length: ns.length)).reversed() {
                        ns.insert("¦", at: m.range.location + 1)
                    }
                    let t = special.stringByReplacingMatches(in: ns as String, range: NSRange(location: 0, length: ns.length), withTemplate: "<¦$1¦>")
                    return context.tokenizer.encode(text: t, addSpecialTokens: false)
                }
                let temperature = Float((try json("d1a_config.json"))["temperature"] as! Double)
                let head = try MLX.loadArrays(url: dir.appendingPathComponent("head.safetensors")).mapValues { $0.asType(.float32) }
                let wq = head["q.weight"]!, bq = head["q.bias"]!, wk = head["k.weight"]!, bk = head["k.bias"]!
                let scale = 1 / sqrtf(Float(wq.dim(0))) / temperature

                var gate: [Int: [Int: [Double]]] = [:]
                for row in (try json("gate.json"))["rows"] as! [[String: Any]] {
                    gate[row["request"] as! Int, default: [:]][row["q"] as! Int] = row["p"] as? [Double]
                }
                let requests = try String(contentsOf: dir.appendingPathComponent("requests.jsonl"), encoding: .utf8)
                    .split(separator: "\n").map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }

                // Rows per question: ids, the <decide> index and the </opt> indices.
                func rows(_ request: [String: Any]) -> [(ids: [Int], decide: Int, opts: [Int])] {
                    var state = leading, segments: [[Int]] = [], opts: [[Int]] = []
                    for piece in request["pieces"] as! [[String: Any]] {
                        if let d = piece["delimiter"] as? Int {
                            if d == 7 { segments.append([]); opts.append([]) }
                            if segments.isEmpty { state.append(d) } else {
                                if d == 9 { opts[opts.count - 1].append(segments[segments.count - 1].count) }
                                segments[segments.count - 1].append(d)
                            }
                        } else {
                            let ids = userTokens(piece["text"] as! String)
                            if segments.isEmpty { state += ids } else { segments[segments.count - 1] += ids }
                        }
                    }
                    return segments.indices.map { k in
                        (state + segments[k], state.count + segments[k].count - 1, opts[k].map { state.count + $0 })
                    }
                }

                // Adapter off must leave Gemma untouched: compare one row's hidden states before and after.
                let probeIds = MLXArray(rows(requests[0])[0].ids.map { Int32($0) }).expandedDimensions(axis: 0)
                let before = gemma.textHiddenStates(probeIds).asType(.float32)
                eval(before)

                let lora = try LoRAContainer.from(directory: dir)
                var t = now()
                try lora.load(into: gemma)
                let loadSeconds = now() - t

                var times: [Double] = [], maxDp = 0.0, flips = 0, clearFlips = 0, questions = 0
                for (ri, request) in requests.enumerated() {
                    t = now()
                    for (qi, row) in rows(request).enumerated() {
                        let h = gemma.textHiddenStates(MLXArray(row.ids.map { Int32($0) }).expandedDimensions(axis: 0))[0].asType(.float32)
                        let hq = matmul(h[row.decide], wq.T) + bq
                        let hk = matmul(h[MLXArray(row.opts.map { Int32($0) })], wk.T) + bk
                        let p = softmax(matmul(hk, hq) * scale, axis: -1)
                        eval(p)
                        let probs = p.asArray(Float.self).map { Double($0) }
                        questions += 1
                        guard let ref = gate[ri]?[qi] else { continue }
                        maxDp = max(maxDp, zip(probs, ref).map { abs($0 - $1) }.max() ?? 0)
                        let top = probs.indices.max { probs[$0] < probs[$1] }!, refTop = ref.indices.max { ref[$0] < ref[$1] }!
                        if top != refTop {
                            flips += 1
                            let sorted = ref.sorted(by: >)
                            if sorted[0] - sorted[1] > 0.05 { clearFlips += 1 }
                        }
                    }
                    times.append(now() - t)
                }

                lora.unload(from: gemma)
                let after = gemma.textHiddenStates(probeIds).asType(.float32)
                let offDiff = abs(after - before).max().item(Float.self)

                let s = times.sorted()
                return String(format: "adapter DONE: load %.2fs | %d requests, %d questions | p50 %.3fs p95 %.3fs first %.3fs | vs Mac adapter: %d top changes (%d clear-cut), max|dp| %.3f | adapter off max|dh| %g | fp=%dMB peak=%dMB",
                              loadSeconds, times.count, questions, s[s.count / 2], s[min(s.count - 1, s.count * 95 / 100)],
                              times.first ?? 0, flips, clearFlips, maxDp, offDiff, VLMService.footprintMB, VLMService.peakFootprintMB)
            }
            log(summary)
        } catch {
            log("adapter FAILED: \(String(describing: error))")
        }
    }

    /// Vision encoder check: Documents/d1a/vision/{VisionEncoder.mlmodelc, pixels_i.bin, embeds_i.bin}
    /// against the transformers fp32 reference. Gate: min per-row cosine >= 0.98.
    static func runVision(units: MLComputeUnits, label: String) async {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("d1a/vision")
        do {
            let config = MLModelConfiguration()
            config.computeUnits = units
            let t = now()
            let model = try await MLModel.load(contentsOf: dir.appendingPathComponent("VisionEncoder.mlmodelc"), configuration: config)
            log(String(format: "vision load (%@) %.1fs fp=%dMB", label, now() - t, VLMService.footprintMB))
            var times: [Double] = []
            for i in 0..<3 {
                let pixels = try MLMultiArray(shape: [1, 3, 672, 912], dataType: .float32)
                let pdata = try Data(contentsOf: dir.appendingPathComponent("pixels_\(i).bin"))
                pdata.withUnsafeBytes { pixels.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: pdata.count) }
                let input = try MLDictionaryFeatureProvider(dictionary: ["pixels": MLFeatureValue(multiArray: pixels)])
                var out: MLMultiArray?
                for pass in 0..<2 {   // the second pass is the warm timing
                    let tp = now()
                    out = try await model.prediction(from: input).featureValue(for: "image_embeds")?.multiArrayValue
                    if pass == 1 { times.append(now() - tp) }
                }
                guard let out else { log("vision \(i): no output"); continue }
                let ref = try floats(dir.appendingPathComponent("embeds_\(i).bin"))
                let rows = out.shape[0].intValue, cols = out.shape[1].intValue, rs = out.strides[0].intValue
                let o = out.dataPointer.bindMemory(to: Float.self, capacity: out.count)
                var minCos: Float = 1, maxDiff: Float = 0
                for r in 0..<rows {
                    var dot: Float = 0, na: Float = 0, nb: Float = 0
                    for c in 0..<cols {
                        let a = o[r * rs + c], b = ref[r * cols + c]
                        dot += a * b; na += a * a; nb += b * b
                        maxDiff = max(maxDiff, abs(a - b))
                    }
                    minCos = min(minCos, dot / (sqrtf(na) * sqrtf(nb) + 1e-12))
                }
                log(String(format: "vision %d (%@): %.0f ms warm, min cos %.4f, max|d| %.3f %@", i, label,
                           times.last! * 1000, minCos, maxDiff, minCos >= 0.98 ? "PASS" : "FAIL"))
            }
            let s = times.sorted()
            log(String(format: "vision DONE (%@): median %.0f ms/image, fp=%dMB peak=%dMB",
                       label, s[s.count / 2] * 1000, VLMService.footprintMB, VLMService.peakFootprintMB))
        } catch {
            log("vision FAILED (\(label)): \(String(describing: error))")
        }
    }

    /// Weight-format placement check: every Documents/d1a/<folder>/*.mlpackage is compiled, then for
    /// CPU_AND_NE and ALL logs each op's preferred / supported device from MLComputePlan and runs the
    /// first L rows of x.bin ([1, 64, 1536] fp16, L from the model's input "x"), timed over 20 warm calls.
    /// Outputs go to out_<model>_<units>.f32 next to them.
    static func runPlan(folder: String) async {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("d1a/\(folder)")
        func name(_ d: MLComputeDevice) -> String {
            switch d {
            case .cpu: return "cpu"
            case .gpu: return "gpu"
            case .neuralEngine: return "ane"
            @unknown default: return "?"
            }
        }
        do {
            let x = try array(dir.appendingPathComponent("x.bin"), [1, 64, 1536], .float16)
            let packages = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "mlpackage" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
            for pkg in packages {
                let model = pkg.deletingPathExtension().lastPathComponent
                let compiled: URL
                do { compiled = try await MLModel.compileModel(at: pkg) } catch {
                    log("plan \(model): compile FAILED \(error)"); continue
                }
                for (label, units) in [("cpu_and_ne", MLComputeUnits.cpuAndNeuralEngine), ("all", .all)] {
                    let config = MLModelConfiguration()
                    config.computeUnits = units
                    do {
                        let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: config)
                        guard case let .program(program) = plan.modelStructure,
                              let main = program.functions["main"] else { log("plan \(model) \(label): not a program"); continue }
                        for op in main.block.operations {
                            guard let usage = plan.deviceUsage(for: op) else { continue }
                            log("plan \(model) \(label): \(op.operatorName) preferred=\(name(usage.preferred)) supported=\(usage.supported.map(name).joined(separator: ","))")
                        }
                        let t = now()
                        let m = try await MLModel.load(contentsOf: compiled, configuration: config)
                        let tl = now() - t
                        let shape = m.modelDescription.inputDescriptionsByName["x"]!.multiArrayConstraint!.shape
                        let xm = try MLMultiArray(shape: shape, dataType: .float16)
                        xm.dataPointer.copyMemory(from: x.dataPointer, byteCount: xm.count * 2)   // first L rows
                        let input = try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: xm)])
                        let outName = m.modelDescription.outputDescriptionsByName.keys.first!
                        _ = try await m.prediction(from: input)   // warm-up
                        var out: MLMultiArray!
                        let tp = now()
                        for _ in 0..<20 { out = try await m.prediction(from: input).featureValue(for: outName)!.multiArrayValue! }
                        let tpred = (now() - tp) / 20
                        var f = [Float](repeating: 0, count: out.count)
                        if out.dataType == .float16 {
                            let p = out.dataPointer.bindMemory(to: Float16.self, capacity: out.count)
                            for i in 0..<out.count { f[i] = Float(p[i]) }
                        } else {
                            let p = out.dataPointer.bindMemory(to: Float.self, capacity: out.count)
                            for i in 0..<out.count { f[i] = p[i] }
                        }
                        try f.withUnsafeBufferPointer { Data(buffer: $0) }
                            .write(to: dir.appendingPathComponent("out_\(model)_\(label).f32"))
                        log(String(format: "plan %@ %@: load %.2fs, predict OK %.3f ms (mean of 20), out %@ %@ first=%.4f",
                                   model, label, tl, tpred * 1000, outName, out.shape.description, f[0]))
                    } catch {
                        log("plan \(model) \(label): FAILED \(error)")
                    }
                }
            }
            log("plan DONE")
        } catch {
            log("plan FAILED: \(error)")
        }
    }

    /// `feedsDir` holds text feeds, or photo feeds (pixels.bin + image_rows.bin) that first go
    /// through the vision encoder. Answers are compared with `baseline` (the Mac run of the same
    /// build) when present, otherwise with each feed's fp32 `ref`.
    static func run(units: MLComputeUnits, label: String, feedsDir: String = "feeds", baseline: String = "mac_int4.json") async {
        let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("d1a")
        do {
            // Load all five chunks into one process (the Mac reference needed one process per chunk).
            let config = MLModelConfiguration()
            config.computeUnits = units
            var models: [MLModel] = []
            let tLoad = now()
            for name in chunks {
                let t = now()
                models.append(try await MLModel.load(
                    contentsOf: root.appendingPathComponent("compiled/\(name).mlmodelc"), configuration: config))
                log(String(format: "load %@ %.1fs fp=%dMB", name, now() - t, VLMService.footprintMB))
            }
            log(String(format: "loaded 5 chunks (%@) in %.1fs, fp=%dMB peak=%dMB",
                       label, now() - tLoad, VLMService.footprintMB, VLMService.peakFootprintMB))

            // Pointer head: z = (Wk h_opt + bk) . (Wq h_decide + bq) / sqrt(dp), softmax(z / T).
            let headCfg = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("head.json"))) as! [String: Any]
            let dp = headCfg["dp"] as! Int, hid = headCfg["hidden"] as! Int
            let temperature = Float(headCfg["temperature"] as! Double)
            let wq = try floats(root.appendingPathComponent("head_q_weight.bin")), bq = try floats(root.appendingPathComponent("head_q_bias.bin"))
            let wk = try floats(root.appendingPathComponent("head_k_weight.bin")), bk = try floats(root.appendingPathComponent("head_k_bias.bin"))
            func project(_ w: [Float], _ b: [Float], _ h: [Float]) -> [Float] {
                var y = b
                cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(dp), Int32(hid), 1, w, Int32(hid), h, 1, 1, &y, 1)
                return y
            }

            // Mac reference: feed -> question -> p.
            var macP: [String: [Int: [Double]]] = [:]
            if let data = try? Data(contentsOf: root.appendingPathComponent(baseline)),
               let mac = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                for row in mac["rows"] as! [[String: Any]] {
                    macP[(row["feed"] as! String).replacingOccurrences(of: ".npz", with: ""), default: [:]][row["q"] as! Int] = row["p"] as? [Double]
                }
            }
            let reference = macP.isEmpty ? "fp32 ref" : "Mac int4"
            var encoder: MLModel?

            let feedDir = root.appendingPathComponent(feedsDir)
            let feeds = try FileManager.default.contentsOfDirectory(atPath: feedDir.path).sorted()
            var times: [Double] = [], maxDp = 0.0, flips = 0, clearFlips = 0, questions = 0
            for (fi, feed) in feeds.enumerated() {
                let d = feedDir.appendingPathComponent(feed)
                let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: d.appendingPathComponent("meta.json"))) as! [String: Any]
                let common: [String: MLFeatureValue] = [
                    "pos": MLFeatureValue(multiArray: try array(d.appendingPathComponent("pos.bin"), [1, 512], .int32)),
                    "mask_full": MLFeatureValue(multiArray: try array(d.appendingPathComponent("mask_full.bin"), [1, 1, 512, 512], .float16)),
                    "mask_slide": MLFeatureValue(multiArray: try array(d.appendingPathComponent("mask_slide.bin"), [1, 1, 512, 512], .float16)),
                ]
                let emb = try array(d.appendingPathComponent("emb.bin"), [1, 512, 1536], .float16)
                var encodeSeconds = 0.0
                if FileManager.default.fileExists(atPath: d.appendingPathComponent("pixels.bin").path) {
                    // Photo feed: the encoder output replaces the image-token rows of emb.
                    if encoder == nil {
                        encoder = try await MLModel.load(
                            contentsOf: root.appendingPathComponent("vision/VisionEncoder.mlmodelc"), configuration: config)
                    }
                    let pixels = try MLMultiArray(shape: [1, 3, 672, 912], dataType: .float32)
                    let pdata = try Data(contentsOf: d.appendingPathComponent("pixels.bin"))
                    pdata.withUnsafeBytes { pixels.dataPointer.copyMemory(from: $0.baseAddress!, byteCount: pdata.count) }
                    let te = now()
                    let image = try await encoder!.prediction(from: MLDictionaryFeatureProvider(
                        dictionary: ["pixels": MLFeatureValue(multiArray: pixels)])).featureValue(for: "image_embeds")!.multiArrayValue!
                    encodeSeconds = now() - te
                    let rowsIdx = try Data(contentsOf: d.appendingPathComponent("image_rows.bin")).withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
                    let src = image.dataPointer.bindMemory(to: Float.self, capacity: image.count), srcStride = image.strides[0].intValue
                    let dst = emb.dataPointer.bindMemory(to: Float16.self, capacity: emb.count), dstStride = emb.strides[1].intValue
                    for (r, row) in rowsIdx.enumerated() {
                        for c in 0..<1536 { dst[Int(row) * dstStride + c] = Float16(src[r * srcStride + c]) }
                    }
                }
                var state: [String: MLFeatureValue] = [
                    "emb": MLFeatureValue(multiArray: emb),
                    "ple_tok": MLFeatureValue(multiArray: try array(d.appendingPathComponent("ple_tok.bin"), [1, 512, 35, 256], .float16)),
                    "picks": MLFeatureValue(multiArray: try array(d.appendingPathComponent("picks.bin"), [128], .int32)),
                ]

                let t = now()
                var hidden: MLMultiArray?
                for (ci, model) in models.enumerated() {
                    if fi < 2 { log("\(feed) chunk \(ci) start") }   // localizes a hard ANE crash in the file log
                    var feedDict: [String: MLFeatureValue] = [:]
                    for name in model.modelDescription.inputDescriptionsByName.keys { feedDict[name] = common[name] ?? state[name] }
                    let out = try await model.prediction(from: MLDictionaryFeatureProvider(dictionary: feedDict))
                    if let h = out.featureValue(for: "hidden")?.multiArrayValue { hidden = h; break }
                    state["h_in"] = out.featureValue(for: "h")
                    if let p = out.featureValue(for: "ple_out") { state["ple"] = p }
                    for name in out.featureNames where name.hasSuffix("_out") && name != "ple_out" {
                        state[String(name.dropLast(4))] = out.featureValue(for: name)
                    }
                }
                times.append(now() - t)
                guard let hidden else { log("\(feed): no hidden output"); continue }

                // hidden rows follow picks: every question's <decide>, then each question's </opt> positions.
                let rowStride = hidden.strides[0].intValue
                let h16 = hidden.dataPointer.bindMemory(to: Float16.self, capacity: hidden.count)
                func row(_ i: Int) -> [Float] { (0..<hid).map { Float(h16[i * rowStride + $0]) } }
                let q = meta["q"] as! Int, optLens = meta["opt_lens"] as! [Int]
                var j = q
                for qi in 0..<q {
                    let hq = project(wq, bq, row(qi))
                    var z = (0..<optLens[qi]).map { o -> Float in
                        let hk = project(wk, bk, row(j + o))
                        return zip(hk, hq).reduce(0) { $0 + $1.0 * $1.1 } / sqrtf(Float(dp)) / temperature
                    }
                    j += optLens[qi]
                    let m = z.max()!
                    z = z.map { expf($0 - m) }
                    let s = z.reduce(0, +)
                    let p = z.map { Double($0 / s) }
                    questions += 1
                    let fp32 = (meta["ref"] as? [[Double]]).flatMap { qi < $0.count ? $0[qi] : nil }
                    guard let ref = macP[feed]?[qi] ?? fp32 else { continue }
                    if let photo = meta["photo"] as? String {
                        log(String(format: "%@ q%d p=%@ ref=%@ (encode %.0f ms)", photo, qi,
                                   p.map { String(format: "%.3f", $0) }.joined(separator: ","),
                                   ref.map { String(format: "%.3f", $0) }.joined(separator: ","), encodeSeconds * 1000))
                    }
                    maxDp = max(maxDp, zip(p, ref).map { abs($0 - $1) }.max() ?? 0)
                    let top = p.indices.max { p[$0] < p[$1] }!, refTop = ref.indices.max { ref[$0] < ref[$1] }!
                    if top != refTop {
                        flips += 1
                        let sorted = ref.sorted(by: >)
                        if sorted[0] - sorted[1] > 0.05 { clearFlips += 1 }
                    }
                }
                if fi < 3 || fi == feeds.count - 1 {
                    log(String(format: "%@ %.2fs fp=%dMB", feed, times.last!, VLMService.footprintMB))
                }
            }
            let s = times.sorted()
            log(String(format: "DONE %@ (%@): %d requests, %d questions | p50 %.2fs p95 %.2fs first %.2fs | vs %@: %d top changes (%d clear-cut), max|dp| %.3f | peak fp=%dMB thermal=%d",
                       label, feedsDir, times.count, questions, s[s.count / 2], s[min(s.count - 1, s.count * 95 / 100)], times.first ?? 0,
                       reference, flips, clearFlips, maxDp, VLMService.peakFootprintMB, ProcessInfo.processInfo.thermalState.rawValue))
        } catch {
            log("FAILED (\(label)): \(String(describing: error))")
        }
    }
}
#endif
