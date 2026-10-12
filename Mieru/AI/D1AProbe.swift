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

enum D1AProbe {

    private static let chunks = ["L512_00-07", "L512_07-15", "L512_15-22", "L512_22-29", "L512_29-35"]

    private static func log(_ line: String) { VLMService.log("d1a: " + line) }

    private static func now() -> Double { CFAbsoluteTimeGetCurrent() }

    /// Peak physical footprint of the process so far (what jetsam counts), in MB.
    private static var peakFootprintMB: Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.ledger_phys_footprint_peak / (1024 * 1024)) : -1
    }

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

    static func run(units: MLComputeUnits, label: String) async {
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
                       label, now() - tLoad, VLMService.footprintMB, peakFootprintMB))

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
            let mac = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("mac_int4.json"))) as! [String: Any]
            var macP: [String: [Int: [Double]]] = [:]
            for row in mac["rows"] as! [[String: Any]] {
                macP[(row["feed"] as! String).replacingOccurrences(of: ".npz", with: ""), default: [:]][row["q"] as! Int] = row["p"] as? [Double]
            }

            let feedDir = root.appendingPathComponent("feeds")
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
                var state: [String: MLFeatureValue] = [
                    "emb": MLFeatureValue(multiArray: try array(d.appendingPathComponent("emb.bin"), [1, 512, 1536], .float16)),
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
                    guard let ref = macP[feed]?[qi] else { continue }
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
            log(String(format: "DONE %@: %d requests, %d questions | p50 %.2fs p95 %.2fs first %.2fs | vs Mac int4: %d top changes (%d clear-cut), max|dp| %.3f | peak fp=%dMB thermal=%d",
                       label, times.count, questions, s[s.count / 2], s[min(s.count - 1, s.count * 95 / 100)], times.first ?? 0,
                       flips, clearFlips, maxDp, peakFootprintMB, ProcessInfo.processInfo.thermalState.rawValue))
        } catch {
            log("FAILED (\(label)): \(String(describing: error))")
        }
    }
}
#endif
