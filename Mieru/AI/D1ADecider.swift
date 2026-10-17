//
//  D1ADecider.swift
//  Mieru
//
//  D1A decisions on the resident Gemma 4 VLM: D1A's unmerged LoRA is switched on only for the
//  decide pass and read through its pointer head, so describing is untouched and no second model
//  is loaded. Requests follow D1A's System One format (d1a.core.api / d1a.core.encoding).
//

import Foundation
import MLX
import MLXLMCommon
@_spi(GemmaEncoder) import MLXVLM

struct D1ADecider: @unchecked Sendable {

    struct Question {
        let id: String
        let instructions: String
        let options: [String]
    }

    private let lora: LoRAContainer
    private let wq: MLXArray, bq: MLXArray, wk: MLXArray, bk: MLXArray
    private let scale: Float
    private let leading: [Int]
    private let delimiters: [Int]   // <state> <q> <opt> </opt> <decide>
    private let escape: NSRegularExpression
    private static let special = try! NSRegularExpression(pattern: #"<\|([A-Za-z0-9_]+)\|>"#)
    private static let maxState = 384   // d1a.core.encoding.MAX_STATE

    /// `directory` holds adapters.safetensors, adapter_config.json, head.safetensors,
    /// d1a_config.json and layout.json.
    init(directory: URL) throws {
        func json(_ name: String) throws -> [String: Any] {
            try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent(name))) as! [String: Any]
        }
        lora = try LoRAContainer.from(directory: directory)
        let head = try MLX.loadArrays(url: directory.appendingPathComponent("head.safetensors")).mapValues { $0.asType(.float32) }
        wq = head["q.weight"]!; bq = head["q.bias"]!; wk = head["k.weight"]!; bk = head["k.bias"]!
        let temperature = Float((try json("d1a_config.json"))["temperature"] as! Double)
        scale = 1 / sqrtf(Float(wq.dim(0))) / temperature
        let layout = try json("layout.json")
        leading = layout["leading_ids"] as! [Int]
        delimiters = layout["delimiter_ids"] as! [Int]
        escape = try NSRegularExpression(pattern: layout["escape_pattern"] as! String)
    }

    // MARK: - Request text (ports of d1a.core.api)

    /// render() for a flat object: "key: value" lines in order.
    static func render(_ pairs: [(String, String)]) -> String {
        pairs.map { "\($0.0): \($0.1)" }.joined(separator: "\n")
    }

    /// noul questions from a System One `questions` object: options "no" then "yes", each with its
    /// criteria text from the "false" / "true" keys. Choice questions are skipped: JSONSerialization
    /// drops key order, and their option order must match the request's.
    static func questions(from json: [String: Any], order: [String]) -> [Question] {
        order.compactMap { id in
            guard let q = json[id] as? [String: Any], q["type"] as? String == "noul",
                  let instructions = q["instructions"] as? String else { return nil }
            let criteria = q["criteria"] as? [String: Any] ?? [:]
            func option(_ name: String, _ text: Any?) -> String {
                guard let text = text as? String, !text.isEmpty else { return name }
                return name + ": " + text
            }
            return Question(id: id, instructions: instructions,
                            options: [option("no", criteria["false"]), option("yes", criteria["true"])])
        }
    }

    /// user_tokens(): caller text can never produce a delimiter or control token.
    private func userTokens(_ text: String, _ tokenizer: Tokenizer) -> [Int] {
        let ns = NSMutableString(string: text)
        for m in escape.matches(in: text, range: NSRange(location: 0, length: ns.length)).reversed() {
            ns.insert("¦", at: m.range.location + 1)
        }
        let t = Self.special.stringByReplacingMatches(in: ns as String, range: NSRange(location: 0, length: ns.length), withTemplate: "<¦$1¦>")
        return tokenizer.encode(text: t, addSpecialTokens: false)
    }

    // MARK: - Decide

    /// Option probabilities per question. Each question runs as one causal row
    /// (<bos> <state> state <q> instructions (<opt> option </opt>)... <decide>), which equals D1A's
    /// packed block-causal encoding for that question. Gemma is restored before returning.
    func decide(state: String, questions: [Question], gemma: Gemma4, tokenizer: Tokenizer) throws -> [[Float]] {
        try lora.load(into: gemma)
        defer { lora.unload(from: gemma) }
        let head = leading + [delimiters[0]]
        let prefix = head + userTokens(state, tokenizer).prefix(Self.maxState - head.count)
        return questions.map { q in
            var row = prefix + [delimiters[1]] + userTokens(q.instructions, tokenizer)
            var opts: [Int32] = []
            for option in q.options {
                row += [delimiters[2]] + userTokens(option, tokenizer) + [delimiters[3]]
                opts.append(Int32(row.count - 1))
            }
            row.append(delimiters[4])
            let h = gemma.textHiddenStates(MLXArray(row.map { Int32($0) }).expandedDimensions(axis: 0))[0].asType(.float32)
            let hq = matmul(h[row.count - 1], wq.T) + bq
            let hk = matmul(h[MLXArray(opts)], wk.T) + bk
            let p = softmax(matmul(hk, hq) * scale, axis: -1)
            eval(p)
            return p.asArray(Float.self)
        }
    }
}
