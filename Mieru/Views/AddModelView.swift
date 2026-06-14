//
//  AddModelView.swift
//  Mieru
//
//  Form for adding a custom HuggingFace model to the catalog.
//

import SwiftUI

struct AddModelView: View {

    @Environment(\.dismiss) private var dismiss
    @State private var catalog = ModelCatalog.shared

    @State private var hfId: String = "mlx-community/"
    @State private var displayName: String = ""
    @State private var family: ModelFamily = .gemma
    @State private var kind: ModelKind = .vlm
    @State private var contextLength: String = "131072"
    @State private var estimatedSizeGB: String = "2.0"
    @State private var architecture: String = "Gemma 4"
    @State private var modelType: String = "Gemma4_VLM"

    private var canAdd: Bool {
        !hfId.trimmingCharacters(in: .whitespaces).isEmpty
            && hfId.contains("/")
            && !displayName.trimmingCharacters(in: .whitespaces).isEmpty
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                Form {
                    Section("HuggingFace Model") {
                        TextField("org/model-id", text: $hfId)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .font(.system(size: 14, design: .monospaced))

                        TextField("Display name", text: $displayName)
                    }

                    Section("Type") {
                        Picker("Family", selection: $family) {
                            ForEach(ModelFamily.allCases, id: \.self) { family in
                                Text(family.displayName).tag(family)
                            }
                        }

                        Picker("Kind", selection: $kind) {
                            Text("Vision (VLM)").tag(ModelKind.vlm)
                            Text("Voice (TTS)").tag(ModelKind.tts)
                            Text("Listen (STT)").tag(ModelKind.stt)
                        }
                    }

                    Section("Specs") {
                        TextField("Architecture (e.g. Gemma 4)", text: $architecture)

                        TextField("Mode Type (e.g. Gemma4_VLM)", text: $modelType)

                        HStack {
                            Text("Context length")
                            Spacer()
                            TextField("131072", text: $contextLength)
                                .keyboardType(.numberPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                        }

                        HStack {
                            Text("Estimated size (GB)")
                            Spacer()
                            TextField("2.0", text: $estimatedSizeGB)
                                .keyboardType(.decimalPad)
                                .multilineTextAlignment(.trailing)
                                .frame(width: 100)
                        }
                    }
                }
                .scrollContentBackground(.hidden)
            }
            .navigationTitle("Add Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(Color.black, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                        .foregroundColor(.gray)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Add") {
                        add()
                    }
                    .disabled(!canAdd)
                    .foregroundColor(canAdd ? .blue : .gray)
                    .fontWeight(.semibold)
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    private func add() {
        let trimmedId = hfId.trimmingCharacters(in: .whitespaces)
        let trimmedName = displayName.trimmingCharacters(in: .whitespaces)
        let ctx = Int(contextLength) ?? 131072
        let size = Double(estimatedSizeGB) ?? 2.0
        let arch = architecture.trimmingCharacters(in: .whitespaces).isEmpty ? "Custom" : architecture
        let type = modelType.trimmingCharacters(in: .whitespaces).isEmpty ? "Custom" : modelType

        let model = ModelInfo(
            id: trimmedId,
            displayName: trimmedName,
            family: family,
            kind: kind,
            architecture: arch,
            modelType: type,
            format: kind == .tts ? .coreml : .mlx,
            contextLength: ctx, batch: 1, chunks: 1,
            estimatedSizeGB: size
        )

        catalog.addCustomModel(model)
        dismiss()
    }
}
