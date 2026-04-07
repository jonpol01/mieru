//
//  CameraManager.swift
//  Mieru
//
//  AVCaptureSession wrapper that delivers camera frames.
//

import AVFoundation
import UIKit

@Observable
class CameraManager: NSObject {

    // MARK: - Public state

    /// Most recent camera frame, updated continuously.
    public private(set) var latestFrame: CVPixelBuffer?

    /// Whether the camera session is running.
    public private(set) var isRunning = false

    /// The capture session — exposed for CameraPreviewView.
    let session = AVCaptureSession()

    // MARK: - Private

    private let outputQueue = DispatchQueue(label: "camera.output", qos: .userInteractive)
    private var lastFrameTime = CFAbsoluteTimeGetCurrent()
    private let minFrameInterval: CFAbsoluteTime = 0.1 // ~10 FPS cap

    // MARK: - Setup

    func setup() {
        guard !isRunning else { return }

        session.beginConfiguration()
        session.sessionPreset = .photo

        // Back camera
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            NSLog("[Camera] Failed to access back camera")
            session.commitConfiguration()
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        // Video data output
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: outputQueue)

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        session.commitConfiguration()
    }

    func start() {
        guard !session.isRunning else { return }
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
            Task { @MainActor in
                self?.isRunning = true
            }
        }
    }

    func stop() {
        guard session.isRunning else { return }
        session.stopRunning()
        isRunning = false
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastFrameTime >= minFrameInterval else { return }
        lastFrameTime = now

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        latestFrame = pixelBuffer
    }
}
