import AVFoundation
import Combine
import CoreML
import CreateML
import Darwin
import Dispatch
import Foundation
import ImageIO
import Network
import Vision

struct Config {
    var sampleIntervalSeconds: Double = 2.0
    var maxSamples: Int = 30
    var confidenceThreshold: Float = 0.2
    var logLabels: Bool = false
    var logTopLabelsCount: Int = 5
    var maxLabelsToScan: Int = 50
    var centerCropScale: Double = 0.6
    var roiGrid: Int = 1
    var motionROIEnabled: Bool = false
    var motionThreshold: Float = 0.08
    var motionMinAreaFraction: Double = 0.002
    var motionDownscaleMax: Int = 160
    var perClassThresholds: [String: Float] = [
        "bird": 0.2,
        "badger": 0.15,
        "fox": 0.35,
        "cat": 0.35,
        "mouse": 0.18,
        "squirrel": 0.25
    ]
    var badgerOverrideThreshold: Float = 0.12
    var dryRun: Bool = false
}

enum RecognizeError: Error, CustomStringConvertible {
    case invalidArguments(String)
    case notDirectory(String)
    case unsupportedOS

    var description: String {
        switch self {
        case .invalidArguments(let message):
            return message
        case .notDirectory(let path):
            return "Not a directory: \(path)"
        case .unsupportedOS:
            return "This tool requires macOS 14 or later for Vision image classification."
        }
    }
}

let animalMatchers: [(animal: String, aliases: [String])] = [
    (
        animal: "bird",
        aliases: [
            "bird", "sparrow", "eagle", "hawk", "owl", "duck", "goose", "pigeon", "crow",
            "raven", "seagull", "gull", "parrot", "chicken", "rooster", "hen", "turkey",
            "woodpecker", "falcon"
        ]
    ),
    (animal: "badger", aliases: ["badger"]),
    (animal: "fox", aliases: ["fox"]),
    (animal: "cat", aliases: ["cat", "kitten", "tabby"]),
    (animal: "mouse", aliases: ["mouse", "mice", "rodent"]),
    (animal: "squirrel", aliases: ["squirrel", "chipmunk"])
]

func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

let usageText = """
Usage:
  recognize train <coco_dir> <output_model> [--images-root <path>]
  recognize detect <model_path> <images_dir> [--threshold <0-1>]
  recognize detect-video <model_path> <video_dir> [options]
  recognize ls-backend --model <model_path> --from-name <name> --to-name <name> [options]
  recognize <video_dir> [options]

Training:
  coco_dir must contain result.json and images/ from a Label Studio COCO export.
  Use --images-root to override image lookup when result.json paths are wrong.
  output_model should be a .mlmodel file (Create ML will write it).

Examples:
  recognize train training/trainingset1 Models/Animals.mlmodel
  recognize train training/trainingset1 Models/Animals.mlmodel --images-root "/Volumes/Samsung 2TB/Trailcam/labelling/images"
  recognize detect Models/Animals.mlmodel ./images --threshold 0.3
  recognize detect-video Models/Animals.mlmodel ./videos --interval 1.0 --max-samples 60 --threshold 0.3
  recognize ls-backend --model Models/Animals.mlmodel --from-name label --to-name image

Detect-video options:
  --interval <seconds>     Seconds between sampled frames (default: 2.0)
  --max-samples <count>    Max frames to sample per file (default: 30)
  --threshold <0-1>        Confidence threshold (default: 0.3)

Label Studio backend options:
  --model <path>           CoreML model (.mlmodel or .mlmodelc)
  --host <host>            Host to advertise (default: 0.0.0.0)
  --port <port>            Port to listen on (default: 9090)
  --threshold <0-1>        Confidence threshold (default: 0.3)
  --from-name <name>       Label config "from_name"
  --to-name <name>         Label config "to_name"
  --data-key <key>         Task data key for image URL (default: image)
  --data-root <path>       Optional root to resolve relative image paths
  --label-studio-url <url> Base URL for Label Studio to fetch /data/local-files
  --label-studio-token <t> API token for Label Studio downloads (optional)
  --project-id <id>        Default project ID for /train (optional)
  --train-output <path>    Output .mlmodel path for /train (optional)
  --model-version <text>   Model version string (default: recognize)
  --allow-absolute         Allow absolute file paths in task data

Legacy video options:
  --interval <seconds>     Seconds between sampled frames (default: 2.0)
  --max-samples <count>    Max frames to sample per file (default: 30)
  --threshold <0-1>        Confidence threshold (default: 0.2)
  --max-labels <count>     Max labels to scan per frame (default: 50)
  --center-crop <0-1>      Center-crop scale after full-frame miss (default: 0.6, set to 1 to disable)
  --roi-grid <n>           Grid tiles to scan after misses (default: 1, values 1-4)
  --motion-roi             Enable motion-based ROI crop between frames
  --motion-threshold <0-1> Pixel diff threshold for motion ROI (default: 0.08)
  --motion-min-area <0-1>  Minimum ROI area fraction to accept (default: 0.002)
  --motion-downscale <px>  Max downscale dimension for motion ROI (default: 160)
  --log-labels             Print matched label and confidence per file; if no match, print top labels
  --log-top <count>        Count of labels to show when no match (default: 5)
  --dry-run                Show planned moves without changing files
  -h, --help               Show this help
"""

func usageError(_ message: String? = nil) -> RecognizeError {
    if let message {
        return .invalidArguments("\(message)\n\n\(usageText)")
    }
    return .invalidArguments(usageText)
}

func isMP4File(_ url: URL) -> Bool {
    let ext = url.pathExtension.lowercased()
    return ext == "mp4" || ext == "mov"
}

func isImageFile(_ url: URL) -> Bool {
    let ext = url.pathExtension.lowercased()
    return ["jpg", "jpeg", "png"].contains(ext)
}

func loadCGImage(from url: URL) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw RecognizeError.invalidArguments("Unable to load image: \(url.path)")
    }
    return image
}

struct COCOFile: Decodable {
    struct Image: Decodable {
        let id: Int
        let fileName: String
        let width: Int?
        let height: Int?

        enum CodingKeys: String, CodingKey {
            case id
            case fileName = "file_name"
            case width
            case height
        }
    }

    struct Annotation: Decodable {
        let id: Int
        let imageId: Int
        let categoryId: Int
        let bbox: [Double]

        enum CodingKeys: String, CodingKey {
            case id
            case imageId = "image_id"
            case categoryId = "category_id"
            case bbox
        }
    }

    struct Category: Decodable {
        let id: Int
        let name: String
    }

    let images: [Image]
    let annotations: [Annotation]
    let categories: [Category]
}

func resolveImageRelativePath(_ fileName: String) -> String {
    let normalized = fileName.replacingOccurrences(of: "\\", with: "/")
    let decodedNormalized = normalized.removingPercentEncoding ?? normalized
    let trimmed = decodedNormalized.hasPrefix("../../label-studio/data/media/")
        ? String(decodedNormalized.dropFirst("../../label-studio/data/media/".count))
        : decodedNormalized
    if let range = trimmed.range(of: "images/") {
        let suffix = trimmed[range.upperBound...]
        return String(suffix)
    }
    if trimmed.hasPrefix("/") {
        return URL(fileURLWithPath: trimmed).lastPathComponent
    }
    return trimmed
}

func prepareCreateMLDataset(cocoDir: URL, imagesRootOverride: URL?) throws -> URL {
    let cocoFile = cocoDir.appendingPathComponent("result.json")
    let imagesDir = cocoDir.appendingPathComponent("images")

    guard FileManager.default.fileExists(atPath: cocoFile.path) else {
        throw RecognizeError.invalidArguments("Missing result.json in \(cocoDir.path)")
    }
    let sourceImagesDir: URL
    if let override = imagesRootOverride {
        sourceImagesDir = override
    } else {
        sourceImagesDir = imagesDir
    }

    guard FileManager.default.fileExists(atPath: sourceImagesDir.path) else {
        throw RecognizeError.invalidArguments("Missing images directory at \(sourceImagesDir.path)")
    }

    let data = try Data(contentsOf: cocoFile)
    let coco = try JSONDecoder().decode(COCOFile.self, from: data)

    var categories: [Int: String] = [:]
    for category in coco.categories {
        categories[category.id] = category.name
    }

    var annotationsByImage: [Int: [COCOFile.Annotation]] = [:]
    for annotation in coco.annotations {
        annotationsByImage[annotation.imageId, default: []].append(annotation)
    }

    let datasetDir = cocoDir.appendingPathComponent("createml")
    if FileManager.default.fileExists(atPath: datasetDir.path) {
        try FileManager.default.removeItem(at: datasetDir)
    }
    try FileManager.default.createDirectory(at: datasetDir, withIntermediateDirectories: true)

    let datasetImagesDir = datasetDir

    var createMLAnnotations: [[String: Any]] = []
    var missingImages = 0
    var missingImagePaths: [String] = []
    var copiedImages = Set<String>()

    for image in coco.images {
        let relativePath = resolveImageRelativePath(image.fileName)
        let imageURL = sourceImagesDir.appendingPathComponent(relativePath)
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            missingImages += 1
            missingImagePaths.append(relativePath)
            continue
        }
        let flattenedName = relativePath.replacingOccurrences(of: "/", with: "__")
        let destURL = datasetImagesDir.appendingPathComponent(flattenedName)
        if !copiedImages.contains(relativePath) {
            if !FileManager.default.fileExists(atPath: destURL.path) {
                try FileManager.default.copyItem(at: imageURL, to: destURL)
            }
            copiedImages.insert(relativePath)
        }

        let anns = annotationsByImage[image.id] ?? []
        let entries: [[String: Any]] = anns.compactMap { ann in
            guard ann.bbox.count == 4 else { return nil }
            guard let label = categories[ann.categoryId] else { return nil }
        let x = ann.bbox[0]
        let y = ann.bbox[1]
        let width = ann.bbox[2]
        let height = ann.bbox[3]

        return [
            "label": label,
            "coordinates": [
                "x": x,
                "y": y,
                "width": width,
                "height": height
            ]
        ]
    }

        createMLAnnotations.append([
            "image": flattenedName,
            "annotations": entries
        ])
    }

    let outputURL = datasetDir.appendingPathComponent("annotations.json")
    let json = try JSONSerialization.data(withJSONObject: createMLAnnotations, options: [.prettyPrinted, .sortedKeys])
    try json.write(to: outputURL)

    if missingImages > 0 {
        print("Warning: \(missingImages) images referenced in COCO were not found in images/")
        for path in missingImagePaths.sorted() {
            print("Missing image: \(path)")
        }
    }

    return datasetDir
}

func trainModel(cocoDir: URL, outputModelURL: URL, imagesRootOverride: URL?) throws {
    let datasetDir = try prepareCreateMLDataset(cocoDir: cocoDir, imagesRootOverride: imagesRootOverride)
    let data: MLObjectDetector.DataSource = .directoryWithImagesAndJsonAnnotation(at: datasetDir)
    let parameters = MLObjectDetector.ModelParameters()
    let annotationType = MLObjectDetector.AnnotationType.boundingBox(units: .pixel, origin: .topLeft, anchor: .topLeft)
    let job = try MLObjectDetector.train(
        trainingData: data,
        annotationType: annotationType,
        parameters: parameters
    )

    var trainedModel: MLObjectDetector?
    var trainingError: Error?
    let semaphore = DispatchSemaphore(value: 0)
    let cancellable = job.result.sink(
        receiveCompletion: { completion in
            if case .failure(let error) = completion {
                trainingError = error
            }
            semaphore.signal()
        },
        receiveValue: { model in
            trainedModel = model
        }
    )

    let progress = job.progress
    while semaphore.wait(timeout: .now() + 5) == .timedOut {
        if progress.isIndeterminate {
            print("Training progress: indeterminate")
        } else {
            let percent = Int(progress.fractionCompleted * 100)
            print("Training progress: \(percent)%")
        }
    }
    _ = cancellable

    if let error = trainingError {
        throw error
    }
    guard let model = trainedModel else {
        throw RecognizeError.invalidArguments("Training did not return a model")
    }
    let outputDir = outputModelURL.deletingLastPathComponent()
    if !FileManager.default.fileExists(atPath: outputDir.path) {
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
    }
    try model.write(to: outputModelURL)
    print("Model written to \(outputModelURL.path)")
}

func loadCoreMLModel(from url: URL) throws -> VNCoreMLModel {
    let modelURL: URL
    if url.pathExtension == "mlmodelc" {
        modelURL = url
    } else {
        modelURL = try MLModel.compileModel(at: url)
    }
    let model = try MLModel(contentsOf: modelURL)
    return try VNCoreMLModel(for: model)
}

struct Detection {
    let label: String
    let confidence: Float
    let boundingBox: CGRect
}

func detectObjects(in image: CGImage, model: VNCoreMLModel, threshold: Float) throws -> [Detection] {
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .scaleFit

    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try handler.perform([request])

    if let results = request.results as? [VNRecognizedObjectObservation] {
        return results.compactMap { observation in
            guard let topLabel = observation.labels.max(by: { $0.confidence < $1.confidence }) else { return nil }
            guard topLabel.confidence >= threshold else { return nil }
            return Detection(
                label: topLabel.identifier,
                confidence: topLabel.confidence,
                boundingBox: observation.boundingBox
            )
        }
    }

    if let results = request.results as? [VNClassificationObservation] {
        return results.compactMap { observation in
            guard observation.confidence >= threshold else { return nil }
            return Detection(
                label: observation.identifier,
                confidence: observation.confidence,
                boundingBox: .zero
            )
        }
    }

    return []
}

func detectObjects(in imageURL: URL, model: VNCoreMLModel, threshold: Float) throws -> [Detection] {
    let image = try loadCGImage(from: imageURL)
    return try detectObjects(in: image, model: model, threshold: threshold)
}

func classifyImage(in cgImage: CGImage) throws -> [VNClassificationObservation] {
    guard #available(macOS 14.0, *) else {
        throw RecognizeError.unsupportedOS
    }

    let request = VNClassifyImageRequest()
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try handler.perform([request])

    return request.results ?? []
}

struct MatchResult {
    let animal: String
    let label: String
    let confidence: Float
    let source: String
}

func scanObservations(_ observations: [VNClassificationObservation], source: String, config: Config) -> MatchResult? {
    let maxCount = min(config.maxLabelsToScan, observations.count)

    var bestByAnimal: [String: MatchResult] = [:]

    for observation in observations.prefix(maxCount) {
        let identifier = observation.identifier.lowercased()
        for matcher in animalMatchers {
            if matcher.aliases.contains(where: { identifier.contains($0) }) {
                let candidate = MatchResult(
                    animal: matcher.animal,
                    label: observation.identifier,
                    confidence: observation.confidence,
                    source: source
                )
                if let current = bestByAnimal[matcher.animal] {
                    if candidate.confidence > current.confidence {
                        bestByAnimal[matcher.animal] = candidate
                    }
                } else {
                    bestByAnimal[matcher.animal] = candidate
                }
                break
            }
        }
    }

    if let badger = bestByAnimal["badger"], badger.confidence >= config.badgerOverrideThreshold {
        return badger
    }

    var best: MatchResult?
    for match in bestByAnimal.values {
        let threshold = config.perClassThresholds[match.animal] ?? config.confidenceThreshold
        guard match.confidence >= threshold else { continue }
        if let currentBest = best {
            if match.confidence > currentBest.confidence {
                best = match
            }
        } else {
            best = match
        }
    }

    return best
}

func updateBestLabels(_ observations: [VNClassificationObservation], bestLabels: inout [String: Float], limit: Int) {
    let maxCount = min(limit, observations.count)
    for observation in observations.prefix(maxCount) {
        let current = bestLabels[observation.identifier] ?? 0
        if observation.confidence > current {
            bestLabels[observation.identifier] = observation.confidence
        }
    }
}

func centerCropRect(for image: CGImage, scale: Double) -> CGRect? {
    guard scale > 0, scale < 1 else { return nil }
    let width = Double(image.width)
    let height = Double(image.height)
    let cropWidth = width * scale
    let cropHeight = height * scale
    let originX = (width - cropWidth) / 2
    let originY = (height - cropHeight) / 2
    return CGRect(x: originX, y: originY, width: cropWidth, height: cropHeight).integral
}

func gridCropRects(for image: CGImage, grid: Int) -> [CGRect] {
    guard grid > 1 else { return [] }
    let width = Double(image.width)
    let height = Double(image.height)
    let tileWidth = width / Double(grid)
    let tileHeight = height / Double(grid)
    var rects: [CGRect] = []
    for row in 0..<grid {
        for col in 0..<grid {
            let originX = Double(col) * tileWidth
            let originY = Double(row) * tileHeight
            rects.append(CGRect(x: originX, y: originY, width: tileWidth, height: tileHeight).integral)
        }
    }
    return rects
}

func grayscaleBuffer(for image: CGImage, maxDimension: Int) -> (width: Int, height: Int, bytes: [UInt8])? {
    guard maxDimension > 0 else { return nil }
    let width = image.width
    let height = image.height
    guard width > 0, height > 0 else { return nil }

    let maxSide = max(width, height)
    let scale = Double(maxDimension) / Double(maxSide)
    let targetWidth = max(1, Int(Double(width) * scale))
    let targetHeight = max(1, Int(Double(height) * scale))

    var pixels = [UInt8](repeating: 0, count: targetWidth * targetHeight)
    let colorSpace = CGColorSpace(name: CGColorSpace.genericGrayGamma2_2) ?? CGColorSpaceCreateDeviceGray()
    guard let context = CGContext(
        data: &pixels,
        width: targetWidth,
        height: targetHeight,
        bitsPerComponent: 8,
        bytesPerRow: targetWidth,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.none.rawValue
    ) else {
        return nil
    }

    context.interpolationQuality = .low
    context.draw(image, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
    return (targetWidth, targetHeight, pixels)
}

func motionROI(previous: CGImage, current: CGImage, config: Config) -> CGRect? {
    guard previous.width == current.width, previous.height == current.height else {
        return nil
    }
    guard let prevBuffer = grayscaleBuffer(for: previous, maxDimension: config.motionDownscaleMax),
          let currBuffer = grayscaleBuffer(for: current, maxDimension: config.motionDownscaleMax),
          prevBuffer.width == currBuffer.width,
          prevBuffer.height == currBuffer.height else {
        return nil
    }

    let width = prevBuffer.width
    let height = prevBuffer.height
    let threshold = UInt8(max(0, min(255, Int(config.motionThreshold * 255))))

    var minX = width
    var minY = height
    var maxX = -1
    var maxY = -1

    for y in 0..<height {
        let rowStart = y * width
        for x in 0..<width {
            let index = rowStart + x
            let a = prevBuffer.bytes[index]
            let b = currBuffer.bytes[index]
            let diff = a > b ? a - b : b - a
            if diff > threshold {
                if x < minX { minX = x }
                if y < minY { minY = y }
                if x > maxX { maxX = x }
                if y > maxY { maxY = y }
            }
        }
    }

    if maxX < minX || maxY < minY {
        return nil
    }

    let roiWidth = maxX - minX + 1
    let roiHeight = maxY - minY + 1
    let areaFraction = Double(roiWidth * roiHeight) / Double(width * height)
    if areaFraction < config.motionMinAreaFraction {
        return nil
    }

    let scaleX = Double(current.width) / Double(width)
    let scaleY = Double(current.height) / Double(height)
    var rect = CGRect(
        x: Double(minX) * scaleX,
        y: Double(minY) * scaleY,
        width: Double(roiWidth) * scaleX,
        height: Double(roiHeight) * scaleY
    )

    let padX = rect.width * 0.1
    let padY = rect.height * 0.1
    rect = rect.insetBy(dx: -padX, dy: -padY)

    let maxRect = CGRect(x: 0, y: 0, width: current.width, height: current.height)
    let clamped = rect.intersection(maxRect).integral
    return clamped.isNull ? nil : clamped
}

func detectAnimal(in fileURL: URL, fileName: String, config: Config) async throws -> String? {
    let asset = AVAsset(url: fileURL)
    let duration = try await asset.load(.duration)
    let durationSeconds = CMTimeGetSeconds(duration)
    guard durationSeconds.isFinite, durationSeconds > 0 else {
        return nil
    }

    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true

    let maxDuration = min(durationSeconds, config.sampleIntervalSeconds * Double(config.maxSamples))
    var times: [CMTime] = []
    var t: Double = 0
    while t < maxDuration {
        times.append(CMTime(seconds: t, preferredTimescale: 600))
        t += config.sampleIntervalSeconds
    }

    var bestLabels: [String: Float] = [:]
    var previousImage: CGImage?

    for time in times {
        var actualTime = CMTime.zero
        do {
            let cgImage = try generator.copyCGImage(at: time, actualTime: &actualTime)
            let results = try classifyImage(in: cgImage)

            if config.logLabels {
                updateBestLabels(results, bestLabels: &bestLabels, limit: config.logTopLabelsCount)
            }

            if let match = scanObservations(results, source: "full", config: config) {
                if config.logLabels {
                    let timeSeconds = CMTimeGetSeconds(actualTime)
                    let timeString = timeSeconds.isFinite ? String(format: "%.2f", timeSeconds) : "unknown"
                    let confidence = String(format: "%.2f", match.confidence)
                    print("Match \(fileName) at \(timeString)s [\(match.source)]: \(match.label) (\(confidence)) -> \(match.animal)")
                }
                return match.animal
            }

            if config.motionROIEnabled, let previousImage,
               let motionRect = motionROI(previous: previousImage, current: cgImage, config: config),
               let motionCrop = cgImage.cropping(to: motionRect) {
                let motionResults = try classifyImage(in: motionCrop)
                if config.logLabels {
                    updateBestLabels(motionResults, bestLabels: &bestLabels, limit: config.logTopLabelsCount)
                }
                if let match = scanObservations(motionResults, source: "motion-roi", config: config) {
                    if config.logLabels {
                        let timeSeconds = CMTimeGetSeconds(actualTime)
                        let timeString = timeSeconds.isFinite ? String(format: "%.2f", timeSeconds) : "unknown"
                        let confidence = String(format: "%.2f", match.confidence)
                        print("Match \(fileName) at \(timeString)s [\(match.source)]: \(match.label) (\(confidence)) -> \(match.animal)")
                    }
                    return match.animal
                }
            }

            if config.centerCropScale < 1, let cropRect = centerCropRect(for: cgImage, scale: config.centerCropScale),
               let cropped = cgImage.cropping(to: cropRect) {
                let cropResults = try classifyImage(in: cropped)
                if config.logLabels {
                    updateBestLabels(cropResults, bestLabels: &bestLabels, limit: config.logTopLabelsCount)
                }
                if let match = scanObservations(cropResults, source: "center-crop", config: config) {
                    if config.logLabels {
                        let timeSeconds = CMTimeGetSeconds(actualTime)
                        let timeString = timeSeconds.isFinite ? String(format: "%.2f", timeSeconds) : "unknown"
                        let confidence = String(format: "%.2f", match.confidence)
                        print("Match \(fileName) at \(timeString)s [\(match.source)]: \(match.label) (\(confidence)) -> \(match.animal)")
                    }
                    return match.animal
                }
            }

            if config.roiGrid > 1 {
                let rects = gridCropRects(for: cgImage, grid: config.roiGrid)
                for (index, rect) in rects.enumerated() {
                    guard let tile = cgImage.cropping(to: rect) else { continue }
                    let tileResults = try classifyImage(in: tile)
                    if config.logLabels {
                        updateBestLabels(tileResults, bestLabels: &bestLabels, limit: config.logTopLabelsCount)
                    }
                    if let match = scanObservations(tileResults, source: "tile-\(index + 1)", config: config) {
                        if config.logLabels {
                            let timeSeconds = CMTimeGetSeconds(actualTime)
                            let timeString = timeSeconds.isFinite ? String(format: "%.2f", timeSeconds) : "unknown"
                            let confidence = String(format: "%.2f", match.confidence)
                            print("Match \(fileName) at \(timeString)s [\(match.source)]: \(match.label) (\(confidence)) -> \(match.animal)")
                        }
                        return match.animal
                    }
                }
            }
            previousImage = cgImage
        } catch {
            continue
        }
    }

    if config.logLabels {
        let sorted = bestLabels.sorted { $0.value > $1.value }
        if sorted.isEmpty {
            print("No labels available for \(fileName)")
        } else {
            let top = sorted.prefix(config.logTopLabelsCount)
            let summary = top.map { "\($0.key) (\(String(format: "%.2f", $0.value)))" }.joined(separator: ", ")
            print("Top labels for \(fileName): \(summary)")
        }
    }

    return nil
}

func uniqueDestinationURL(in directory: URL, fileName: String, fileManager: FileManager) -> URL {
    var candidate = directory.appendingPathComponent(fileName)
    if !fileManager.fileExists(atPath: candidate.path) {
        return candidate
    }

    let baseName = (fileName as NSString).deletingPathExtension
    let ext = (fileName as NSString).pathExtension
    var counter = 1
    while true {
        let newName = "\(baseName)-\(counter)" + (ext.isEmpty ? "" : ".\(ext)")
        candidate = directory.appendingPathComponent(newName)
        if !fileManager.fileExists(atPath: candidate.path) {
            return candidate
        }
        counter += 1
    }
}

struct ParsedArguments {
    let inputURL: URL
    let config: Config
}

func parseLegacyArguments(_ args: [String]) throws -> ParsedArguments {
    guard args.count >= 2 else {
        throw usageError()
    }

    var config = Config()
    var inputPath: String?

    var index = 1
    while index < args.count {
        let arg = args[index]
        switch arg {
        case "-h", "--help":
            print(usageText)
            exit(0)
        case "--interval":
            index += 1
            guard index < args.count, let value = Double(args[index]), value > 0 else {
                throw usageError("Invalid --interval value.")
            }
            config.sampleIntervalSeconds = value
        case "--max-samples":
            index += 1
            guard index < args.count, let value = Int(args[index]), value > 0 else {
                throw usageError("Invalid --max-samples value.")
            }
            config.maxSamples = value
        case "--threshold":
            index += 1
            guard index < args.count, let value = Float(args[index]), value >= 0, value <= 1 else {
                throw usageError("Invalid --threshold value.")
            }
            config.confidenceThreshold = value
        case "--max-labels":
            index += 1
            guard index < args.count, let value = Int(args[index]), value > 0 else {
                throw usageError("Invalid --max-labels value.")
            }
            config.maxLabelsToScan = value
        case "--center-crop":
            index += 1
            guard index < args.count, let value = Double(args[index]), value > 0, value <= 1 else {
                throw usageError("Invalid --center-crop value.")
            }
            config.centerCropScale = value
        case "--roi-grid":
            index += 1
            guard index < args.count, let value = Int(args[index]), value >= 1, value <= 4 else {
                throw usageError("Invalid --roi-grid value.")
            }
            config.roiGrid = value
        case "--motion-roi":
            config.motionROIEnabled = true
        case "--motion-threshold":
            index += 1
            guard index < args.count, let value = Float(args[index]), value >= 0, value <= 1 else {
                throw usageError("Invalid --motion-threshold value.")
            }
            config.motionThreshold = value
        case "--motion-min-area":
            index += 1
            guard index < args.count, let value = Double(args[index]), value >= 0, value <= 1 else {
                throw usageError("Invalid --motion-min-area value.")
            }
            config.motionMinAreaFraction = value
        case "--motion-downscale":
            index += 1
            guard index < args.count, let value = Int(args[index]), value >= 16 else {
                throw usageError("Invalid --motion-downscale value.")
            }
            config.motionDownscaleMax = value
        case "--log-labels":
            config.logLabels = true
        case "--log-top":
            index += 1
            guard index < args.count, let value = Int(args[index]), value > 0 else {
                throw usageError("Invalid --log-top value.")
            }
            config.logTopLabelsCount = value
        case "--dry-run":
            config.dryRun = true
        default:
            if arg.hasPrefix("-") {
                throw usageError("Unknown option: \(arg)")
            }
            if inputPath == nil {
                inputPath = arg
            } else {
                throw usageError("Multiple directories provided.")
            }
        }
        index += 1
    }

    guard let inputPath else {
        throw usageError("Missing directory.")
    }

    let inputURL = URL(fileURLWithPath: inputPath).standardizedFileURL
    return ParsedArguments(inputURL: inputURL, config: config)
}

struct DetectArguments {
    let modelURL: URL
    let imagesURL: URL
    let threshold: Float
}

struct DetectVideoArguments {
    let modelURL: URL
    let videosURL: URL
    let threshold: Float
    let intervalSeconds: Double
    let maxSamples: Int
}

struct BackendArguments {
    let modelURL: URL
    let host: String
    let port: UInt16
    let threshold: Float
    let fromName: String
    let toName: String
    let dataKey: String
    let dataRoot: URL?
    let modelVersion: String
    let allowAbsolutePaths: Bool
    let labelStudioURL: URL?
    let labelStudioToken: String?
    let projectID: Int?
    let trainOutputURL: URL?
}

enum Command {
    case train(cocoDir: URL, outputModel: URL, imagesRootOverride: URL?)
    case detect(DetectArguments)
    case detectVideo(DetectVideoArguments)
    case backend(BackendArguments)
    case legacy(ParsedArguments)
}

func parseCommand(_ args: [String]) throws -> Command {
    guard args.count >= 2 else {
        throw usageError()
    }

    if args.contains("-h") || args.contains("--help") {
        print(usageText)
        exit(0)
    }

    switch args[1] {
    case "train":
        var cocoPath: String?
        var outputPathValue: String?
        var imagesRoot: URL?

        var index = 2
        while index < args.count {
            let arg = args[index]
            switch arg {
            case "--images-root":
                index += 1
                guard index < args.count else {
                    throw usageError("Missing value for --images-root.")
                }
                imagesRoot = URL(fileURLWithPath: args[index]).standardizedFileURL
            default:
                if arg.hasPrefix("-") {
                    throw usageError("Unknown option: \(arg)")
                }
                if cocoPath == nil {
                    cocoPath = arg
                } else if outputPathValue == nil {
                    outputPathValue = arg
                } else {
                    throw usageError("Too many arguments for train.")
                }
            }
            index += 1
        }

        guard let cocoPath, let outputPathValue else {
            throw usageError("Usage: recognize train <coco_dir> <output_model> [--images-root <path>]")
        }

        let cocoDir = URL(fileURLWithPath: cocoPath).standardizedFileURL
        let outputPath = URL(fileURLWithPath: outputPathValue).standardizedFileURL
        let outputURL: URL
        if outputPath.hasDirectoryPath {
            outputURL = outputPath.appendingPathComponent("AnimalDetector.mlmodel")
        } else {
            outputURL = outputPath
        }
        return .train(cocoDir: cocoDir, outputModel: outputURL, imagesRootOverride: imagesRoot)
    case "ls-backend":
        var modelPath: String?
        var host = "0.0.0.0"
        var port: UInt16 = 9090
        var threshold: Float = 0.3
        var fromName: String?
        var toName: String?
        var dataKey = "image"
        var dataRoot: URL?
        var modelVersion = "recognize"
        var allowAbsolutePaths = false
        var labelStudioURL: URL?
        var labelStudioToken: String?
        var projectID: Int?
        var trainOutputURL: URL?

        var index = 2
        while index < args.count {
            let arg = args[index]
            switch arg {
            case "--model":
                index += 1
                guard index < args.count else {
                    throw usageError("Missing value for --model.")
                }
                modelPath = args[index]
            case "--host":
                index += 1
                guard index < args.count else {
                    throw usageError("Missing value for --host.")
                }
                host = args[index]
            case "--port":
                index += 1
                guard index < args.count, let value = UInt16(args[index]) else {
                    throw usageError("Invalid --port value.")
                }
                port = value
            case "--threshold":
                index += 1
                guard index < args.count, let value = Float(args[index]), value >= 0, value <= 1 else {
                    throw usageError("Invalid --threshold value.")
                }
                threshold = value
            case "--from-name":
                index += 1
                guard index < args.count else {
                    throw usageError("Missing value for --from-name.")
                }
                fromName = args[index]
            case "--to-name":
                index += 1
                guard index < args.count else {
                    throw usageError("Missing value for --to-name.")
                }
                toName = args[index]
            case "--data-key":
                index += 1
                guard index < args.count else {
                    throw usageError("Missing value for --data-key.")
                }
                dataKey = args[index]
            case "--data-root":
                index += 1
                guard index < args.count else {
                    throw usageError("Missing value for --data-root.")
                }
                dataRoot = URL(fileURLWithPath: args[index]).standardizedFileURL
            case "--label-studio-url":
                index += 1
                guard index < args.count else {
                    throw usageError("Missing value for --label-studio-url.")
                }
                labelStudioURL = URL(string: args[index])
            case "--label-studio-token":
                index += 1
                guard index < args.count else {
                    throw usageError("Missing value for --label-studio-token.")
                }
                labelStudioToken = args[index]
            case "--project-id":
                index += 1
                guard index < args.count, let value = Int(args[index]) else {
                    throw usageError("Invalid --project-id value.")
                }
                projectID = value
            case "--train-output":
                index += 1
                guard index < args.count else {
                    throw usageError("Missing value for --train-output.")
                }
                trainOutputURL = URL(fileURLWithPath: args[index]).standardizedFileURL
            case "--model-version":
                index += 1
                guard index < args.count else {
                    throw usageError("Missing value for --model-version.")
                }
                modelVersion = args[index]
            case "--allow-absolute":
                allowAbsolutePaths = true
            default:
                if arg.hasPrefix("-") {
                    throw usageError("Unknown option: \(arg)")
                }
                if modelPath == nil {
                    modelPath = arg
                } else {
                    throw usageError("Too many arguments for ls-backend.")
                }
            }
            index += 1
        }

        if labelStudioToken == nil {
            let envToken = ProcessInfo.processInfo.environment["LS_TOKEN"] ?? ""
            if !envToken.isEmpty {
                labelStudioToken = envToken
            }
        }

        guard let modelPath, let fromName, let toName else {
            throw usageError("Usage: recognize ls-backend --model <model_path> --from-name <name> --to-name <name> [options]")
        }

        let modelURL = URL(fileURLWithPath: modelPath).standardizedFileURL
        return .backend(BackendArguments(
            modelURL: modelURL,
            host: host,
            port: port,
            threshold: threshold,
            fromName: fromName,
            toName: toName,
            dataKey: dataKey,
            dataRoot: dataRoot,
            modelVersion: modelVersion,
            allowAbsolutePaths: allowAbsolutePaths,
            labelStudioURL: labelStudioURL,
            labelStudioToken: labelStudioToken,
            projectID: projectID,
            trainOutputURL: trainOutputURL
        ))
    case "detect":
        var modelPath: String?
        var imagesPath: String?
        var threshold: Float = 0.3

        var index = 2
        while index < args.count {
            let arg = args[index]
            switch arg {
            case "--threshold":
                index += 1
                guard index < args.count, let value = Float(args[index]), value >= 0, value <= 1 else {
                    throw usageError("Invalid --threshold value.")
                }
                threshold = value
            default:
                if arg.hasPrefix("-") {
                    throw usageError("Unknown option: \(arg)")
                }
                if modelPath == nil {
                    modelPath = arg
                } else if imagesPath == nil {
                    imagesPath = arg
                } else {
                    throw usageError("Too many arguments for detect.")
                }
            }
            index += 1
        }

        guard let modelPath, let imagesPath else {
            throw usageError("Usage: recognize detect <model_path> <images_dir> [--threshold <0-1>]")
        }
        let modelURL = URL(fileURLWithPath: modelPath).standardizedFileURL
        let imagesURL = URL(fileURLWithPath: imagesPath).standardizedFileURL
        return .detect(DetectArguments(modelURL: modelURL, imagesURL: imagesURL, threshold: threshold))
    case "detect-video":
        var modelPath: String?
        var videosPath: String?
        var threshold: Float = 0.3
        var intervalSeconds: Double = 2.0
        var maxSamples: Int = 30

        var index = 2
        while index < args.count {
            let arg = args[index]
            switch arg {
            case "--threshold":
                index += 1
                guard index < args.count, let value = Float(args[index]), value >= 0, value <= 1 else {
                    throw usageError("Invalid --threshold value.")
                }
                threshold = value
            case "--interval":
                index += 1
                guard index < args.count, let value = Double(args[index]), value > 0 else {
                    throw usageError("Invalid --interval value.")
                }
                intervalSeconds = value
            case "--max-samples":
                index += 1
                guard index < args.count, let value = Int(args[index]), value > 0 else {
                    throw usageError("Invalid --max-samples value.")
                }
                maxSamples = value
            default:
                if arg.hasPrefix("-") {
                    throw usageError("Unknown option: \(arg)")
                }
                if modelPath == nil {
                    modelPath = arg
                } else if videosPath == nil {
                    videosPath = arg
                } else {
                    throw usageError("Too many arguments for detect-video.")
                }
            }
            index += 1
        }

        guard let modelPath, let videosPath else {
            throw usageError("Usage: recognize detect-video <model_path> <video_dir> [options]")
        }
        let modelURL = URL(fileURLWithPath: modelPath).standardizedFileURL
        let videosURL = URL(fileURLWithPath: videosPath).standardizedFileURL
        return .detectVideo(DetectVideoArguments(
            modelURL: modelURL,
            videosURL: videosURL,
            threshold: threshold,
            intervalSeconds: intervalSeconds,
            maxSamples: maxSamples
        ))
    default:
        return .legacy(try parseLegacyArguments(args))
    }
}

func runLegacy(_ parsed: ParsedArguments) async throws {
    let inputURL = parsed.inputURL
    let config = parsed.config

    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: inputURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw RecognizeError.notDirectory(inputURL.path)
    }

    let fileManager = FileManager.default
    let items = try fileManager.contentsOfDirectory(at: inputURL, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])

    let videoFiles = items.filter { url in
        guard isMP4File(url) else { return false }
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
        return values?.isRegularFile == true
    }

    if videoFiles.isEmpty {
        print("No MP4 or MOV files found in \(inputURL.path)")
        return
    }

    for fileURL in videoFiles {
        let fileName = fileURL.lastPathComponent
        do {
            if let animal = try await detectAnimal(in: fileURL, fileName: fileName, config: config) {
                let targetDir = inputURL.appendingPathComponent(animal, isDirectory: true)
                let destination = uniqueDestinationURL(in: targetDir, fileName: fileName, fileManager: fileManager)
                if config.dryRun {
                    print("Would move \(fileName) -> \(animal)/\(destination.lastPathComponent)")
                } else {
                    if !fileManager.fileExists(atPath: targetDir.path) {
                        try fileManager.createDirectory(at: targetDir, withIntermediateDirectories: true)
                    }
                    try fileManager.moveItem(at: fileURL, to: destination)
                    print("Moved \(fileName) -> \(animal)/\(destination.lastPathComponent)")
                }
            } else {
                print("No target animals detected in \(fileName)")
            }
        } catch {
            printError("Failed to process \(fileName): \(error)")
        }
    }
}

func runDetect(_ args: DetectArguments) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: args.imagesURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw RecognizeError.notDirectory(args.imagesURL.path)
    }

    let model = try loadCoreMLModel(from: args.modelURL)
    let fileManager = FileManager.default
    let items = try fileManager.contentsOfDirectory(at: args.imagesURL, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])
    let imageFiles = items.filter { url in
        guard isImageFile(url) else { return false }
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
        return values?.isRegularFile == true
    }

    if imageFiles.isEmpty {
        print("No JPG or PNG files found in \(args.imagesURL.path)")
        return
    }

    for imageURL in imageFiles {
        do {
            let detections = try detectObjects(in: imageURL, model: model, threshold: args.threshold)
            if detections.isEmpty {
                print("\(imageURL.lastPathComponent): no detections")
            } else {
                let summary = detections.map {
                    "\($0.label) (\(String(format: "%.2f", $0.confidence)))"
                }.joined(separator: ", ")
                print("\(imageURL.lastPathComponent): \(summary)")
            }
        } catch {
            printError("Failed to process \(imageURL.lastPathComponent): \(error)")
        }
    }
}

func detectObjectsInVideo(_ videoURL: URL, model: VNCoreMLModel, threshold: Float, intervalSeconds: Double, maxSamples: Int) async throws -> [String: (confidence: Float, time: Double)] {
    let asset = AVAsset(url: videoURL)
    let duration = try await asset.load(.duration)
    let durationSeconds = CMTimeGetSeconds(duration)
    guard durationSeconds.isFinite, durationSeconds > 0 else {
        return [:]
    }

    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true

    let maxDuration = min(durationSeconds, intervalSeconds * Double(maxSamples))
    var times: [CMTime] = []
    var t: Double = 0
    while t < maxDuration {
        times.append(CMTime(seconds: t, preferredTimescale: 600))
        t += intervalSeconds
    }

    var bestByLabel: [String: (confidence: Float, time: Double)] = [:]

    for time in times {
        var actualTime = CMTime.zero
        do {
            let cgImage = try generator.copyCGImage(at: time, actualTime: &actualTime)
            let detections = try detectObjects(in: cgImage, model: model, threshold: threshold)
            if detections.isEmpty {
                continue
            }
            let timeSeconds = CMTimeGetSeconds(actualTime)
            for detection in detections {
                if let current = bestByLabel[detection.label] {
                    if detection.confidence > current.confidence {
                        bestByLabel[detection.label] = (detection.confidence, timeSeconds)
                    }
                } else {
                    bestByLabel[detection.label] = (detection.confidence, timeSeconds)
                }
            }
        } catch {
            continue
        }
    }

    return bestByLabel
}

func runDetectVideo(_ args: DetectVideoArguments) async throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: args.videosURL.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw RecognizeError.notDirectory(args.videosURL.path)
    }

    let model = try loadCoreMLModel(from: args.modelURL)
    let fileManager = FileManager.default
    let items = try fileManager.contentsOfDirectory(at: args.videosURL, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])
    let videoFiles = items.filter { url in
        guard isMP4File(url) else { return false }
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
        return values?.isRegularFile == true
    }

    if videoFiles.isEmpty {
        print("No MP4 or MOV files found in \(args.videosURL.path)")
        return
    }

    for videoURL in videoFiles {
        do {
            let bestByLabel = try await detectObjectsInVideo(
                videoURL,
                model: model,
                threshold: args.threshold,
                intervalSeconds: args.intervalSeconds,
                maxSamples: args.maxSamples
            )
            if bestByLabel.isEmpty {
                print("\(videoURL.lastPathComponent): no detections")
            } else {
                let sorted = bestByLabel.sorted { $0.value.confidence > $1.value.confidence }
                let summary = sorted.map {
                    let confidence = String(format: "%.2f", $0.value.confidence)
                    let time = $0.value.time.isFinite ? String(format: "%.2f", $0.value.time) : "unknown"
                    return "\($0.key) (\(confidence) @ \(time)s)"
                }.joined(separator: ", ")
                print("\(videoURL.lastPathComponent): \(summary)")
            }
        } catch {
            printError("Failed to process \(videoURL.lastPathComponent): \(error)")
        }
    }
}

final class ModelStore {
    private let queue = DispatchQueue(label: "recognize.lsbackend.model")
    private var model: VNCoreMLModel

    init(model: VNCoreMLModel) {
        self.model = model
    }

    func withModel<T>(_ block: (VNCoreMLModel) throws -> T) rethrows -> T {
        return try queue.sync {
            try block(model)
        }
    }

    func updateModel(from url: URL) throws {
        let newModel = try loadCoreMLModel(from: url)
        queue.sync {
            model = newModel
        }
    }
}

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data
}

func parseHTTPRequest(from buffer: Data) -> (request: HTTPRequest, consumed: Int)? {
    let separator = Data("\r\n\r\n".utf8)
    guard let headerRange = buffer.range(of: separator) else {
        return nil
    }

    let headerData = buffer.subdata(in: 0..<headerRange.lowerBound)
    guard let headerText = String(data: headerData, encoding: .utf8) else {
        return nil
    }

    let lines = headerText.split(separator: "\r\n")
    guard let requestLine = lines.first else {
        return nil
    }
    let requestParts = requestLine.split(separator: " ")
    guard requestParts.count >= 2 else {
        return nil
    }

    var headers: [String: String] = [:]
    for line in lines.dropFirst() {
        if let colonIndex = line.firstIndex(of: ":") {
            let key = line[..<colonIndex].trimmingCharacters(in: .whitespaces).lowercased()
            let valueStart = line.index(after: colonIndex)
            let value = line[valueStart...].trimmingCharacters(in: .whitespaces)
            headers[String(key)] = String(value)
        }
    }

    let contentLength = Int(headers["content-length"] ?? "0") ?? 0
    let bodyStart = headerRange.upperBound
    let bodyEnd = bodyStart + contentLength
    guard buffer.count >= bodyEnd else {
        return nil
    }

    let body = buffer.subdata(in: bodyStart..<bodyEnd)
    let method = String(requestParts[0])
    let path = String(requestParts[1])
    let request = HTTPRequest(method: method, path: path, headers: headers, body: body)
    return (request, bodyEnd)
}

func httpStatusText(_ code: Int) -> String {
    switch code {
    case 200: return "OK"
    case 400: return "Bad Request"
    case 404: return "Not Found"
    case 500: return "Internal Server Error"
    default: return "OK"
    }
}

func makeHTTPResponse(status: Int, body: Data, contentType: String = "application/json") -> Data {
    let lines = [
        "HTTP/1.1 \(status) \(httpStatusText(status))",
        "Content-Length: \(body.count)",
        "Content-Type: \(contentType)",
        "Connection: close"
    ]
    let headerString = lines.joined(separator: "\r\n") + "\r\n\r\n"
    let headerData = headerString.data(using: .utf8) ?? Data()
    var response = Data()
    response.append(headerData)
    response.append(body)
    return response
}

func clampPercent(_ value: Double) -> Double {
    return min(100.0, max(0.0, value))
}

func normalizePredictionLabel(_ label: String) -> String {
    let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return label }
    let lower = trimmed.lowercased()
    return lower.prefix(1).uppercased() + lower.dropFirst()
}

func parseProjectID(from payload: [String: Any], fallback: Int?) -> Int? {
    if let projectID = payload["project_id"] as? Int {
        return projectID
    }
    if let projectID = payload["project_id"] as? String, let value = Int(projectID) {
        return value
    }
    if let project = payload["project"] as? Int {
        return project
    }
    if let project = payload["project"] as? String, let value = Int(project) {
        return value
    }
    if let project = payload["project"] as? [String: Any] {
        if let value = project["id"] as? Int {
            return value
        }
        if let value = project["id"] as? String, let id = Int(value) {
            return id
        }
    }
    return fallback
}

func looksLikeJWT(_ token: String) -> Bool {
    return token.split(separator: ".").count == 3
}

func base64URLDecode(_ value: String) -> Data? {
    var base64 = value.replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
    let padding = 4 - (base64.count % 4)
    if padding < 4 {
        base64 += String(repeating: "=", count: padding)
    }
    return Data(base64Encoded: base64)
}

func jwtPayload(_ token: String) -> [String: Any]? {
    let parts = token.split(separator: ".")
    guard parts.count == 3 else { return nil }
    guard let data = base64URLDecode(String(parts[1])) else { return nil }
    return (try? JSONSerialization.jsonObject(with: data, options: [])) as? [String: Any]
}

func authorizationHeaderValue(for token: String?) -> String? {
    guard let token, !token.isEmpty else { return nil }
    if token.contains(" ") {
        return token
    }
    if looksLikeJWT(token) {
        return "Bearer \(token)"
    }
    return "Token \(token)"
}

func performRequest(_ request: URLRequest) throws -> (Data, HTTPURLResponse) {
    let semaphore = DispatchSemaphore(value: 0)
    var resultData: Data?
    var resultResponse: URLResponse?
    var resultError: Error?

    URLSession.shared.dataTask(with: request) { data, response, error in
        resultData = data
        resultResponse = response
        resultError = error
        semaphore.signal()
    }.resume()

    semaphore.wait()

    if let error = resultError {
        throw error
    }
    guard let response = resultResponse as? HTTPURLResponse else {
        throw RecognizeError.invalidArguments("No HTTP response from \(request.url?.absoluteString ?? "request")")
    }
    guard let data = resultData else {
        throw RecognizeError.invalidArguments("Empty response from \(request.url?.absoluteString ?? "request")")
    }
    return (data, response)
}

func refreshAccessToken(refreshToken: String, labelStudioURL: URL) throws -> String {
    let baseString = labelStudioURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    guard let url = URL(string: "\(baseString)/api/token/refresh/") else {
        throw RecognizeError.invalidArguments("Invalid Label Studio refresh URL")
    }
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    let payload = ["refresh": refreshToken]
    request.httpBody = try JSONSerialization.data(withJSONObject: payload, options: [])
    let (data, response) = try performRequest(request)
    guard (200...299).contains(response.statusCode) else {
        throw RecognizeError.invalidArguments("Token refresh failed: HTTP \(response.statusCode)")
    }
    guard let json = (try? JSONSerialization.jsonObject(with: data, options: [])) as? [String: Any],
          let access = json["access"] as? String else {
        throw RecognizeError.invalidArguments("Token refresh response missing access token")
    }
    return access
}

func downloadData(from url: URL, token: String?, labelStudioURL: URL?) throws -> (Data, HTTPURLResponse) {
    var tokenToUse = token
    if let token, looksLikeJWT(token),
       let payload = jwtPayload(token),
       let tokenType = payload["token_type"] as? String,
       tokenType == "refresh",
       let baseURL = labelStudioURL {
        tokenToUse = try refreshAccessToken(refreshToken: token, labelStudioURL: baseURL)
    }

    var request = URLRequest(url: url)
    if let headerValue = authorizationHeaderValue(for: tokenToUse) {
        request.setValue(headerValue, forHTTPHeaderField: "Authorization")
    }

    let (data, response) = try performRequest(request)
    if response.statusCode == 401,
       let token,
       looksLikeJWT(token),
       let baseURL = labelStudioURL {
        let access = try refreshAccessToken(refreshToken: token, labelStudioURL: baseURL)
        var retry = URLRequest(url: url)
        retry.setValue(authorizationHeaderValue(for: access), forHTTPHeaderField: "Authorization")
        return try performRequest(retry)
    }
    return (data, response)
}

func extractZipData(_ data: Data, to directory: URL) throws {
    let zipURL = directory.appendingPathComponent("export.zip")
    try data.write(to: zipURL, options: .atomic)
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
    process.arguments = ["-o", zipURL.path, "-d", directory.path]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        throw RecognizeError.invalidArguments("Unzip failed: \(output)")
    }
}

func locateCocoJSON(in directory: URL) throws -> URL {
    let fileManager = FileManager.default
    guard let enumerator = fileManager.enumerator(at: directory, includingPropertiesForKeys: nil) else {
        throw RecognizeError.invalidArguments("Unable to scan export directory")
    }
    var candidates: [URL] = []
    for case let fileURL as URL in enumerator {
        if fileURL.pathExtension.lowercased() == "json" {
            candidates.append(fileURL)
        }
    }
    if let result = candidates.first(where: { $0.lastPathComponent == "result.json" }) {
        return result
    }
    if let result = candidates.first(where: { $0.lastPathComponent == "annotations.json" }) {
        return result
    }
    if let first = candidates.first {
        return first
    }
    throw RecognizeError.invalidArguments("No JSON export found in Label Studio export")
}

func ensureResultJSON(in directory: URL) throws {
    let targetURL = directory.appendingPathComponent("result.json")
    if FileManager.default.fileExists(atPath: targetURL.path) {
        return
    }
    let jsonURL = try locateCocoJSON(in: directory)
    if FileManager.default.fileExists(atPath: targetURL.path) {
        try FileManager.default.removeItem(at: targetURL)
    }
    try FileManager.default.copyItem(at: jsonURL, to: targetURL)
    if !FileManager.default.fileExists(atPath: targetURL.path) {
        throw RecognizeError.invalidArguments("Export JSON copy failed")
    }
}

func listExportFiles(in directory: URL, limit: Int = 50) -> String {
    guard let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil) else {
        return "(unable to list files)"
    }
    var items: [String] = []
    for case let fileURL as URL in enumerator {
        items.append(fileURL.path)
        if items.count >= limit {
            break
        }
    }
    return items.isEmpty ? "(no files found)" : items.joined(separator: "\n")
}

func exportCocoFromLabelStudio(projectID: Int, config: BackendArguments) throws -> URL {
    guard let baseURL = config.labelStudioURL else {
        throw RecognizeError.invalidArguments("Missing --label-studio-url for training export")
    }
    let baseString = baseURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    let urlString = "\(baseString)/api/projects/\(projectID)/export?exportType=COCO"
    guard let exportURL = URL(string: urlString) else {
        throw RecognizeError.invalidArguments("Invalid export URL")
    }

    let (data, response) = try downloadData(from: exportURL, token: config.labelStudioToken, labelStudioURL: config.labelStudioURL)
    guard (200...299).contains(response.statusCode) else {
        throw RecognizeError.invalidArguments("Label Studio export failed: HTTP \(response.statusCode)")
    }

    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("recognize_ls_export_\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

    if data.starts(with: [0x50, 0x4B]) {
        try extractZipData(data, to: tempDir)
        try ensureResultJSON(in: tempDir)
        return tempDir
    }

    if let json = (try? JSONSerialization.jsonObject(with: data, options: [])) as? [String: Any] {
        if json["annotations"] != nil && json["images"] != nil {
            let targetURL = tempDir.appendingPathComponent("result.json")
            try data.write(to: targetURL, options: .atomic)
            return tempDir
        }
        if let downloadURL = json["download_url"] as? String ?? json["url"] as? String {
            let absolute: URL?
            if let url = URL(string: downloadURL), url.scheme != nil {
                absolute = url
            } else {
                let baseString = baseURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                let trimmedPath = downloadURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                absolute = URL(string: "\(baseString)/\(trimmedPath)")
            }
            if let absolute {
                let (downloaded, dlResponse) = try downloadData(from: absolute, token: config.labelStudioToken, labelStudioURL: config.labelStudioURL)
                guard (200...299).contains(dlResponse.statusCode) else {
                    throw RecognizeError.invalidArguments("Label Studio export download failed: HTTP \(dlResponse.statusCode)")
                }
                if downloaded.starts(with: [0x50, 0x4B]) {
                    try extractZipData(downloaded, to: tempDir)
                    try ensureResultJSON(in: tempDir)
                    return tempDir
                }
                let targetURL = tempDir.appendingPathComponent("result.json")
                try downloaded.write(to: targetURL, options: .atomic)
                return tempDir
            }
        }
        if let detail = json["detail"] as? String {
            throw RecognizeError.invalidArguments("Label Studio export did not return COCO data: \(detail)")
        }
    }

    let targetURL = tempDir.appendingPathComponent("result.json")
    try data.write(to: targetURL, options: .atomic)
    if !FileManager.default.fileExists(atPath: targetURL.path) {
        let listing = listExportFiles(in: tempDir)
        throw RecognizeError.invalidArguments("Export missing result.json. Files:\n\(listing)")
    }
    return tempDir
}

func defaultTrainOutputURL(for modelURL: URL) -> URL {
    if modelURL.pathExtension == "mlmodelc" {
        return modelURL.deletingPathExtension().appendingPathExtension("mlmodel")
    }
    return modelURL
}

final class TrainingController {
    private let queue = DispatchQueue(label: "recognize.lsbackend.training")
    private var inProgress = false
    private let config: BackendArguments
    private let modelStore: ModelStore

    init(config: BackendArguments, modelStore: ModelStore) {
        self.config = config
        self.modelStore = modelStore
    }

    func enqueueTraining(body: Data) throws -> String {
        let payload = (try? JSONSerialization.jsonObject(with: body, options: [])) as? [String: Any] ?? [:]
        guard let projectID = parseProjectID(from: payload, fallback: config.projectID) else {
            throw RecognizeError.invalidArguments("Missing project_id for training")
        }
        guard config.labelStudioURL != nil else {
            throw RecognizeError.invalidArguments("Missing --label-studio-url for training export")
        }
        var shouldStart = false
        queue.sync {
            if !inProgress {
                inProgress = true
                shouldStart = true
            }
        }
        guard shouldStart else {
            return "training already in progress"
        }

        queue.async { [config, modelStore] in
            do {
                print("Starting training for Label Studio project \(projectID)")
                let cocoDir = try exportCocoFromLabelStudio(projectID: projectID, config: config)
                let outputURL = config.trainOutputURL ?? defaultTrainOutputURL(for: config.modelURL)
                try trainModel(cocoDir: cocoDir, outputModelURL: outputURL, imagesRootOverride: config.dataRoot)
                try modelStore.updateModel(from: outputURL)
                print("Training complete. Model updated at \(outputURL.path)")
            } catch {
                printError("Training failed: \(error)")
            }
            self.inProgress = false
        }
        return "training started"
    }
}

func downloadImage(from url: URL, token: String?) throws -> URL {
    let (data, response) = try downloadData(from: url, token: token, labelStudioURL: nil)
    guard (200...299).contains(response.statusCode) else {
        throw RecognizeError.invalidArguments("Download failed: HTTP \(response.statusCode)")
    }

    let fileName = url.lastPathComponent.isEmpty ? UUID().uuidString : url.lastPathComponent
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("recognize_ls_backend", isDirectory: true)
    if !FileManager.default.fileExists(atPath: tempDir.path) {
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }
    let destURL = tempDir.appendingPathComponent(fileName)
    try data.write(to: destURL, options: .atomic)
    return destURL
}

func resolveImageURL(from value: Any, config: BackendArguments) throws -> URL {
    guard let rawString = value as? String else {
        throw RecognizeError.invalidArguments("Expected image URL string in task data")
    }

    let decoded = rawString.removingPercentEncoding ?? rawString
    if decoded.hasPrefix("http://") || decoded.hasPrefix("https://") {
        guard let url = URL(string: decoded) else {
            throw RecognizeError.invalidArguments("Invalid image URL: \(decoded)")
        }
        return try downloadImage(from: url, token: config.labelStudioToken)
    }

    if decoded.hasPrefix("file://"), let url = URL(string: decoded) {
        return url
    }

    if decoded.hasPrefix("/") {
        if decoded.contains("/data/local-files/") {
            if let components = URLComponents(string: "http://localhost\(decoded)"),
               let dValue = components.queryItems?.first(where: { $0.name == "d" })?.value?.removingPercentEncoding {
                if let root = config.dataRoot {
                    var relative = dValue
                    if relative.hasPrefix("local-files/") {
                        relative = String(relative.dropFirst("local-files/".count))
                    }
                    if root.lastPathComponent == "images", relative.hasPrefix("images/") {
                        relative = String(relative.dropFirst("images/".count))
                    }
                    return root.appendingPathComponent(relative)
                }
                if let base = config.labelStudioURL {
                    let baseString = base.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                    let urlString = "\(baseString)\(decoded)"
                    if let url = URL(string: urlString) {
                        return try downloadImage(from: url, token: config.labelStudioToken)
                    }
                }
            }
        }
        if config.allowAbsolutePaths {
            return URL(fileURLWithPath: decoded)
        }
        if let root = config.dataRoot {
            let trimmed = decoded.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            return root.appendingPathComponent(trimmed)
        }
        return URL(fileURLWithPath: decoded)
    }

    if let root = config.dataRoot {
        return root.appendingPathComponent(decoded)
    }
    return URL(fileURLWithPath: decoded)
}

func handlePredictRequest(body: Data, config: BackendArguments, model: VNCoreMLModel) throws -> [[String: Any]] {
    let json = try JSONSerialization.jsonObject(with: body, options: [])
    guard let payload = json as? [String: Any] else {
        throw RecognizeError.invalidArguments("Invalid JSON payload")
    }

    let tasks = payload["tasks"] as? [[String: Any]] ?? []
    var predictions: [[String: Any]] = []

    for task in tasks {
        let data = task["data"] as? [String: Any]
        let imageValue = data?[config.dataKey]
        var results: [[String: Any]] = []
        var topScore: Float = 0

        if let imageValue {
            do {
                let imageURL = try resolveImageURL(from: imageValue, config: config)
                let cgImage = try loadCGImage(from: imageURL)
                let detections = try detectObjects(in: cgImage, model: model, threshold: config.threshold)
                let imageWidth = cgImage.width
                let imageHeight = cgImage.height
                for detection in detections {
                    let box = detection.boundingBox
                    let x = clampPercent(Double(box.minX) * 100.0)
                    let y = clampPercent((1.0 - Double(box.maxY)) * 100.0)
                    let width = clampPercent(Double(box.width) * 100.0)
                    let height = clampPercent(Double(box.height) * 100.0)
                    let label = normalizePredictionLabel(detection.label)
                    let result: [String: Any] = [
                        "from_name": config.fromName,
                        "to_name": config.toName,
                        "type": "rectanglelabels",
                        "value": [
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "rectanglelabels": [label]
                        ],
                        "score": detection.confidence,
                        "original_width": imageWidth,
                        "original_height": imageHeight,
                        "image_rotation": 0
                    ]
                    results.append(result)
                    if detection.confidence > topScore {
                        topScore = detection.confidence
                    }
                }
            } catch {
                printError("Prediction error: \(error)")
            }
        }

        let prediction: [String: Any] = [
            "result": results,
            "score": topScore,
            "model_version": config.modelVersion
        ]
        predictions.append(prediction)
    }

    return predictions
}

final class HTTPConnectionHandler {
    private let connection: NWConnection
    private let config: BackendArguments
    private let modelStore: ModelStore
    private let trainingController: TrainingController
    private let queue: DispatchQueue
    private var buffer = Data()
    var onClose: (() -> Void)?

    init(connection: NWConnection, config: BackendArguments, modelStore: ModelStore, trainingController: TrainingController, queue: DispatchQueue) {
        self.connection = connection
        self.config = config
        self.modelStore = modelStore
        self.trainingController = trainingController
        self.queue = queue
    }

    func start() {
        connection.start(queue: queue)
        receive()
    }

    private func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            if let data {
                self.buffer.append(data)
            }

            if let (request, consumed) = parseHTTPRequest(from: self.buffer) {
                self.buffer.removeSubrange(0..<consumed)
                self.respond(to: request)
                return
            }

            if error != nil || isComplete {
                self.connection.cancel()
                self.onClose?()
                return
            }

            self.receive()
        }
    }

    private func respond(to request: HTTPRequest) {
        var path = request.path.split(separator: "?").first.map(String.init) ?? request.path
        if path.count > 1 && path.hasSuffix("/") {
            path.removeLast()
        }
        print("HTTP \(request.method) \(path)")
        let responseData: Data

        do {
            switch (request.method, path) {
            case ("GET", "/health"):
                let body = try JSONSerialization.data(withJSONObject: ["status": "ok"], options: [])
                responseData = makeHTTPResponse(status: 200, body: body)
            case ("POST", "/setup"):
                let body = try JSONSerialization.data(withJSONObject: ["model_version": config.modelVersion], options: [])
                responseData = makeHTTPResponse(status: 200, body: body)
            case ("POST", "/predict"):
                let predictions = try modelStore.withModel { model in
                    try handlePredictRequest(body: request.body, config: config, model: model)
                }
                let body = try JSONSerialization.data(withJSONObject: ["results": predictions], options: [])
                responseData = makeHTTPResponse(status: 200, body: body)
            case ("POST", "/train"), ("GET", "/train"), ("POST", "/webhook"):
                print("Received /train request (\(request.body.count) bytes)")
                let message: String
                do {
                    message = try trainingController.enqueueTraining(body: request.body)
                } catch {
                    let errorMessage = "error: \(error)"
                    printError("Training request failed: \(errorMessage)")
                    message = errorMessage
                }
                let body = try JSONSerialization.data(withJSONObject: ["status": message], options: [])
                responseData = makeHTTPResponse(status: 200, body: body)
            default:
                let body = try JSONSerialization.data(withJSONObject: ["error": "Not found"], options: [])
                responseData = makeHTTPResponse(status: 404, body: body)
            }
        } catch {
            let body = (try? JSONSerialization.data(withJSONObject: ["error": "\(error)"], options: [])) ?? Data()
            responseData = makeHTTPResponse(status: 500, body: body)
        }

        connection.send(content: responseData, completion: .contentProcessed { _ in
            self.connection.cancel()
            self.onClose?()
        })
    }
}

func runLabelStudioBackend(_ args: BackendArguments) throws {
    let modelStore = ModelStore(model: try loadCoreMLModel(from: args.modelURL))
    guard let port = NWEndpoint.Port(rawValue: args.port) else {
        throw RecognizeError.invalidArguments("Invalid port: \(args.port)")
    }
    let listener = try NWListener(using: .tcp, on: port)
    let queue = DispatchQueue(label: "recognize.lsbackend")
    var activeHandlers: [ObjectIdentifier: HTTPConnectionHandler] = [:]
    let trainingController = TrainingController(config: args, modelStore: modelStore)

    listener.newConnectionHandler = { connection in
        let handler = HTTPConnectionHandler(
            connection: connection,
            config: args,
            modelStore: modelStore,
            trainingController: trainingController,
            queue: queue
        )
        let key = ObjectIdentifier(handler)
        activeHandlers[key] = handler
        handler.onClose = {
            activeHandlers.removeValue(forKey: key)
        }
        handler.start()
    }

    listener.stateUpdateHandler = { state in
        switch state {
        case .ready:
            print("Label Studio backend listening on http://\(args.host):\(args.port)")
        case .failed(let error):
            printError("Backend listener failed: \(error)")
        default:
            break
        }
    }

    listener.start(queue: queue)

    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)
    let shutdown = DispatchSemaphore(value: 0)
    let signalQueue = DispatchQueue(label: "recognize.lsbackend.signal")
    let sigint = DispatchSource.makeSignalSource(signal: SIGINT, queue: signalQueue)
    let sigterm = DispatchSource.makeSignalSource(signal: SIGTERM, queue: signalQueue)
    sigint.setEventHandler {
        shutdown.signal()
    }
    sigterm.setEventHandler {
        shutdown.signal()
    }
    sigint.resume()
    sigterm.resume()
    shutdown.wait()
}

func run() async throws {
    let command = try parseCommand(CommandLine.arguments)
    switch command {
    case .train(let cocoDir, let outputModel, let imagesRootOverride):
        try trainModel(cocoDir: cocoDir, outputModelURL: outputModel, imagesRootOverride: imagesRootOverride)
    case .detect(let args):
        try runDetect(args)
    case .detectVideo(let args):
        try await runDetectVideo(args)
    case .backend(let args):
        try runLabelStudioBackend(args)
    case .legacy(let parsed):
        try await runLegacy(parsed)
    }
}

@main
struct Recognize {
    static func main() async {
        do {
            try await run()
        } catch let error as RecognizeError {
            printError(error.description)
            exit(1)
        } catch {
            printError("Unexpected error: \(error)")
            exit(1)
        }
    }
}
