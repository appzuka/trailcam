import AVFoundation
import Foundation
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
Usage: recognize <directory> [options]

Options:
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

func parseArguments(_ args: [String]) throws -> ParsedArguments {
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

func run() async throws {
    let parsed = try parseArguments(CommandLine.arguments)
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
