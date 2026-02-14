// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Recognize",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "recognize", targets: ["Recognize"])
    ],
    targets: [
        .executableTarget(
            name: "Recognize"
        )
    ]
)
