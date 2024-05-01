// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "NeuralK",
    products: [
        .library(
            name: "NeuralK-debug",
            targets: ["NeuralK-debug"]),
    ],
    targets: [
        .binaryTarget(
            name: "NeuralK-debug",
            path: "NeuralK/shared/build/XCFrameworks/debug/NeuralK.xcframework"),
    ]
)
