//
//  AppDelegate.swift
//  sample
//
//  Created by Takenoko on 2024/05/01.
//

import SwiftUI
import NeuralK

class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil
    ) -> Bool {
        print(IOSPlatform().name)
        return true
    }
}
