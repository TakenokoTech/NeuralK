//
//  sampleApp.swift
//  sample
//
//  Created by Takenoko on 2024/05/01.
//

import SwiftUI

@main
struct sampleApp: App {
    
    @UIApplicationDelegateAdaptor (AppDelegate.self) var appDelegate
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
