package tech.takenoko.neuralk

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform