package tech.takenoko.neuralk.load

import kotlinx.serialization.ExperimentalSerializationApi

class Loader {
    @OptIn(ExperimentalSerializationApi::class)
    fun load(rawData: ByteArray) {
        // val model = ProtoBuf.decodeFromByteArray<LoaderModel>(rawData)
        println(rawData.size)
    }
}
