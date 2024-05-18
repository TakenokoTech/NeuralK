package tech.takenoko.neuralk.load

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.decodeFromByteArray
import kotlinx.serialization.protobuf.ProtoBuf
import tech.takenoko.neuralk.entity.LoaderModel

class Loader {
    @OptIn(ExperimentalSerializationApi::class)
    fun load(rawData: ByteArray) {
        println(rawData.size)
        val model = ProtoBuf.decodeFromByteArray<LoaderModel>(rawData)
        println(model)
    }
}
