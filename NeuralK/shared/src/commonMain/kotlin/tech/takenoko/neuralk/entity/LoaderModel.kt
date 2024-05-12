package tech.takenoko.neuralk.entity

import kotlinx.serialization.Serializable
import kotlinx.serialization.protobuf.ProtoNumber

@Serializable
data class LoaderModel(
    @ProtoNumber(3) val version: Byte
)
