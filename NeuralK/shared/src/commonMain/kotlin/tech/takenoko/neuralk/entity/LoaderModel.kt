package tech.takenoko.neuralk.entity

import kotlinx.serialization.Serializable
import kotlinx.serialization.protobuf.ProtoNumber

@Serializable
data class LoaderModel(
    @ProtoNumber(1) val a1: List<Map<String, String>>,
//    @ProtoNumber(2) val a2: String,
//    @ProtoNumber(6) val a2: String,
)

//@Serializable
//data class NodeDef(
//    @ProtoNumber(8) val a0: FunctionDefLibrary,
//)
//
//@Serializable
//data class FunctionDefLibrary(
//    @ProtoNumber(10) val a0: String,
//)
