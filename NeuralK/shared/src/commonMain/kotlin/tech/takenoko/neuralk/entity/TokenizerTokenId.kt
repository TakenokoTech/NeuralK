package tech.takenoko.neuralk.entity

sealed class TokenizerTokenId {
    data class IntType(val value: Int) : TokenizerTokenId()
    data class IntListType(val value: List<Int>) : TokenizerTokenId()
    data class TensorType(val value: Tensor) : TokenizerTokenId() {
        class Tensor
    }

    fun concatTokenIds() = when (this) {
        is IntListType -> value
        is IntType -> listOf(value)
        is TensorType -> TODO()
    }
}
