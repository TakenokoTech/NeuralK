package tech.takenoko.neuralk.entity

sealed class TokenIdUnion {
    data class IntType(val value: Int) : TokenIdUnion()
    data class IntListType(val value: List<Int>) : TokenIdUnion()
    data class TensorType(val value: Tensor) : TokenIdUnion() {
        class Tensor
    }

    fun concatTokenIds() = when (this) {
        is IntListType -> value
        is IntType -> listOf(value)
        is TensorType -> TODO()
    }
}
