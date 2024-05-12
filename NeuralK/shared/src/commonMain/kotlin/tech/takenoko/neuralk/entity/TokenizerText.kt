package tech.takenoko.neuralk.entity

sealed class TokenizerText {
    data class StringType(val value: String) : TokenizerText()
    data class StringListType(val value: List<String>) : TokenizerText()

    fun concatText() = when (this) {
        is StringListType -> value
        is StringType -> listOf(value)
    }.joinToString(" ") { it }
}
