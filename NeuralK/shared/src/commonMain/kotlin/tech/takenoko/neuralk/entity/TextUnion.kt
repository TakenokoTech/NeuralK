package tech.takenoko.neuralk.entity

sealed class TextUnion {
    data class StringType(val value: String) : TextUnion()
    data class StringListType(val value: List<String>) : TextUnion()

    fun concatText() = when (this) {
        is StringListType -> value
        is StringType -> listOf(value)
    }.joinToString(" ") { it }
}
