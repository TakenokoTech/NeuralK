package tech.takenoko.neuralk.entity

sealed class TextUnion {
    data class StringType(val value: String) : TextUnion()
    data class StringListType(val value: List<String>) : TextUnion()

    fun concatText() = when (this) {
        is TextUnion.StringListType -> value
        is TextUnion.StringType -> listOf(value)
    }.joinToString(" ") { it }
}
