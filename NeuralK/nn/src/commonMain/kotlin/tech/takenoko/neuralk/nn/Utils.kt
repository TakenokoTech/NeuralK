package tech.takenoko.neuralk.nn

object Utils {
    fun requireEqual(self: Any, other: Any) {
        require(self == other) { "Expected $self to be equal to $other" }
    }
}
