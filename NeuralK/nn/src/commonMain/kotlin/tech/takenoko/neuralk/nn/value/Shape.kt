package tech.takenoko.neuralk.nn.value

data class Shape(val value: List<Int>) {
    val rows: Int get() = value.getOrNull(0) ?: 1
    val cols: Int get() = value.getOrNull(1) ?: 1

    constructor(vararg values: Int) : this(values.toList())

    override fun toString(): String = "Shape(${value.joinToString { it.toString() }})"
}
