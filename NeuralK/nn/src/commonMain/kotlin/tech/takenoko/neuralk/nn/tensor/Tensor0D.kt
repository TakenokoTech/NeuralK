package tech.takenoko.neuralk.nn.tensor

class Tensor0D(val data: Float) : Tensor() {
    override val shape = listOf(1)
    override val rows = 1
    override val cols = 1

    constructor(data: Double) : this(data.toFloat())
}
