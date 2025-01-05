package tech.takenoko.neuralk.nn.tensor

class Tensor1D(val data: Array<Float>) : Tensor() {
    override val shape = listOf(data.size)
    override val rows = shape[0]
    override val cols = 1

    constructor(size: Int, data: Double) : this(Array(size) { data.toFloat() })
}
