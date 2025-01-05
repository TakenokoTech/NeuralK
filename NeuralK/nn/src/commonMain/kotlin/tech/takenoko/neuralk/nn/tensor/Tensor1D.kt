package tech.takenoko.neuralk.nn.tensor

import tech.takenoko.neuralk.nn.value.Shape

class Tensor1D(val data: Array<Float>) : Tensor() {
    override val shape = Shape(data.size)

    constructor(size: Int, data: Double) : this(Array(size) { data.toFloat() })
    constructor(shape: Shape, data: Double) : this(shape.value[0], data)
}
