package tech.takenoko.neuralk.nn.tensor

import tech.takenoko.neuralk.nn.value.Shape

class Tensor0D(val data: Float) : Tensor() {
    override val shape = Shape(1)

    constructor(data: Double) : this(data.toFloat())
}
