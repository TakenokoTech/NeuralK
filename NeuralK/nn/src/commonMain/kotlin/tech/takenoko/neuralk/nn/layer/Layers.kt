package tech.takenoko.neuralk.nn.layer

import tech.takenoko.neuralk.nn.optimizer.Optimizer
import tech.takenoko.neuralk.nn.tensor.Tensor

@JvmInline
value class Layers(private val value: MutableList<Layer>) {

    constructor(vararg value: Layer) : this(value.toMutableList())

    fun isEmpty() = value.isEmpty()
    operator fun plusAssign(layer: Layer) = value.plusAssign(layer)

    internal fun forward(input: Tensor): Tensor {
        var currentOutput = input
        for (layer in value) currentOutput = layer.forward(currentOutput)
        return currentOutput
    }

    internal fun backward(gradOutput: Tensor) {
        var currentGrad = gradOutput
        for (layer in value.reversed()) currentGrad = layer.backward(currentGrad)
    }

    internal fun update(optimizer: Optimizer) {
        for (layer in value) {
            if (!layer.trainable) continue
            layer.update(optimizer)
        }
    }
}
