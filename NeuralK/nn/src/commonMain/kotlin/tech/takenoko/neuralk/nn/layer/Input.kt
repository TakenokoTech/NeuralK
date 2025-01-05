package tech.takenoko.neuralk.nn.layer

import tech.takenoko.neuralk.nn.optimizer.Optimizer
import tech.takenoko.neuralk.nn.tensor.Tensor
import tech.takenoko.neuralk.nn.value.Shape

class Input(private val shape: Shape) : Layer() {
    override val trainable = false
    override fun initialize(input: Tensor) = Unit
    override fun forward(input: Tensor): Tensor {
        require(input.shape == shape) { "Input shape must be $shape" }
        return input
    }

    override fun backward(output: Tensor) = output
    override fun update(optimizer: Optimizer) = Unit
}
