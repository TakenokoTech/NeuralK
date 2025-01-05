package tech.takenoko.neuralk.nn.layer

import tech.takenoko.neuralk.nn.optimizer.Optimizer
import tech.takenoko.neuralk.nn.tensor.Tensor

sealed class Layer {
    abstract val trainable: Boolean
    abstract fun initialize(input: Tensor)
    abstract fun forward(input: Tensor): Tensor
    abstract fun backward(output: Tensor): Tensor
    abstract fun update(optimizer: Optimizer)
}
