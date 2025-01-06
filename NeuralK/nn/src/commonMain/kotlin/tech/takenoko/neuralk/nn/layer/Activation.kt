package tech.takenoko.neuralk.nn.layer

import tech.takenoko.neuralk.nn.activation.Activation
import tech.takenoko.neuralk.nn.optimizer.Optimizer
import tech.takenoko.neuralk.nn.tensor.Tensor

class Activation(private val activation: Activation) : Layer() {
    override val trainable = false
    override fun initialize(input: Tensor) = Unit
    override fun forward(input: Tensor): Tensor = activation.forward(input)
    override fun backward(output: Tensor): Tensor = output
    override fun update(optimizer: Optimizer) = Unit
}
