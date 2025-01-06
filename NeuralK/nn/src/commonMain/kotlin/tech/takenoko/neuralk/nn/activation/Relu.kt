package tech.takenoko.neuralk.nn.activation

import tech.takenoko.neuralk.nn.tensor.Tensor
import tech.takenoko.neuralk.nn.tensor.Tensor0D
import tech.takenoko.neuralk.nn.tensor.Tensor1D
import tech.takenoko.neuralk.nn.tensor.Tensor2D

data object Relu : Activation() {
    override fun forward(input: Tensor): Tensor = when (input) {
        is Tensor0D -> Tensor0D(input.data.takeIf { it > 0.0F } ?: 0.0F)
        is Tensor1D -> Tensor1D(input.data.map { it.takeIf { it > 0.0F } ?: 0.0F })
        is Tensor2D -> Tensor2D(input.data.map { it.map { it.takeIf { it > 0.0F } ?: 0.0F } })
    }
}
