package tech.takenoko.neuralk.nn.activation

import tech.takenoko.neuralk.nn.tensor.Tensor

sealed class Activation {
    abstract fun forward(input: Tensor): Tensor
}
