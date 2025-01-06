package tech.takenoko.neuralk.nn.optimizer

import tech.takenoko.neuralk.nn.tensor.Tensor

sealed class Optimizer {
    internal abstract fun update(parameter: Parameter, gradient: Tensor)

    interface Parameter {
        fun getValue(): Tensor
        fun update(velocity: Tensor)
    }
}
