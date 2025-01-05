package tech.takenoko.neuralk.nn.optimizer

import tech.takenoko.neuralk.nn.tensor.Tensor

sealed class Optimizer {
    abstract fun update(parameter: Parameter, gradient: Tensor)

    interface Parameter {
        fun getData(): Tensor
        fun update(grad: Tensor, learningRate: Double)
    }
}
