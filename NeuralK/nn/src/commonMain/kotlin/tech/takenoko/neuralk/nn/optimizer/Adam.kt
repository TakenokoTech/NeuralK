package tech.takenoko.neuralk.nn.optimizer

import tech.takenoko.neuralk.nn.tensor.Tensor

class Adam(
    val learningRate: Double = 0.001,
    val beta1: Double = 0.9,
    val beta2: Double = 0.999,
    val epsilon: Double = 1e-07,
) : Optimizer() {
    override fun update(parameter: Parameter, gradient: Tensor) {
        TODO("Not yet implemented")
    }
}
