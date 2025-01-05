package tech.takenoko.neuralk.nn.optimizer

import tech.takenoko.neuralk.nn.tensor.Tensor

class Sgd(val learningRate: Double = 0.01, val momentum: Double = 0.0) : Optimizer() {
//    override fun update(weights: Tensor, gradients: Tensor) =
//        weights - (gradients * Tensor0D(learningRate))

    override fun update(parameter: Parameter, gradient: Tensor) {
        parameter.update(gradient, learningRate)
    }
}
