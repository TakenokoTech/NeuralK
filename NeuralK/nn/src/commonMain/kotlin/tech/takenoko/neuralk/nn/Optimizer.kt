package tech.takenoko.neuralk.nn

sealed class Optimizer {
    abstract fun update(weights: Tensor, gradients: Tensor): Tensor
}

class SGD(private val learningRate: Float) : Optimizer() {
    override fun update(weights: Tensor, gradients: Tensor) =
        weights - (gradients * Scalar(learningRate))
}
