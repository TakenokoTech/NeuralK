package tech.takenoko.neuralk.model

sealed class Optimizer {
    abstract fun update(weights: Tensor, gradients: Tensor): Tensor
}

class SGD(private val learningRate: Float) : Optimizer() {
    override fun update(weights: Tensor, gradients: Tensor) =
        weights - (gradients * Scalar(learningRate))
}
