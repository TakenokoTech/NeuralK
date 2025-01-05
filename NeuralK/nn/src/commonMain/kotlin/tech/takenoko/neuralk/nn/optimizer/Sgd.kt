package tech.takenoko.neuralk.nn.optimizer

import tech.takenoko.neuralk.nn.tensor.Tensor
import tech.takenoko.neuralk.nn.tensor.Tensor0D

class Sgd(val learningRate: Double = 0.01, val momentum: Double = 0.0) : Optimizer() {
    private val velocities = mutableMapOf<Parameter, Tensor>()

    // ref: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD
    override fun update(parameter: Parameter, gradient: Tensor) {
        val velocity = velocities.getOrPut(parameter) { Tensor.zeros(gradient.shape) }
        val updatedVelocity = velocity * Tensor0D(momentum) + gradient * Tensor0D(learningRate)
        velocities[parameter] = updatedVelocity
        parameter.update(-updatedVelocity)
    }
}
