package tech.takenoko.neuralk.nn.optimizer

import tech.takenoko.neuralk.nn.tensor.Tensor
import tech.takenoko.neuralk.nn.tensor.Tensor0D
import kotlin.math.pow

class Adam(
    val learningRate: Double = 0.001,
    private val beta1: Double = 0.9,
    private val beta2: Double = 0.999,
    private val epsilon: Double = 1e-07,
) : Optimizer() {
    private val times = mutableMapOf<Parameter, Int>()
    private val momentums: MutableMap<Parameter, Tensor> = mutableMapOf()
    private val velocities: MutableMap<Parameter, Tensor> = mutableMapOf()

    /**
     * ref: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
     *
     * mt = β1 * mt-1 + (1 - β1) * g
     * vt = β2 * vt-1 + (1 - β2) * g²
     * mtHat = mt / (1 - β1^t)
     * vtHat = vt / (1 - β2^t)
     * θt = θt-1 - η * mtHat / (√vtHat + ε)
     */
    override fun update(parameter: Parameter, gradient: Tensor) {
        val t = times.getOrPut(parameter) { 0 } + 1
        val m = momentums.getOrPut(parameter) { Tensor.zeros(gradient.shape) }
        val v = velocities.getOrPut(parameter) { Tensor.zeros(gradient.shape) }
        val mt = m * Tensor0D(beta1) + gradient * Tensor0D(1 - beta1)
        val vt = v * Tensor0D(beta2) + gradient * gradient * Tensor0D(1 - beta2)
        val mtHat = mt / Tensor0D(1 - beta1.pow(t))
        val vtHat = vt / Tensor0D(1 - beta2.pow(t))
        val updateValue = mtHat / (vtHat.sqrt() + Tensor0D(epsilon)) * Tensor0D(learningRate)
        times[parameter] = t
        momentums[parameter] = mt
        velocities[parameter] = vt
        parameter.update(-updateValue)
    }
}
