package tech.takenoko.neuralk.nn.layer

import tech.takenoko.neuralk.nn.Utils.requireEqual
import tech.takenoko.neuralk.nn.optimizer.Optimizer
import tech.takenoko.neuralk.nn.tensor.Tensor
import tech.takenoko.neuralk.nn.tensor.Tensor1D
import tech.takenoko.neuralk.nn.tensor.Tensor2D
import tech.takenoko.neuralk.nn.value.Shape

class Dense(
    private val units: Int,
    private val initWeights: Weight? = null,
    private val initBias: Bias? = null,
    override val trainable: Boolean = true
) : Layer() {
    private lateinit var weights: Weight
    private lateinit var bias: Bias
    private lateinit var inputShape: Shape
    private lateinit var savedInput: Tensor2D
    private lateinit var gradient: Gradient

    override fun initialize(input: Tensor) {
        if (!::inputShape.isInitialized) {
            inputShape = input.shape
        }
        if (!::weights.isInitialized) {
            weights = initWeights ?: Weight(Tensor2D(units, inputShape.rows, data = 0.5))
        }
        if (!::bias.isInitialized) {
            bias = initBias ?: Bias(Tensor1D(units, data = 0.5))
        }
    }

    override fun forward(input: Tensor): Tensor {
        initialize(input)
        require(input is Tensor2D)
        requireEqual(input.rows, inputShape.rows)
        this.savedInput = input
        return weights.getValue() * input + bias.getValue()
    }

    override fun backward(output: Tensor): Tensor {
        require(output is Tensor2D)
        gradient = Gradient(savedInput, output)
        return weights.getValue().transpose() * output
    }

    override fun update(optimizer: Optimizer) {
        optimizer.update(weights, gradient.weights)
        optimizer.update(bias, gradient.bias)
//        println("gradWeights: ${gradient.weights.toList()}, gradBias: ${gradient.bias.toList()}")
//        println("weights: ${weights.getData().toList()}, bias: ${bias.getData().toList()}")
    }

    private class Gradient(input: Tensor2D, output: Tensor2D) {
        val weights: Tensor2D = (output * input.transpose()) as Tensor2D
        val bias: Tensor1D = output.sum()
    }

    class Weight(private var value: Tensor2D) : Optimizer.Parameter {
        constructor(value: Array<Array<Float>>) : this(Tensor2D(value))

        override fun getValue() = value
        override fun update(velocity: Tensor) {
            require(velocity is Tensor2D)
            value = (value + velocity) as Tensor2D
        }
    }

    class Bias(private var value: Tensor1D) : Optimizer.Parameter {
        constructor(value: Array<Float>) : this(Tensor1D(value))

        override fun getValue() = value
        override fun update(velocity: Tensor) {
            require(velocity is Tensor1D)
            value = (value + velocity) as Tensor1D
        }
    }
}
