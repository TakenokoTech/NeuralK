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
    override val trainable: Boolean = true,
) : Layer() {
    private lateinit var weights: Weight
    private lateinit var bias: Bias
    private lateinit var inputShape: Shape
    private lateinit var outputShape: Shape
    private lateinit var savedInput: Tensor2D
    private lateinit var gradient: Gradient

    override fun initialize(input: Tensor) {
        if (!::inputShape.isInitialized) {
            inputShape = input.shape
            outputShape = Shape(*inputShape.value.dropLast(1).toIntArray(), units)
        }
        if (!::weights.isInitialized) {
            weights = initWeights ?: Weight(Tensor2D(inputShape.value.last(), units, data = 0.5))
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
        return input * weights.getValue() + bias.getValue()
    }

    override fun backward(output: Tensor): Tensor {
        require(output is Tensor2D)
        gradient = Gradient(savedInput, output)
        return weights.getValue().transpose() * output
    }

    override fun update(optimizer: Optimizer) {
//        println("gradWeights: ${weights.getValue().toList()} -> ${gradient.weights.toList()}")
//        println("gradBias: ${bias.getValue().toList()} -> ${gradient.bias.toList()}")
        optimizer.update(weights, gradient.weights)
        optimizer.update(bias, gradient.bias)
    }

    private class Gradient(val input: Tensor2D, val output: Tensor2D) {
        val weights: Tensor2D = (input.transpose() * output) as Tensor2D
        val bias: Tensor1D = output.sum()
    }

    class Weight(private var value: Tensor2D) : Optimizer.Parameter {
        override fun getValue() = value
        override fun update(velocity: Tensor) {
            require(velocity is Tensor2D)
            value = (value + velocity) as Tensor2D
        }
    }

    class Bias(private var value: Tensor1D) : Optimizer.Parameter {
        override fun getValue() = value
        override fun update(velocity: Tensor) {
            require(velocity is Tensor1D)
            value = (value + velocity) as Tensor1D
        }
    }
}
