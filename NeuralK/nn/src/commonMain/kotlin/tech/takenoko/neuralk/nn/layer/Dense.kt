package tech.takenoko.neuralk.nn.layer

import tech.takenoko.neuralk.nn.optimizer.Optimizer
import tech.takenoko.neuralk.nn.optimizer.Optimizer.Parameter
import tech.takenoko.neuralk.nn.tensor.Tensor
import tech.takenoko.neuralk.nn.tensor.Tensor0D
import tech.takenoko.neuralk.nn.tensor.Tensor1D
import tech.takenoko.neuralk.nn.tensor.Tensor2D

class Dense(
    private val units: Int,
    private val initWeights: Weight? = null,
    private val initBias: Bias? = null,
    override val trainable: Boolean = true
) : Layer() {
    private lateinit var input: Tensor2D
    private lateinit var weights: Weight
    private lateinit var bias: Bias
    private lateinit var gradient: Gradient
    private val inputSize: Int get() = input.rows

    override fun initialize(input: Tensor) {
        require(input is Tensor2D)
        this.input = input
        if (!::weights.isInitialized) {
            weights = initWeights ?: Weight(Tensor2D(rows = units, cols = inputSize, data = 0.5))
        }
        if (!::bias.isInitialized) {
            bias = initBias ?: Bias(Tensor1D(size = units, data = 0.5))
        }
    }

    override fun forward(input: Tensor): Tensor {
        initialize(input)
        require(input.rows == inputSize) {
            "Expected input with $inputSize rows, but got ${input.rows}."
        }
        return weights.getData() * input + bias.getData()
    }

    override fun backward(output: Tensor): Tensor {
        require(output is Tensor2D)
        gradient = Gradient(input, output)
        return weights.getData().transpose() * output
    }

    override fun update(optimizer: Optimizer) {
        optimizer.update(weights, gradient.weights)
        optimizer.update(bias, gradient.bias)
//        println("gradWeights: ${gradient.weights.toList()}, gradBias: ${gradient.bias.toList()}")
//        println("weights: ${weights.getData().toList()}, bias: ${bias.getData().toList()}")
    }

    class Gradient(input: Tensor2D, output: Tensor2D) {
        val weights: Tensor2D = (output * input.transpose()) as Tensor2D
        val bias: Tensor1D = output.sum()
    }

    class Weight(private var data: Tensor2D) : Parameter {
        override fun getData() = data
        override fun update(grad: Tensor, learningRate: Double) {
            require(grad is Tensor2D)
            data = (data - grad * Tensor0D(learningRate)) as Tensor2D
        }
    }

    class Bias(private var data: Tensor1D) : Parameter {
        override fun getData() = data
        override fun update(grad: Tensor, learningRate: Double) {
            require(grad is Tensor1D)
            data = (data - grad * Tensor0D(learningRate)) as Tensor1D
        }
    }
}
