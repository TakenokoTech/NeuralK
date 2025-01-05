package tech.takenoko.neuralk.nn

sealed class Layer {
    abstract fun initialize(input: Tensor)
    abstract fun forward(input: Tensor): Tensor
    abstract fun backward(gradOutput: Tensor): Tensor
}

class DenseLayer(
    private val units: Int,
    private val initWeights: Matrix? = null,
    private val initBias: Vector? = null
) : Layer() {
    private val learningRate = 0.001
    private var inputSize: Int = 0
    private lateinit var weights: Matrix
    private lateinit var bias: Vector
    private lateinit var input: Matrix

    override fun initialize(input: Tensor) {
        inputSize = input.rows
        if (!::weights.isInitialized) {
            weights = initWeights ?: Matrix(rows = units, cols = inputSize, data = 0.5)
        }
        if (!::bias.isInitialized) {
            bias = initBias ?: Vector(size = units, data = 0.5)
        }
    }

    override fun forward(input: Tensor): Tensor {
        initialize(input)
        require(input is Matrix)
        require(input.rows == inputSize) {
            "Expected input with $inputSize rows, but got ${input.rows}."
        }
        this.input = input
        return weights * input + bias
    }

    override fun backward(gradOutput: Tensor): Tensor {
        initialize(input)
        require(gradOutput is Matrix)
        val gradWeights = gradOutput * input.transpose()
        val gradBias = gradOutput.sum()
        val gradInput = weights.transpose() * gradOutput
//        println("gradOutput: ${gradOutput.toList()}")
//        println("gradWeights: ${gradWeights.toList()}")
//        println("gradBias: $gradBias")
//        println("gradInput: ${gradInput.toList()}")
        this.weights = (this.weights - gradWeights * Scalar(learningRate)) as Matrix
        this.bias = (this.bias - Scalar(gradBias * learningRate)) as Vector
//        println("weights: ${weights.toList()}, bias: ${bias.toList()}")
        return gradInput
    }
}
