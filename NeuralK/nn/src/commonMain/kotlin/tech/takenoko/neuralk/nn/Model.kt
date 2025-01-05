package tech.takenoko.neuralk.nn

class Model(private val layers: List<Layer>, private val optimizer: SGD) {
    fun fit(inputs: List<Tensor>, labels: List<Tensor>, epochs: Int) {
        require(inputs.size == labels.size) { "Inputs and labels must have the same size" }
        for (epoch in 1..epochs) {
            var totalLoss = 0f
            for ((input, label) in inputs.zip(labels).shuffled()) {
                val predictions = forward(input)
                totalLoss += mseLoss(predictions, label)
                val grad = mseLossGrad(predictions, label)
                backward(grad)
            }
            if (epoch % 10 == 0) println("Epoch $epoch: Loss = ${totalLoss / inputs.size}")
        }
    }

    fun evaluate(inputs: List<Tensor>, labels: List<Tensor>): Float {
        require(inputs.size == labels.size) { "Inputs and labels must have the same size" }
        var totalLoss = 0f
        for ((input, label) in inputs.zip(labels)) {
            val predictions = forward(input)
            totalLoss += mseLoss(predictions, label)
        }
        return totalLoss / inputs.size
    }

    fun predict(input: Tensor): Tensor = forward(input)

    private fun forward(input: Tensor): Tensor {
        var currentOutput = input
        for (layer in layers) currentOutput = layer.forward(currentOutput)
        return currentOutput
    }

    private fun backward(gradOutput: Tensor) {
        var currentGrad = gradOutput
        for (layer in layers.reversed()) currentGrad = layer.backward(currentGrad)
    }

    // TODO: Lossクラスを作る
    private fun mseLoss(predictions: Tensor, labels: Tensor): Float {
        val diff = (predictions - labels) as Tensor2D
        val sum = diff.data.sumOf { row -> row.sumOf { it.toDouble() * it } }.toFloat()
        return sum / (diff.rows * diff.cols)
    }

    private fun mseLossGrad(predictions: Tensor, labels: Tensor): Tensor {
        val diff = (predictions - labels) as Tensor2D
        return diff * Scalar(2f / (diff.rows * diff.cols))
    }
}
