package tech.takenoko.neuralk.nn.model

import tech.takenoko.neuralk.nn.layer.Layers
import tech.takenoko.neuralk.nn.optimizer.Optimizer
import tech.takenoko.neuralk.nn.tensor.Tensor
import tech.takenoko.neuralk.nn.tensor.Tensor0D
import tech.takenoko.neuralk.nn.tensor.Tensor2D

internal interface Trainer {
    var layers: Layers
    var optimizer: Optimizer?

    fun fit(inputs: List<Tensor>, labels: List<Tensor>, epochs: Int, shuffle: Boolean = true) {
        requireNotNull(optimizer) { "Optimizer must be set" }
        require(inputs.size == labels.size) { "Inputs and labels must have the same size" }
        for (epoch in 1..epochs) {
            var totalLoss = 0f
            val shuffledInputs = inputs.zip(labels).let { if (shuffle) it.shuffled() else it }
            for ((input, label) in shuffledInputs) {
                val predictions = layers.forward(input)
                totalLoss += mseLoss(predictions, label)
                val grad = mseLossGrad(predictions, label)
                layers.backward(grad)
                layers.update(optimizer!!)
            }
            if (epoch % 3 == 0) println("Epoch $epoch: Loss = ${totalLoss / inputs.size}")
        }
    }

    fun evaluate(inputs: List<Tensor>, labels: List<Tensor>): Float {
        require(inputs.size == labels.size) { "Inputs and labels must have the same size" }
        var totalLoss = 0f
        for ((input, label) in inputs.zip(labels)) {
            val predictions = layers.forward(input)
            totalLoss += mseLoss(predictions, label)
        }
        return totalLoss / inputs.size
    }

    // TODO: Lossクラスを作る
    private fun mseLoss(predictions: Tensor, labels: Tensor): Float {
        val diff = (predictions - labels) as Tensor2D
        val sum = diff.data.sumOf { row -> row.sumOf { it.toDouble() * it } }.toFloat()
        return sum / (diff.rows * diff.cols)
    }

    private fun mseLossGrad(predictions: Tensor, labels: Tensor): Tensor {
        val diff = (predictions - labels) as Tensor2D
        return diff * Tensor0D(2f / (diff.rows * diff.cols))
    }
}
