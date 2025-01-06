package tech.takenoko.neuralk.nn.model

import tech.takenoko.neuralk.nn.layer.Layers
import tech.takenoko.neuralk.nn.loss.Loss
import tech.takenoko.neuralk.nn.optimizer.Optimizer
import tech.takenoko.neuralk.nn.tensor.Tensor

internal interface Trainer {
    var layers: Layers
    var optimizer: Optimizer?
    var loss: Loss?

    fun fit(inputs: List<Tensor>, labels: List<Tensor>, epochs: Int, shuffle: Boolean = true) {
        require(inputs.size == labels.size) { "Inputs and labels must have the same size" }
        val optimizer = optimizer ?: error("Optimizer must be set")
        val loss = loss ?: error("Loss must be set")
        for (epoch in 1..epochs) {
            var totalLoss = 0.0
            val shuffledInputs = inputs.zip(labels).let { if (shuffle) it.shuffled() else it }
            for ((input, label) in shuffledInputs) {
                val predictions = layers.forward(input)
                totalLoss += loss.computeLoss(predictions, label)
                val grad = loss.computeGradient(predictions, label)
                layers.backward(grad)
                layers.update(optimizer)
            }
            if (epoch % 3 == 0) println("Epoch $epoch: Loss = ${totalLoss / inputs.size}")
        }
    }

    fun evaluate(inputs: List<Tensor>, labels: List<Tensor>): Float {
        require(inputs.size == labels.size) { "Inputs and labels must have the same size" }
        val loss = loss ?: error("Loss must be set")
        var totalLoss = 0f
        for ((input, label) in inputs.zip(labels)) {
            val predictions = layers.forward(input)
            totalLoss += loss.computeLoss(predictions, label)
        }
        return totalLoss / inputs.size
    }
}
