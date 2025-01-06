package tech.takenoko.neuralk.nn.loss

import tech.takenoko.neuralk.nn.tensor.Tensor
import tech.takenoko.neuralk.nn.tensor.Tensor0D
import tech.takenoko.neuralk.nn.tensor.Tensor2D

data object MeanSquaredError : Loss() {
    override fun computeLoss(predictions: Tensor, targets: Tensor): Float {
        val diff = (predictions - targets) as Tensor2D
        val sum = diff.data.sumOf { row -> row.sumOf { it.toDouble() * it } }.toFloat()
        return sum / (diff.rows * diff.cols)
    }

    override fun computeGradient(predictions: Tensor, targets: Tensor): Tensor {
        val diff = (predictions - targets) as Tensor2D
        return diff * Tensor0D(2f / (diff.rows * diff.cols))
    }
}
