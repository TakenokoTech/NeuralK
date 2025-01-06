package tech.takenoko.neuralk.nn.loss

import tech.takenoko.neuralk.nn.tensor.Tensor
import tech.takenoko.neuralk.nn.tensor.Tensor2D
import kotlin.math.ln

data object CategoricalCrossentropy : Loss() {
    override fun computeLoss(predictions: Tensor, targets: Tensor): Float {
        predictions as? Tensor2D ?: error("Predictions must be 2D tensor")
        targets as? Tensor2D ?: error("Targets must be 2D tensor")
        return predictions.data.zip(targets.data) { predRow, targetRow ->
            predRow.zip(targetRow) { pred, target -> -target * ln(pred.coerceIn(1e-15f, 1f)) }.sum()
        }.sum()
    }

    override fun computeGradient(predictions: Tensor, targets: Tensor): Tensor {
        predictions as? Tensor2D ?: error("Predictions must be 2D tensor")
        targets as? Tensor2D ?: error("Targets must be 2D tensor")
        val grad = predictions.data.zip(targets.data) { predRow, targetRow ->
            predRow.zip(targetRow) { pred, target -> pred - target }
        }
        return Tensor2D(grad)
    }
}
