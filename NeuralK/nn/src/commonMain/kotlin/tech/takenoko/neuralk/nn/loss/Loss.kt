package tech.takenoko.neuralk.nn.loss

import tech.takenoko.neuralk.nn.tensor.Tensor

sealed class Loss {
    abstract fun computeLoss(predictions: Tensor, targets: Tensor): Float
    abstract fun computeGradient(predictions: Tensor, targets: Tensor): Tensor
}
