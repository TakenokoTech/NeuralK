package tech.takenoko.neuralk.nn.model

import tech.takenoko.neuralk.nn.layer.Layers
import tech.takenoko.neuralk.nn.tensor.Tensor

internal interface Predictor {
    var layers: Layers
    fun predict(input: Tensor): Tensor = layers.forward(input)
}
