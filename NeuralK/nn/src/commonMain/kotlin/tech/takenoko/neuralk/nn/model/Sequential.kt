package tech.takenoko.neuralk.nn.model

import tech.takenoko.neuralk.nn.layer.Dense
import tech.takenoko.neuralk.nn.layer.Input
import tech.takenoko.neuralk.nn.layer.Layer
import tech.takenoko.neuralk.nn.optimizer.Optimizer
import tech.takenoko.neuralk.nn.optimizer.Sgd
import tech.takenoko.neuralk.nn.value.Shape

class Sequential(block: Sequential.() -> Unit) : Model(emptyList(), Sgd()) {
    init {
        block()
    }

    fun Sequential.input(shape: Shape) = addLayer(Input(shape))
    fun Sequential.dense(units: Int) = addLayer(Dense(units))
    fun compile(optimizer: Optimizer): Sequential = apply { this.optimizer = optimizer }

    private fun Sequential.addLayer(layer: Layer) {
        if (layers.isEmpty()) {
            require(layer is Input) { "Input layer must be the first layer" }
        }
        this.layers += layer
    }
}
