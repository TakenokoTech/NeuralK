package tech.takenoko.neuralk.nn.model

import tech.takenoko.neuralk.nn.layer.Dense
import tech.takenoko.neuralk.nn.optimizer.Optimizer
import tech.takenoko.neuralk.nn.optimizer.Sgd

class Sequential(block: Sequential.() -> Unit) : Model(emptyList(), Sgd()) {
    init {
        block()
    }

    fun Sequential.dense(units: Int) {
        this.layers += Dense(units)
    }

    fun compile(optimizer: Optimizer): Sequential {
        this.optimizer = optimizer
        return this
    }
}
