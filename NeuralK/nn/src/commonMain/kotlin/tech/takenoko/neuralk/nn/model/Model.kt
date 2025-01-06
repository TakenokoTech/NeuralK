package tech.takenoko.neuralk.nn.model

import tech.takenoko.neuralk.nn.layer.Layers
import tech.takenoko.neuralk.nn.loss.Loss
import tech.takenoko.neuralk.nn.optimizer.Optimizer

open class Model(
    override var layers: Layers,
    override var optimizer: Optimizer? = null,
    override var loss: Loss? = null,
) : Predictor,
    Trainer
