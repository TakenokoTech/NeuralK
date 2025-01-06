package tech.takenoko.neuralk.nn.activation

import tech.takenoko.neuralk.nn.tensor.Tensor
import tech.takenoko.neuralk.nn.tensor.Tensor1D
import tech.takenoko.neuralk.nn.tensor.Tensor2D
import kotlin.math.exp

data object Softmax : Activation() {
    override fun forward(input: Tensor): Tensor = when (input) {
        is Tensor1D -> {
            val exp = input.data.map(::exp)
            val sum = exp.sum()
            val normalized = exp.map { it / (sum + 1e-15f) }
            Tensor1D(normalized)
        }
        is Tensor2D -> {
            val exp = input.data.map { it.map(::exp) }
            val sum = exp.map { it.sum() }
            val normalized = exp.mapIndexed { i, row -> row.map { it / (sum[i] + 1e-15f) } }
            Tensor2D(normalized)
        }
        else -> throw IllegalArgumentException("Unsupported shape.")
    }
}
