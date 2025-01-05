package tech.takenoko.neuralk.nn.tensor

import tech.takenoko.neuralk.nn.value.Shape

class Tensor2D(val data: Array<Array<Float>>) : Tensor() {
    override val shape = Shape(data.size, data.first().size)

    constructor(rows: Int, cols: Int, data: Double) :
        this(Array(rows) { Array(cols) { data.toFloat() } })

    constructor(shape: Shape, data: Double) :
        this(shape.value[0], shape.value[1], data)

    fun sum(): Tensor1D = Tensor1D(
        data = Array(size = cols) { col ->
            var sum = 0f
            for (row in 0 until rows) sum += data[row][col]
            sum
        }
    )

    fun transpose() = Tensor2D(Array(cols) { i -> Array(rows) { j -> data[j][i] } })
}
