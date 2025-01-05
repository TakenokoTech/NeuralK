package tech.takenoko.neuralk.nn.tensor

class Tensor2D(val data: Array<Array<Float>>) : Tensor() {
    override val shape = listOf(data.size, data.first().size)
    override val rows = shape[0]
    override val cols = shape[1]

    constructor(rows: Int, cols: Int, data: Double) :
        this(Array(rows) { Array(cols) { data.toFloat() } })

    fun sum(): Tensor1D = Tensor1D(
        data = Array(size = cols) { col ->
            var sum = 0f
            for (row in 0 until rows) sum += data[row][col]
            sum
        }
    )

    fun transpose() = Tensor2D(Array(cols) { i -> Array(rows) { j -> data[j][i] } })
}
