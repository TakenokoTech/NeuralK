package tech.takenoko.neuralk.nn.tensor

sealed class Tensor {
    abstract val shape: List<Int>
    abstract val rows: Int
    abstract val cols: Int

    operator fun plus(other: Tensor): Tensor {
        val vector = { a: Float, b: Float -> a + b }
        val matrix = { a: Array<Float>, b: Array<Float> -> a.mapElements(b, vector) }
        return when {
            this is Tensor0D && other is Tensor0D -> Tensor0D(data + other.data)
            this is Tensor1D && other is Tensor1D -> Tensor1D(data.mapElements(other.data, vector))
            this is Tensor2D && other is Tensor1D -> Tensor2D(data.broadcastMap(other.data, vector))
            this is Tensor2D && other is Tensor2D -> Tensor2D(data.mapElements(other.data, matrix))
            else -> throw IllegalArgumentException("Unsupported tensor types")
        }
    }

    operator fun minus(other: Tensor): Tensor {
        val vector = { a: Float, b: Float -> a - b }
        val matrix = { a: Array<Float>, b: Array<Float> -> a.mapElements(b, vector) }
        return when {
            this is Tensor0D && other is Tensor0D -> Tensor0D(data - other.data)
            this is Tensor1D && other is Tensor0D -> Tensor1D(data.broadcastMap(other.data, vector))
            this is Tensor1D && other is Tensor1D -> Tensor1D(data.mapElements(other.data, vector))
            this is Tensor2D && other is Tensor2D -> Tensor2D(data.mapElements(other.data, matrix))
            else -> throw IllegalArgumentException("Unsupported tensor types")
        }
    }

    operator fun times(other: Tensor): Tensor {
        val vector = { a: Float, b: Float -> a * b }
        val matrix = { a: Array<Float>, b: Array<Float> -> a.mapElements(b, vector) }
        return when {
            this is Tensor0D && other is Tensor0D -> Tensor0D(data * other.data)
            this is Tensor1D && other is Tensor0D -> Tensor1D(data.broadcastMap(other.data, vector))
            this is Tensor1D && other is Tensor1D -> Tensor1D(data.mapElements(other.data, vector))
            this is Tensor2D && other is Tensor0D -> Tensor2D(data.mapScalar(other.data, vector))
            this is Tensor2D && other is Tensor2D -> Tensor2D(data.mapElements(other.data, matrix))
            else -> throw IllegalArgumentException("Unsupported tensor types")
        }
    }

    private inline fun <T1, T2, reified R> Array<T1>.broadcastMap(
        other: T2,
        transform: (T1, T2) -> R
    ) = map { transform(it, other) }.toTypedArray()

    private inline fun <T1, T2, reified R> Array<Array<T1>>.broadcastMap(
        other: Array<T2>,
        transform: (T1, T2) -> R
    ) = map { it.zip(other, transform).toTypedArray() }.toTypedArray()

    private inline fun <T1, T2, reified R> Array<Array<T1>>.mapScalar(
        other: T2,
        transform: (T1, T2) -> R
    ) = map { it.map { v -> transform(v, other) }.toTypedArray() }.toTypedArray()

    private inline fun <T1, T2, reified R> Array<T1>.mapElements(
        other: Array<T2>,
        transform: (T1, T2) -> R
    ) = zip(other).map { transform(it.first, it.second) }.toTypedArray()

    fun printShape() = println("Tensor shape: ($rows, $cols), ${Throwable().stackTraceToString()}")
    fun toList(): List<*> = when (this) {
        is Tensor0D -> listOf(data)
        is Tensor1D -> data.toList()
        is Tensor2D -> data.flatMap { it.toList() }.toList()
    }
}
