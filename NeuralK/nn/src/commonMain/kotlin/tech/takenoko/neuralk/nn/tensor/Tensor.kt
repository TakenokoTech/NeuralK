package tech.takenoko.neuralk.nn.tensor

import tech.takenoko.neuralk.nn.value.Shape
import kotlin.math.sqrt

sealed class Tensor {
    abstract val shape: Shape
    val rows: Int get() = shape.rows
    val cols: Int get() = shape.cols

    operator fun plus(other: Tensor) = calc(other) { a, b -> a + b }
    operator fun minus(other: Tensor) = calc(other) { a, b -> a - b }
    operator fun times(other: Tensor) = calc(other) { a, b -> a * b }
    operator fun div(other: Tensor) = calc(other) { a, b -> a / b }
    operator fun unaryMinus() = calc(this) { a, _ -> -a }
    fun sqrt() = calc(this) { a, _ -> sqrt(a.toDouble()).toFloat() }

    private inline fun <T1, T2, reified R> Array<T1>.broadcastMap(
        other: T2,
        transform: (T1, T2) -> R,
    ) = map { transform(it, other) }.toTypedArray()

    private inline fun <T1, T2, reified R> Array<Array<T1>>.broadcastMap(
        other: Array<T2>,
        transform: (T1, T2) -> R,
    ) = map { it.zip(other, transform).toTypedArray() }.toTypedArray()

    private inline fun <T1, T2, reified R> Array<Array<T1>>.mapScalar(
        other: T2,
        transform: (T1, T2) -> R,
    ) = map { it.map { v -> transform(v, other) }.toTypedArray() }.toTypedArray()

    private inline fun <T1, T2, reified R> Array<T1>.mapElements(
        other: Array<T2>,
        transform: (T1, T2) -> R,
    ) = zip(other).map { transform(it.first, it.second) }.toTypedArray()

    private inline fun calc(
        other: Tensor,
        crossinline vector: (Float, Float) -> Float,
    ): Tensor {
        val matrix = { a: Array<Float>, b: Array<Float> -> a.mapElements(b, vector) }
        return when {
            this is Tensor0D && other is Tensor0D -> Tensor0D(vector(data, other.data))
            this is Tensor1D && other is Tensor0D -> Tensor1D(data.broadcastMap(other.data, vector))
            this is Tensor1D && other is Tensor1D -> Tensor1D(data.mapElements(other.data, vector))
            this is Tensor2D && other is Tensor0D -> Tensor2D(data.mapScalar(other.data, vector))
            this is Tensor2D && other is Tensor1D -> Tensor2D(data.broadcastMap(other.data, vector))
            this is Tensor2D && other is Tensor2D -> Tensor2D(data.mapElements(other.data, matrix))
            else -> throw IllegalArgumentException("Unsupported tensor types")
        }
    }

    fun printShape() = println("Tensor shape: ($rows, $cols), ${Throwable().stackTraceToString()}")
    fun toList(): List<*> = when (this) {
        is Tensor0D -> listOf(data)
        is Tensor1D -> data.toList()
        is Tensor2D -> data.flatMap { it.toList() }.toList()
    }

    companion object {
        fun zeros(shape: Shape): Tensor = when (shape.value.size) {
            0 -> Tensor0D(0.0)
            1 -> Tensor1D(shape, 0.0)
            2 -> Tensor2D(shape, 0.0)
            else -> throw IllegalArgumentException("Unsupported shape size")
        }
    }
}
