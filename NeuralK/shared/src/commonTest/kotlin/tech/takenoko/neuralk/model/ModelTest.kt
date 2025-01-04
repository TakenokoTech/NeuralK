package tech.takenoko.neuralk.model

import kotlin.random.Random
import kotlin.test.Test

class ModelTest {

    @Test
    fun test() {
        // データ準備（簡易例）
        fun rand() = Random.nextFloat() * 0.01F
        fun input(index: Int) = Tensor2D(Array(1) { Array(3) { (it.plus(1) + rand()) * index } })
        fun label(index: Int) = Tensor2D(Array(1) { Array(1) { index.toFloat() } })
        val inputs = (0..100).flatMap { listOf(input(1), input(2), input(3)) }
        val labels = (0..100).flatMap { listOf(label(1), label(2), label(3)) }

        // モデルの構築
        val layers = listOf(
            DenseLayer(5),
            DenseLayer(3)
        )
        val optimizer = SGD(learningRate = 0.01f)
        val model = Model(layers, optimizer)

        // 学習
        println("=== Training ===")
        model.fit(inputs, labels, epochs = 50)

        // 評価
        println("=== Evaluation ===")
        val loss = model.evaluate(inputs, labels)
        println("Loss: $loss")

        // 推論
        println("=== Prediction ===")
        listOf(
            Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 0F } }),
            Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 1F } }),
            Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 2F } }),
            Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 3F } })
        ).forEach { testInput ->
            val prediction = model.predict(testInput) as Tensor2D
            println("Input: ${testInput.toList()}, Prediction: ${prediction.toList()}")
        }
    }
}
