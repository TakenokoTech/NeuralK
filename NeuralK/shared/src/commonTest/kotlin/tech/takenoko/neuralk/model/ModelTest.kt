package tech.takenoko.neuralk.model

import kotlin.random.Random
import kotlin.test.Test

class ModelTest {

    @Test
    fun test() {
        // データ準備（簡易例）
        fun rand() = Random.nextFloat() * 0.01F
        val inputs = listOf(
            Tensor2D(Array(1) { Array(3) { (it + rand()) * 1 } }),
            Tensor2D(Array(1) { Array(3) { (it + rand()) * 1 } }),
            Tensor2D(Array(1) { Array(3) { (it + rand()) * 1 } }),
            Tensor2D(Array(1) { Array(3) { (it + rand()) * 2 } }),
            Tensor2D(Array(1) { Array(3) { (it + rand()) * 2 } }),
            Tensor2D(Array(1) { Array(3) { (it + rand()) * 2 } }),
            Tensor2D(Array(1) { Array(3) { (it + rand()) * 3 } }),
            Tensor2D(Array(1) { Array(3) { (it + rand()) * 3 } }),
            Tensor2D(Array(1) { Array(3) { (it + rand()) * 3 } })
        )
        val labels = listOf(
            Tensor2D(Array(1) { Array(1) { 1f } }),
            Tensor2D(Array(1) { Array(1) { 1f } }),
            Tensor2D(Array(1) { Array(1) { 1f } }),
            Tensor2D(Array(1) { Array(1) { 2f } }),
            Tensor2D(Array(1) { Array(1) { 2f } }),
            Tensor2D(Array(1) { Array(1) { 2f } }),
            Tensor2D(Array(1) { Array(1) { 3f } }),
            Tensor2D(Array(1) { Array(1) { 3f } }),
            Tensor2D(Array(1) { Array(1) { 3f } })
        )

        // モデルの構築
        val layers = listOf(
            DenseLayer(10)
        )
        val optimizer = SGD(learningRate = 0.01f)
        val model = Model(layers, optimizer)

        // 学習
        println("=== Training ===")
        model.fit(inputs, labels, epochs = 100)

        // 評価
        println("=== Evaluation ===")
        val loss = model.evaluate(inputs, labels)
        println("Loss: $loss")

        // 推論
        println("=== Prediction ===")
        listOf(
            Tensor2D(Array(1) { Array(3) { (it + 1).toFloat() * 1 } }),
            Tensor2D(Array(1) { Array(3) { (it + 1).toFloat() * 2 } }),
            Tensor2D(Array(1) { Array(3) { (it + 1).toFloat() * 3 } })
        ).forEach { testInput ->
            println("Input: ${testInput.data.map { it.toList() }}")
            val prediction = model.predict(testInput) as Tensor2D
            println("Prediction: ${prediction.data.map { it.toList() }}")
        }
    }
}
