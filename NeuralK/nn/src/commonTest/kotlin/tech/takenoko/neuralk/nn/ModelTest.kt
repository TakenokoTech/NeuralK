package tech.takenoko.neuralk.nn

import kotlin.random.Random
import kotlin.test.Test
import tech.takenoko.neuralk.nn.layer.Dense
import tech.takenoko.neuralk.nn.model.Model
import tech.takenoko.neuralk.nn.model.Sequential
import tech.takenoko.neuralk.nn.optimizer.Sgd
import tech.takenoko.neuralk.nn.tensor.Tensor2D

class ModelTest {

    private fun rand() = Random.nextFloat() * 0.01F
    private fun input(i: Int) = Tensor2D(Array(1) { Array(3) { (it.plus(1) + rand()) * i } })
    private fun label(index: Int) = Tensor2D(Array(1) { Array(1) { index.toFloat() } })
    private val inputs = (0..100).flatMap { listOf(input(1), input(2), input(3)) }
    private val labels = (0..100).flatMap { listOf(label(1), label(2), label(3)) }

    private val testData = listOf(
        Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 0F } }),
        Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 1F } }),
        Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 2F } }),
        Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 3F } })
    )

    @Test
    fun test1() {
        // モデルの構築
        val layers = listOf(
            Dense(5),
            Dense(3)
        )
        val optimizer = Sgd(learningRate = 0.01)
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
        testData.forEach { testInput ->
            val prediction = model.predict(testInput) as Tensor2D
            println("Input: ${testInput.toList()}, Prediction: ${prediction.toList()}")
        }
    }

    @Test
    fun test2() {
        val model = Sequential {
            dense(5)
            dense(3)
        }.compile(Sgd(learningRate = 0.01))

        println("=== Training ===")
        model.fit(inputs, labels, epochs = 50)

        println("=== Evaluation ===")
        val loss = model.evaluate(inputs, labels)
        println("Loss: $loss")

        println("=== Prediction ===")
        testData.forEach { testInput ->
            val prediction = model.predict(testInput) as Tensor2D
            println("Input: ${testInput.toList()}, Prediction: ${prediction.toList()}")
        }
    }
}
