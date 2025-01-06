package tech.takenoko.neuralk.nn

import tech.takenoko.neuralk.nn.layer.Dense
import tech.takenoko.neuralk.nn.layer.Layers
import tech.takenoko.neuralk.nn.loss.CategoricalCrossentropy
import tech.takenoko.neuralk.nn.loss.MeanSquaredError
import tech.takenoko.neuralk.nn.model.Model
import tech.takenoko.neuralk.nn.model.Sequential
import tech.takenoko.neuralk.nn.optimizer.Sgd
import tech.takenoko.neuralk.nn.tensor.Tensor2D
import tech.takenoko.neuralk.nn.value.Shape
import kotlin.random.Random
import kotlin.test.Test

class ModelTest {

    companion object {
        private val epochs = 30
        private val shuffle = false
        private fun rand() = Random.nextFloat() * 0.1F
        private fun input(i: Int) = Tensor2D(Array(1) { Array(3) { (it.plus(1) + rand()) * i } })
        private fun label(index: Int) = Tensor2D(Array(1) { Array(3) { if (it == index) 1F else 0F } })

        private val inputs = (0..100).flatMap {
            listOf(input(1), input(4), input(3), input(2), input(5))
        }
        private val labels = (0..100).flatMap {
            listOf(label(1), label(4), label(3), label(2), label(5))
        }

        private val testData = listOf(
            Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 0F } }),
            Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 1F } }),
            Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 2F } }),
            Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 3F } }),
            Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 4F } }),
            Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 5F } }),
            Tensor2D(Array(1) { Array(3) { it.plus(1).toFloat() * 6F } }),
        )
    }

    @Test
    fun test1() {
        // モデルの構築
        val model = Model(
            layers = Layers(
                Dense(3),
//                Dense(5),
            ),
            optimizer = Sgd(learningRate = 0.01, momentum = 0.0),
            loss = MeanSquaredError,
        )

        // 推論
        println("=== Prediction ===")
        testData.forEach { testInput ->
            val prediction = model.predict(testInput) as Tensor2D
            println("Input: ${testInput.toList()}, Prediction: ${prediction.toList()}")
        }

        // 学習
        println("=== Training ===")
        model.fit(inputs, labels, epochs, shuffle)

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
            input(shape = Shape(1, 3))
            dense(3)
//            dense(5)
        }.compile(
            optimizer = Sgd(learningRate = 0.001, momentum = 0.0),
//            optimizer = Adam(learningRate = 0.01),
            loss = CategoricalCrossentropy,
//            loss = MeanSquaredError,
        )

        println("=== Prediction ===")
        testData.forEach { testInput ->
            val prediction = model.predict(testInput) as Tensor2D
            println("Input: ${testInput.toList()}, Prediction: ${prediction.toList()}")
        }

        println("=== Training ===")
        model.fit(inputs, labels, epochs, shuffle)

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
