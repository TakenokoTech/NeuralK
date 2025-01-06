package tech.takenoko.neuralk.nn

import tech.takenoko.neuralk.nn.activation.Softmax
import tech.takenoko.neuralk.nn.layer.Activation
import tech.takenoko.neuralk.nn.layer.Dense
import tech.takenoko.neuralk.nn.layer.Layers
import tech.takenoko.neuralk.nn.loss.CategoricalCrossentropy
import tech.takenoko.neuralk.nn.loss.MeanSquaredError
import tech.takenoko.neuralk.nn.model.Model
import tech.takenoko.neuralk.nn.model.Sequential
import tech.takenoko.neuralk.nn.optimizer.Adam
import tech.takenoko.neuralk.nn.optimizer.Sgd
import tech.takenoko.neuralk.nn.tensor.Tensor
import tech.takenoko.neuralk.nn.tensor.Tensor2D
import tech.takenoko.neuralk.nn.value.Shape
import kotlin.random.Random
import kotlin.test.Test

class ModelTest {

    companion object {
        private val epochs = 30
        private val shuffle = false
        private fun rand() = Random.nextFloat() * 0.1F
        private val inputs = (0..100).flatMap {
            fun input(i: Int) = Tensor2D(Array(1) { Array(3) { (it.plus(1) + rand()) * i } })
            listOf(input(1), input(4), input(3), input(2), input(5))
        }
        private val labels1 = (0..100).flatMap {
            fun label(index: Int) = Tensor2D(Array(1) { Array(5) { if (it == index) 1F else 0F } })
            listOf(label(1), label(4), label(3), label(2), label(5))
        }
        private val labels2 = (0..100).flatMap {
            fun label(index: Int) = Tensor2D(Array(1) { Array(1) { index + 1.0F } })
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

    private fun run(model: Model, labels: List<Tensor>) {
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
            val argmax = prediction.toList().mapIndexed { i, v -> i to v as Float }.maxByOrNull { it.second }?.first
            println("Input: ${testInput.toList()}, ArgMax: $argmax, Prediction: ${prediction.toList()}")
        }
    }

    @Test
    fun test1() {
        // モデルの構築
        val model = Model(
            layers = Layers(
                Dense(3),
                Dense(5),
                Activation(Softmax),
            ),
            optimizer = Sgd(learningRate = 0.01, momentum = 0.0),
            loss = CategoricalCrossentropy,
        )
        run(model, labels1)
    }

    @Test
    fun test2() {
        val model = Sequential {
            input(shape = Shape(1, 3))
            dense(5)
            dense(1)
            Activation(Softmax)
        }.compile(
            optimizer = Sgd(learningRate = 0.001),
            loss = MeanSquaredError,
        )
        run(model, labels2)
    }

    @Test
    fun test3() {
        val model = Sequential {
            input(shape = Shape(1, 3))
            dense(3)
            dense(5)
            Activation(Softmax)
        }.compile(
            optimizer = Adam(learningRate = 0.01),
            loss = CategoricalCrossentropy,
        )
        run(model, labels1)
    }
}
