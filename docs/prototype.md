以下に、Kotlinを使った簡単なプロトタイプのコードを示します。  
このコードは、Kotlin Multiplatform Mobile（KMM）で動作するライブラリの基盤を形成するものです。

## ファイル構成

```md
- src
  - commonMain
    - model
      - Tensor.kt
      - Layer.kt
      - Model.kt
      - JsonHandler.kt
    - optimizer
      - Optimizer.kt
```

## 1. Tensor.kt
テンソルの基本的な操作を実装します。

```kotlin
package model

class Tensor(val data: Array<Array<Float>>) {
    val rows = data.size
    val cols = if (data.isNotEmpty()) data[0].size else 0

    operator fun plus(other: Tensor): Tensor {
        require(rows == other.rows && cols == other.cols) { "Dimension mismatch" }
        val result = Array(rows) { row -> 
            Array(cols) { col -> data[row][col] + other.data[row][col] } 
        }
        return Tensor(result)
    }

    operator fun minus(other: Tensor): Tensor {
        require(rows == other.rows && cols == other.cols) { "Dimension mismatch" }
        val result = Array(rows) { row ->
            Array(cols) { col -> data[row][col] - other.data[row][col] }
        }
        return Tensor(result)
    }

    operator fun times(scalar: Float): Tensor {
        val result = Array(rows) { row -> 
            Array(cols) { col -> data[row][col] * scalar } 
        }
        return Tensor(result)
    }
}
```

## 2. Layer.kt
CNNのレイヤーを定義します（例: 畳み込み層、全結合層）。

```kotlin
package model

abstract class Layer {
    abstract fun forward(input: Tensor): Tensor
    abstract fun backward(gradOutput: Tensor): Tensor
}

class DenseLayer(private val inputSize: Int, private val outputSize: Int) : Layer() {
    private val weights = Tensor(Array(outputSize) { Array(inputSize) { Math.random().toFloat() } })
    private val bias = Tensor(Array(outputSize) { Array(1) { 0f } })

    override fun forward(input: Tensor): Tensor {
        // Dense layer forward pass (simplified)
        return (weights * input) + bias
    }

    override fun backward(gradOutput: Tensor): Tensor {
        // Backpropagation logic (simplified for prototype)
        return gradOutput
    }
}
```

## 3. Model.kt
モデル全体を管理します。

```kotlin
package model

import optimizer.SGD

class Model(private val layers: List<Layer>, private val optimizer: SGD) {

    fun fit(inputs: List<Tensor>, labels: List<Tensor>, epochs: Int) {
        require(inputs.size == labels.size) { "Inputs and labels must have the same size" }
        for (epoch in 1..epochs) {
            var totalLoss = 0f
            for ((input, label) in inputs.zip(labels)) {
                // Forward pass
                val predictions = forward(input)

                // Calculate loss (Mean Squared Error as example)
                val loss = mseLoss(predictions, label)
                totalLoss += loss

                // Backward pass
                val grad = mseLossGrad(predictions, label)
                backward(grad)
            }
            println("Epoch $epoch: Loss = ${totalLoss / inputs.size}")
        }
    }

    fun evaluate(inputs: List<Tensor>, labels: List<Tensor>): Float {
        require(inputs.size == labels.size) { "Inputs and labels must have the same size" }
        var totalLoss = 0f
        for ((input, label) in inputs.zip(labels)) {
            val predictions = forward(input)
            totalLoss += mseLoss(predictions, label)
        }
        return totalLoss / inputs.size
    }

    fun predict(input: Tensor): Tensor {
        return forward(input)
    }

    private fun forward(input: Tensor): Tensor {
        var currentOutput = input
        for (layer in layers) {
            currentOutput = layer.forward(currentOutput)
        }
        return currentOutput
    }

    private fun backward(gradOutput: Tensor) {
        var currentGrad = gradOutput
        for (layer in layers.reversed()) {
            currentGrad = layer.backward(currentGrad)
        }
    }

    private fun mseLoss(predictions: Tensor, labels: Tensor): Float {
        val diff = predictions - labels
        return diff.data.sumOf { row -> row.sumOf { it * it } } / (diff.rows * diff.cols)
    }

    private fun mseLossGrad(predictions: Tensor, labels: Tensor): Tensor {
        val diff = predictions - labels
        return diff * (2f / (diff.rows * diff.cols))
    }
}
```

## 4. JsonHandler.kt
モデルをJSON形式で保存・復元します。

```kotlin
package model

import kotlinx.serialization.*
import kotlinx.serialization.json.*

@Serializable
data class ModelData(val layers: List<Map<String, Any>>)

class JsonHandler {
    fun saveModel(model: Model): String {
        val modelData = ModelData(
            layers = model.layers.map { mapOf("type" to it::class.simpleName!!) }
        )
        return Json.encodeToString(modelData)
    }

    fun loadModel(json: String): Model {
        val modelData = Json.decodeFromString<ModelData>(json)
        val layers = modelData.layers.map {
            when (it["type"]) {
                "DenseLayer" -> DenseLayer(10, 10) // Adjust sizes as needed
                else -> throw IllegalArgumentException("Unsupported layer type")
            }
        }
        return Model(layers)
    }
}
```

## 5. Optimizer.kt
最適化アルゴリズムを実装します（例: SGD）。

```kotlin
package optimizer

import model.Tensor

class SGD(private val learningRate: Float) {
    fun update(weights: Tensor, gradients: Tensor): Tensor {
        return weights - (gradients * learningRate)
    }
}
```

## 使用例
以下のコードで、プロトタイプが動作することを確認できます。

```kotlin
import model.*
import optimizer.SGD

fun main() {
    // データ準備（簡易例）
    val inputs = listOf(
        Tensor(Array(3) { Array(1) { it.toFloat() } }), // 例: 3x1行列
        Tensor(Array(3) { Array(1) { it.toFloat() * 2 } })
    )
    val labels = listOf(
        Tensor(Array(1) { Array(1) { 1f } }), // ラベル
        Tensor(Array(1) { Array(1) { 2f } })
    )

    // モデルの構築
    val layers = listOf(DenseLayer(3, 1))
    val optimizer = SGD(learningRate = 0.01f)
    val model = Model(layers, optimizer)

    // 学習
    println("=== Training ===")
    model.fit(inputs, labels, epochs = 10)

    // 評価
    println("=== Evaluation ===")
    val loss = model.evaluate(inputs, labels)
    println("Loss: $loss")

    // 推論
    println("=== Prediction ===")
    val testInput = Tensor(Array(3) { Array(1) { it.toFloat() * 3 } })
    val prediction = model.predict(testInput)
    println("Prediction: ${prediction.data}")
}
```
