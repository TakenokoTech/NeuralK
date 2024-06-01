package tech.takenoko.neuralk.classifier

import kotlin.test.Test

class RandomForestClassifierTest {

    @Test
    fun test() {
        val randomForestClassifier = RandomForestClassifier(
            trainingData = Data.Frame(
                mapOf(
                    "A" to listOf(1.11, 2.22, 3.33),
                    "B" to listOf(4.44, 5.55, 6.66),
                    "C" to listOf(2, 1, 0)
                )
            ),
            targetColumn = "C",
            featureColumns = listOf("A", "B"),
            parameters = RandomForestClassifier.ModelParameters()
        )
        val result = randomForestClassifier.predictions(
            data = Data.Frame(
                mapOf(
                    "A" to listOf(1.12),
                    "B" to listOf(4.34),
                )
            )
        )
        println(result)
        println(randomForestClassifier.trees)
    }
}
