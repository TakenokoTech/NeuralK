package tech.takenoko.neuralk.classifier

fun RandomForestClassifier.predictions(
    data: Data.Frame
): Int {
    val features = featureColumns.map { data.map[it] }[0]!!.map { it as Double }
    val predictions = trees.map { it.predict(features) }
    return predictions.groupBy { it }.maxByOrNull { it.value.size }?.key ?: -1
}
