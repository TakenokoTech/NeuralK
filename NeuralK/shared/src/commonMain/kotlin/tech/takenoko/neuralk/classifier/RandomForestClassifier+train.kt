package tech.takenoko.neuralk.classifier

// Trains a random forest classifier.
fun RandomForestClassifier.Companion.train(
    trainingData: Data.Frame,
    targetColumn: String,
    featureColumns: List<String>,
    parameters: RandomForestClassifier.ModelParameters,
    sessionParameters: RandomForestClassifier.TrainingSessionParameters
): RandomForestClassifier.Job<RandomForestClassifier> = runCatching {
    RandomForestClassifier(
        trainingData,
        targetColumn,
        featureColumns,
        parameters
    )
}.let(RandomForestClassifier::Job)

fun RandomForestClassifier.train(trainingData: Data.Frame) {
    val sample = trainingData.labels(targetColumn).mapIndexed { index, label ->
        DecisionTree.Data(trainingData.features(featureColumns, index), label)
    }
    val tree = DecisionTree(modelParameters.maxDepth)
    tree.fit(sample.shuffled())
    trees.add(tree)
}
