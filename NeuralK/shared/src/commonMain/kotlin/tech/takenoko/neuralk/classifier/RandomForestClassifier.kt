package tech.takenoko.neuralk.classifier

class RandomForestClassifier(
    // The name of the column you selected at initialization to define which categories the classifier predicts.
    val targetColumn: String,
    // The names of the columns you selected at initialization to train the classifier.
    val featureColumns: List<String>,
    // The underlying parameters used when training the model.
    val modelParameters: ModelParameters,
) {
    class Checkpoint
    class TrainingSession<T>
    class TrainingSessionParameters
    class Job<T>(var result: Result<T>)
    class ClassifierMetrics
    class ModelMetadata
    interface AnyColumn

    class ModelParameters(
        // val columnSubsample: Double,
        var maxDepth: Int = 10,
        // var maxIterations: Int,
        // var minChildWeight: Double,
        // var minLossReduction: Double,
        // var randomSeed: Int,
        // var rowSubsample: Double,
        // var validationData: Data.Table?,
        // var validation: ValidationData,
    ) {
        enum class ValidationData { split, table, dataFrame, none }
    }

    val trees = mutableListOf<DecisionTree>()

    // Measurements of the classifier’s performance on the training data set.
    private var trainingMetrics: ClassifierMetrics? = null

    // Measurements of the classifier’s performance on the validation data set.
    private var validationMetrics: ClassifierMetrics? = null

    // Creates a random forest classifier classifier from a checkpoint.
    // constructor(checkpoint: Checkpoint)

    // Creates a random forest classifier.
    constructor(
        trainingData: Data.Frame,
        targetColumn: String,
        featureColumns: List<String>,
        parameters: ModelParameters
    ) : this(targetColumn, featureColumns, parameters) {
        train(trainingData = trainingData)
    }

    companion object {
        private val TAG = this::class.simpleName
    }
}
