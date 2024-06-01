package tech.takenoko.neuralk.classifier

// Evaluates the classifier on the provided labeled data.
fun RandomForestClassifier.evaluation(
    on: Data
): RandomForestClassifier.ClassifierMetrics = TODO()

// Exports a Core ML model file for use in your app.
fun RandomForestClassifier.write(
    toFile: String,
    metadata: RandomForestClassifier.ModelMetadata
): Unit = TODO()

// Creates or restores a training session.
fun RandomForestClassifier.Companion.makeTrainingSession(
    trainingData: Data,
    targetColumn: String,
    featureColumns: List<String>,
    parameters: RandomForestClassifier.ModelParameters,
    sessionParameters: RandomForestClassifier.TrainingSessionParameters
): RandomForestClassifier.TrainingSession<RandomForestClassifier> = TODO()

// Restores an existing training session.
fun RandomForestClassifier.Companion.restoreTrainingSession(
    sessionParameters: RandomForestClassifier.TrainingSessionParameters
): RandomForestClassifier.TrainingSession<RandomForestClassifier> = TODO()

// Resumes a training session from the last checkpoint if available.
fun RandomForestClassifier.Companion.resume(
    trainingSession: RandomForestClassifier.TrainingSession<RandomForestClassifier>
): RandomForestClassifier.Job<RandomForestClassifier> = TODO()
