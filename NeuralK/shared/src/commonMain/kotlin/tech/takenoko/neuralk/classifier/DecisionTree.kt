package tech.takenoko.neuralk.classifier


data class DecisionTree(
    private val maxDepth: Int,
    private var root: TreeNode? = null
) {
    fun fit(data: List<Data>) {
        root = buildTree(data, 0)
    }

    fun predict(features: List<Double>): Int {
        var node = root
        while (node != null && node.value == null) {
            node = if (features[node.featureIndex] <= node.threshold) node.left else node.right
        }
        return node?.value ?: -1
    }

    internal fun buildTree(data: List<Data>, depth: Int): TreeNode {
        if (data.isEmpty() || depth == maxDepth) {
            return TreeNode(
                featureIndex = -1,
                threshold = -1.0,
                value = getMajorityClass(data)
            )
        }
        val (bestFeature, bestThreshold) = findBestSplit(data)
        val (leftData, rightData) = data.partition { it.features[bestFeature] <= bestThreshold }
        return TreeNode(
            bestFeature,
            bestThreshold,
            buildTree(leftData, depth + 1),
            buildTree(rightData, depth + 1),
            null
        )
    }

    internal fun findBestSplit(data: List<Data>): Pair<Int, Double> {
        var bestFeature = -1
        var bestThreshold = Double.MAX_VALUE
        var bestGini = Double.MAX_VALUE
        if (data.isEmpty()) return bestFeature to bestThreshold
        for (featureIndex in data[0].features.indices) {
            val thresholds = data.map { it.features[featureIndex] }.distinct()
            for (threshold in thresholds) {
                val gini = calculateGini(data, featureIndex, threshold)
                if (gini < bestGini) {
                    bestGini = gini
                    bestFeature = featureIndex
                    bestThreshold = threshold
                }
            }
        }
        return bestFeature to bestThreshold
    }

    internal fun calculateGini(data: List<Data>, featureIndex: Int, threshold: Double): Double {
        if (data.isEmpty()) return 0.0
        val (leftData, rightData) = data.partition { it.features[featureIndex] <= threshold }
        val leftGini = calculateGiniImpurity(leftData)
        val rightGini = calculateGiniImpurity(rightData)
        return (leftData.size.toDouble() / data.size) * leftGini + (rightData.size.toDouble() / data.size) * rightGini
    }

    internal fun calculateGiniImpurity(data: List<Data>): Double {
        if (data.isEmpty()) return 0.0
        val classes = data.groupBy { it.label }
        val proportions = classes.values.map { it.size.toDouble() / data.size }
        return 1.0 - proportions.sumOf { it * it }
    }

    internal fun getMajorityClass(data: List<Data>): Int {
        return data.groupBy { it.label }.maxByOrNull { it.value.size }?.key ?: -1
    }

    override fun toString(): String {
        return "DecisionTree(\n\tmaxDepth=${maxDepth},\n\ttree=\n${root}\n)"
    }

    data class Data(val features: List<Double>, val label: Int)
}