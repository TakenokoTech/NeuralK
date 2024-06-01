package tech.takenoko.neuralk.classifier

data class TreeNode(
    val featureIndex: Int,
    val threshold: Double,
    val left: TreeNode? = null,
    val right: TreeNode? = null,
    val value: Int? = null
) {
    // override fun toString(): String = " ".repeat(9) + toString(10)

    fun toString(nest: Int): String {
        val map = mutableMapOf<String, Any?>().apply {
            this["i"] = featureIndex
            this["t"] = threshold
            if (value != null) this["v"] = value
        }
        val text = "[" + map.toList().joinToString(",") { "${it.first}=${it.second}" } + "]"
        val leftText = if (left != null) {
            "\n${" ".repeat(nest - 1)}<-${left.toString(nest - 1)}"
        } else ""
        val rightText = if (right != null) {
            "\n${" ".repeat(nest + 1)}->${right.toString(nest + 1)}"
        } else ""
        return text /*+ leftText*/ + rightText
    }
}
