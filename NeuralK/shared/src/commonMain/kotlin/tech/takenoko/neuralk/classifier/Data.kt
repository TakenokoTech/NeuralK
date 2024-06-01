package tech.takenoko.neuralk.classifier

sealed class Data {
    data class Frame(val map: Map<String, List<Any>>) : Data() {
        val isEmpty = map.isEmpty()
        val columns: List<Any> = map.keys.toList()
        fun labels(column: String) = map[column]
            ?.map { it as Int }
            .orEmpty()

        fun features(columns: List<String>, index: Int) = columns
            .map { map[it].orEmpty()[index] }
            .map { it as Double }
    }
}
