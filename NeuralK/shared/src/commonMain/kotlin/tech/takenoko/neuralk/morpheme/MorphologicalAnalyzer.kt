package tech.takenoko.neuralk.morpheme

data class Morpheme(
    val surface: String,
    val partOfSpeech: String
)

class Dictionary(
    private val entries: Map<String, Morpheme>
) {
    fun lookup(word: String): Morpheme? = entries[word]

    companion object {
        fun load(dic: String): Dictionary = dic
            .split("\n")
            .map { it.split("\t") }
            .associate { (s, p) -> s to Morpheme(s, p) }
            .let(::Dictionary)
    }
}

class MorphologicalAnalyzer(private val dictionary: Dictionary) {

    // 最長一致法で形態素を抽出
    inner class LongestMatch private constructor(
        var morpheme: Morpheme? = null,
        var length: Int = 0
    ) {
        constructor(state: AnalyzeState) : this() {
            for (j in state.index + 1..state.text.length) {
                val substring = state.text.substring(state.index, j)
                dictionary.lookup(substring)?.let {
                    morpheme = it
                    length = j - state.index
                }
            }
        }
    }

    inner class AnalyzeState(
        val text: String,
        private val _morphemes: MutableList<Morpheme> = mutableListOf()
    ) {
        val morphemes get() = _morphemes
        val index get() = morphemes.sumOf { it.surface.length }
        fun update(longestMatch: LongestMatch) {
            val longestMatchMorpheme = longestMatch.morpheme
            if (longestMatchMorpheme != null) {
                _morphemes.add(longestMatchMorpheme)
            } else {
                _morphemes.add(Morpheme(text.substring(index, index + 1), "UNKNOWN"))
            }
        }
    }

    fun analyze(text: String): List<Morpheme> {
        val state = AnalyzeState(text)
        while (state.index < text.length) {
            val longestMatch = LongestMatch(state)
            state.update(longestMatch)
        }
        return state.morphemes
    }
}
