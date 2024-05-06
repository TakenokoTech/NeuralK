package tech.takenoko.neuralk.tokenize

import kotlinx.serialization.json.Json
import tech.takenoko.neuralk.entity.TextUnion
import tech.takenoko.neuralk.entity.TokenIdUnion
import tech.takenoko.neuralk.entity.TokenizerJson

class Tokenizer(
    private val json: TokenizerJson
) {
    constructor(
        json: String
    ) : this(Json.decodeFromString<TokenizerJson>(json))

    fun encode(
        text: TextUnion,
        maxLength: Int? = null,
    ): TokenIdUnion.IntListType {
        val vocab1 = json.addedTokens.associate { it.content to it.id }
        val vocab2 = json.model.vocab.toList().associate { it.first to it.second }
        val (ans, tmp) = mutableListOf<Int>() to mutableListOf<String>()
        tokenizeBySubWords(text.concatText(), vocab1.keys.toSet()).forEach { word ->
            val id = vocab1[word]
            if (id != null) {
                val tokenIds = tokenizeBySubWords(tmp.joinToString(""), vocab2)
                ans.addAll(tokenIds + id)
                tmp.clear()
            } else {
                tmp.add(word)
            }
        }
        return TokenIdUnion.IntListType(ans)
    }

    fun decode(
        tokenIds: TokenIdUnion,
        skipSpecialTokens: Boolean = false,
        cleanUpTokenizationSpaces: Boolean = false,
    ): TextUnion.StringType {
        val vocab1 = json.addedTokens.associate { it.id to it.content }
        val vocab2 = json.model.vocab.toList().associate { it.second to it.first }
        val text = tokenIds.concatTokenIds()
            .map((vocab1 + vocab2)::get)
            .joinToString("")
            .replace("▁", " ")
        return TextUnion.StringType(text)
    }

    fun tokenizeBySubWords(text: String, vocabs: Map<String, Int>): List<Int> {
        return tokenizeBySubWords(text, vocabs.keys.toSet()).map { vocabs.getOrElse(it) { 0 } }
    }

    fun tokenizeBySubWords(text: String, subWords: Set<String>): List<String> {
        return mutableListOf<String>().also { tokens ->
            var i = 0
            while (i < text.length) {
                var (j, foundSubWord) = text.length to false
                while (!foundSubWord && j > i) {
                    val subWord = text.substring(i, j).replaceFirst("^\\s+".toRegex(), "▁")
                    if (subWords.contains(subWord)) {
                        tokens.add(subWord)
                        i += subWord.length
                        foundSubWord = true
                    }
                    j--
                }
                if (!foundSubWord) tokens.add(text[i++].toString())
            }
        }
    }
}
