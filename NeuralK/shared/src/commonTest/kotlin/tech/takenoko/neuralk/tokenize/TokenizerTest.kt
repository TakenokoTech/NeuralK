package tech.takenoko.neuralk.tokenize

import tech.takenoko.neuralk.entity.TokenizerText
import tech.takenoko.neuralk.entity.TokenizerTokenId
import kotlin.test.Test
import kotlin.test.assertEquals

class TokenizerTest {
    @Test
    fun encode() {
        val text = """
        <s><|system|> I am an excellent assistant AI.<|end|>
        """.trimIndent()
        val actual = Tokenizer(sampleJson).encode(TokenizerText.StringType(text))
        val expected = listOf(
            1, 32006, 306, 626, 385,
            15129, 20255, 319, 29902, 29889, 32007
        )
        assertEquals(TokenizerTokenId.IntListType(expected), actual)
    }

    @Test
    fun decode() {
        val tokenIds = TokenizerTokenId.IntListType(
            listOf(
                1, 32006, 306, 626, 385,
                15129, 20255, 319, 29902, 29889, 32007
            )
        )
        val actual = Tokenizer(sampleJson).decode(tokenIds)
        val expected = """
        <s><|system|> I am an excellent assistant AI.<|end|>
        """.trimIndent()
        assertEquals(TokenizerText.StringType(expected), actual)
    }

    @Test
    fun tokenizeBySubWords() {
        println(
            Tokenizer(sampleJson).tokenizeBySubWords(
                "abc", setOf("a", "b", "ab", "bc")
            )
        )
    }
}
