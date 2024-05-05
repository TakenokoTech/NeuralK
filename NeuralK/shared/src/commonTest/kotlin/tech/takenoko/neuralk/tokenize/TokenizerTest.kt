package tech.takenoko.neuralk.tokenize

import tech.takenoko.neuralk.entity.TextUnion
import tech.takenoko.neuralk.entity.TokenIdUnion
import kotlin.test.Test
import kotlin.test.assertEquals

class TokenizerTest {
    @Test
    fun encode() {
        val text = """
        <s><|system|> I am an excellent assistant AI.<|end|>
        """.trimIndent()
        val actual = Tokenizer(sampleJson).encode(TextUnion.StringType(text))
        val expected = listOf(
            1, 32006, 306, 626, 385,
            15129, 20255, 319, 29902, 29889, 32007
        )
        assertEquals(TokenIdUnion.IntListType(expected), actual)
    }

    @Test
    fun decode() {
        val actual = Tokenizer(sampleJson).decode(TokenIdUnion.IntType(1))
        assertEquals(TextUnion.StringType(""), actual)
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
