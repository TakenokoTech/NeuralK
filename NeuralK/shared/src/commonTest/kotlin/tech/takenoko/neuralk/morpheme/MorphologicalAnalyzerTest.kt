package tech.takenoko.neuralk.morpheme

import kotlin.test.Test

class MorphologicalAnalyzerTest {

    @Test
    fun analyze() {
        val dic = """
        私	名詞,代名詞,一般,*,*,*,私,ワタシ,ワタシ
        は	助詞,係助詞,*,*,*,*,は,ハ,ワ
        日本	名詞,固有名詞,地域,国,*,*,日本,ニッポン,ニッポン
        の	助詞,連体化,*,*,*,*,の,ノ,ノ
        IT	名詞,一般,*,*,*,*,*
        企業	名詞,一般,*,*,*,*,企業,キギョウ,キギョー
        で	助詞,格助詞,一般,*,*,*,で,デ,デ
        働く	動詞,自立,*,*,五段・カ行イ音便,基本形,働く,ハタラク,ハタラク
        エンジニア	名詞,一般,*,*,*,*,エンジニア,エンジニア,エンジニア
        です	助動詞,*,*,*,特殊・デス,基本形,です,デス,デス
        。	記号,句点,*,*,*,*,。,。,。
        """.trimIndent()
        val dictionary = Dictionary.load(dic)
        val analyzer = MorphologicalAnalyzer(dictionary)
        val text = "私は日本のIT企業で働くエンジニアです。"
        val result = analyzer.analyze(text)
        result.forEach { println("${it.surface} (${it.partOfSpeech})") }
    }
}
