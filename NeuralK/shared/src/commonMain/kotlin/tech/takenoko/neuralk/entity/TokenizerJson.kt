package tech.takenoko.neuralk.entity

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class TokenizerJson(
    @SerialName("added_tokens")
    val addedTokens: List<AddedToken>,
    val decoder: Decoder,
    val model: Model,
    val normalizer: Normalizer,
    val padding: String?,
    @SerialName("post_processor")
    val postProcessor: PostProcessor,
    @SerialName("pre_tokenizer")
    val preTokenizer: String?,
    val truncation: String?,
    val version: String
) {
    @Serializable
    data class AddedToken(
        val content: String,
        val id: Int,
        val lstrip: Boolean,
        val normalized: Boolean,
        val rstrip: Boolean,
        val single_word: Boolean,
        val special: Boolean
    )

    @Serializable
    data class Decoder(val type: String, val decoders: List<DecoderItem>) {
        @Serializable
        data class DecoderItem(
            val type: String,
            val content: String? = null,
            val pattern: Map<String, String> = mapOf(),
            val start: Int? = null,
            val stop: Int? = null,
        )
    }

    @Serializable
    data class Model(
        val byte_fallback: Boolean,
        val continuing_subword_prefix: String?,
        val dropout: String?,
        val end_of_word_suffix: String?,
        val fuse_unk: Boolean,
        val merges: List<String>,
        val type: String,
        val unk_token: String,
        val vocab: Map<String, Int>
    )

    @Serializable
    data class Normalizer(
        val type: String,
        val normalizers: List<NormalizerItem>
    ) {
        @Serializable
        data class NormalizerItem(
            val type: String,
            val content: String? = null,
            val pattern: Map<String, String> = mapOf(),
            val prepend: String? = null,
        )
    }

    @Serializable
    data class PostProcessor(
        val type: String,
        val pair: List<Pair>,
        val single: List<Single>,
        @SerialName("special_tokens")
        val specialTokens: Map<String, String>,
    ) {
        @Serializable
        data class Pair(
            @SerialName("Sequence")
            val sequence: Sequence
        )

        @Serializable
        data class Single(
            @SerialName("Sequence")
            val sequence: Sequence
        )

        @Serializable
        data class Sequence(
            val id: String,
            @SerialName("type_id")
            val typeId: Int
        )
    }
}
