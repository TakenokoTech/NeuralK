package tech.takenoko.neuralk.load

import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi
import kotlin.test.Test

class LoaderTest {

    @OptIn(ExperimentalEncodingApi::class)
    @Test
    fun load() {
        Loader().load(Base64.decode(tfmodel.encodeToByteArray()))
    }
}