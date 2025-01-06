package tech.takenoko.neuralk.nn.tensor

import org.junit.experimental.runners.Enclosed
import org.junit.runner.RunWith
import kotlin.test.Test

@RunWith(Enclosed::class)
class TensorTest {

    class PlusTest {
        @Test
        fun tensor2D() {
            val tensor = Tensor2D(
                arrayOf(
                    arrayOf(1F, 2F),
                    arrayOf(3F, 4F),
                    arrayOf(5F, 6F),
                ),
            )
            val result = tensor + Tensor0D(1F)
            result as Tensor2D
            assert(result.data[0][0] == 2F)
            assert(result.data[0][1] == 3F)
            assert(result.data[1][0] == 4F)
            assert(result.data[1][1] == 5F)
            assert(result.data[2][0] == 6F)
            assert(result.data[2][1] == 7F)

            val result2 = tensor + Tensor1D(arrayOf(1F, 2F))
            result2 as Tensor2D
            assert(result2.data[0][0] == 2F)
            assert(result2.data[0][1] == 4F)
            assert(result2.data[1][0] == 4F)
            assert(result2.data[1][1] == 6F)
            assert(result2.data[2][0] == 6F)
            assert(result2.data[2][1] == 8F)

            val result3 = tensor + Tensor2D(
                arrayOf(
                    arrayOf(1F, 2F),
                    arrayOf(3F, 4F),
                    arrayOf(5F, 6F),
                ),
            )
            result3 as Tensor2D
            assert(result3.data[0][0] == 2F)
            assert(result3.data[0][1] == 4F)
            assert(result3.data[1][0] == 6F)
            assert(result3.data[1][1] == 8F)
            assert(result3.data[2][0] == 10F)
            assert(result3.data[2][1] == 12F)
        }
    }

    class TimesTest {
        @Test
        fun tensor2D() {
            val tensor = Tensor2D(
                arrayOf(
                    arrayOf(1F),
                    arrayOf(2F),
                    arrayOf(3F),
                ),
            )
            val result = tensor * Tensor0D(2F)
            result as Tensor2D
            assert(result.data[0][0] == 2F)
            assert(result.data[1][0] == 4F)
            assert(result.data[2][0] == 6F)

            val result2 = tensor * Tensor1D(arrayOf(2F, 3F))
            result2 as Tensor2D
            assert(result2.data[0][0] == 2F)
            assert(result2.data[0][1] == 2F)
            assert(result2.data[0][0] == 2F)

            val result3 = tensor * Tensor2D(
                arrayOf(
                    arrayOf(4F, 5F, 6F),
                ),
            )
            result3 as Tensor2D
            assert(result3.data[0][0] == 4F)
            assert(result3.data[0][1] == 5F)
            assert(result3.data[0][2] == 6F)
            assert(result3.data[1][0] == 8F)
            assert(result3.data[1][1] == 10F)
            assert(result3.data[1][2] == 12F)
            assert(result3.data[2][0] == 12F)
            assert(result3.data[2][1] == 15F)
            assert(result3.data[2][2] == 18F)
        }
    }
}
