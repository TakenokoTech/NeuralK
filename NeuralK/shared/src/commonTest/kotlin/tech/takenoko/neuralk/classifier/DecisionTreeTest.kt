package tech.takenoko.neuralk.classifier

import tech.takenoko.neuralk.classifier.DecisionTree.Data
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class DecisionTreeTest {

    private lateinit var decisionTree: DecisionTree

    @BeforeTest
    fun setup() {
        decisionTree = DecisionTree(1)
    }

    @Test
    fun testFitAndPredict() {
        val data = listOf(
            Data(listOf(1.0, 2.0), 0),
            Data(listOf(2.0, 1.0), 1),
            Data(listOf(3.0, 3.0), 0),
            Data(listOf(4.0, 0.5), 1)
        )
        decisionTree.fit(data)
        assertEquals(data[0].label, decisionTree.predict(data[0].features))
        assertEquals(data[1].label, decisionTree.predict(data[1].features))
        assertEquals(data[2].label, decisionTree.predict(data[2].features))
        assertEquals(data[3].label, decisionTree.predict(data[3].features))
    }

    @Test
    fun testPredictNotFitted() {
        assertEquals(-1, decisionTree.predict(listOf(1.0, 2.0)))
    }

    @Test
    fun testBuildTree() {
        val data = listOf(
            Data(listOf(1.0, 3.0), 0),
            Data(listOf(2.0, 2.0), 1),
            Data(listOf(3.0, 3.0), 0),
            Data(listOf(4.0, 2.0), 1)
        )
        val tree = decisionTree.buildTree(data, depth = 0)
        assertEquals(
            TreeNode(
                featureIndex = 1,
                threshold = 2.0,
                left = TreeNode(-1, -1.0, value = 1),
                right = TreeNode(-1, -1.0, value = 0),
            ), tree
        )
    }

    @Test
    fun testFindBestSplit() {
        val data1 = listOf(
            Data(listOf(1.0, 2.0), 0),
            Data(listOf(2.0, 1.0), 1),
            Data(listOf(3.0, 2.0), 0),
            Data(listOf(4.0, 1.0), 1)
        )
        val actual1 = decisionTree.findBestSplit(data1)
        assertEquals(Pair(1, 1.0), actual1)

        val data2 = listOf(
            Data(listOf(1.0, 3.0), 0),
            Data(listOf(2.0, 2.0), 1),
            Data(listOf(3.0, 1.0), 0),
            Data(listOf(4.0, 0.5), 1)
        )
        val actual2 = decisionTree.findBestSplit(data2)
        assertEquals(Pair(0, 1.0), actual2)

        val data3 = listOf(
            Data(listOf(1.0, 2.0), 0),
            Data(listOf(2.0, 1.0), 0),
            Data(listOf(3.0, 2.0), 1),
            Data(listOf(4.0, 1.0), 1)
        )
        val actual3 = decisionTree.findBestSplit(data3)
        assertEquals(Pair(0, 2.0), actual3)

        val data4 = listOf(
            Data(listOf(1.0, 2.0), 1),
            Data(listOf(2.0, 3.0), 1),
            Data(listOf(3.0, 4.0), 0),
            Data(listOf(4.0, 5.0), 0)
        )
        val actual4 = decisionTree.findBestSplit(data4)
        assertEquals(Pair(0, 2.0), actual4)

        val actual5 = decisionTree.findBestSplit(listOf())
        assertEquals(Pair(-1, Double.MAX_VALUE), actual5)
    }

    @Test
    fun testCalculateGini() {
        // テストケース1: シンプルなデータセット
        val data1 = listOf(
            Data(listOf(1.0, 2.0), 0),
            Data(listOf(1.1, 2.1), 0),
            Data(listOf(2.0, 2.2), 1),
            Data(listOf(2.1, 2.3), 1)
        )
        val gini1 = decisionTree.calculateGini(data1, 0, 1.5)
        assertEquals(0.0, gini1, 0.0001)

        // テストケース2: 混在するデータセット
        val data2 = listOf(
            Data(listOf(1.0, 2.0), 0),
            Data(listOf(1.5, 2.1), 1),
            Data(listOf(2.0, 2.2), 0),
            Data(listOf(2.5, 2.3), 1)
        )
        val gini2 = decisionTree.calculateGini(data2, 0, 1.5)
        assertEquals(0.5, gini2, 0.0001)

        // テストケース3: 異なる閾値でのテスト
        val gini3 = decisionTree.calculateGini(data2, 0, 2.0)
        assertEquals(0.3333, gini3, 0.0001)

        // テストケース4: 異なる特徴量インデックス
        val data3 = listOf(
            Data(listOf(1.0, 2.0), 0),
            Data(listOf(2.0, 1.0), 1),
            Data(listOf(3.0, 2.0), 0),
            Data(listOf(4.0, 1.0), 1)
        )
        val gini4 = decisionTree.calculateGini(data3, 1, 1.5)
        assertEquals(0.0, gini4, 0.0001)

        // テストケース5: 空のデータリスト
        val dataEmpty = listOf<Data>()
        val giniEmpty = decisionTree.calculateGini(dataEmpty, 0, 1.5)
        assertEquals(0.0, giniEmpty, 0.0001)
    }

    @Test
    fun testCalculateGiniImpurity() {
        // テストケース1: 完全に不純なデータ
        val data1 = listOf(
            Data(listOf(1.0, 2.0), 1),
            Data(listOf(1.1, 2.1), 1),
            Data(listOf(1.2, 2.2), 1),
            Data(listOf(1.3, 2.3), 1)
        )
        val gini1 = decisionTree.calculateGiniImpurity(data1)
        assertEquals(0.0, gini1, 0.0001)

        // テストケース2: 完全に均等なデータ
        val data2 = listOf(
            Data(listOf(1.0, 2.0), 1),
            Data(listOf(1.1, 2.1), 2),
            Data(listOf(1.2, 2.2), 1),
            Data(listOf(1.3, 2.3), 2)
        )
        val gini2 = decisionTree.calculateGiniImpurity(data2)
        assertEquals(0.5, gini2, 0.0001)

        // テストケース3: ほぼ不純なデータ
        val data3 = listOf(
            Data(listOf(1.0, 2.0), 1),
            Data(listOf(1.1, 2.1), 1),
            Data(listOf(1.2, 2.2), 1),
            Data(listOf(1.3, 2.3), 2)
        )
        val gini3 = decisionTree.calculateGiniImpurity(data3)
        assertEquals(0.375, gini3, 0.0001)

        // テストケース4: 異なるラベルが混在
        val data4 = listOf(
            Data(listOf(1.0, 2.0), 1),
            Data(listOf(1.1, 2.1), 2),
            Data(listOf(1.2, 2.2), 3),
            Data(listOf(1.3, 2.3), 4)
        )
        val gini4 = decisionTree.calculateGiniImpurity(data4)
        assertEquals(0.75, gini4, 0.0001)

        // テストケース5: 空のデータ
        val gini5 = decisionTree.calculateGiniImpurity(listOf())
        assertEquals(0.0, gini5, 0.0001)
    }

    @Test
    fun testGetMajorityClass() {
        // テストケース1: ラベルが0と1のデータが同数
        val data1 = listOf(
            Data(listOf(1.0, 2.0), 0),
            Data(listOf(2.0, 1.0), 1)
        )
        assertEquals(0, decisionTree.getMajorityClass(data1)) // 同数の場合は最初に見つかったラベルが返される仮定

        // テストケース2: ラベルが0のデータが多い
        val data2 = listOf(
            Data(listOf(1.0, 2.0), 0),
            Data(listOf(2.0, 1.0), 0),
            Data(listOf(3.0, 2.0), 1)
        )
        assertEquals(0, decisionTree.getMajorityClass(data2))

        // テストケース3: ラベルが1のデータが多い
        val data3 = listOf(
            Data(listOf(1.0, 2.0), 1),
            Data(listOf(2.0, 1.0), 1),
            Data(listOf(3.0, 2.0), 0)
        )
        assertEquals(1, decisionTree.getMajorityClass(data3))

        // テストケース4: 全てのデータが同じラベル
        val data4 = listOf(
            Data(listOf(1.0, 2.0), 1),
            Data(listOf(2.0, 3.0), 1),
            Data(listOf(3.0, 4.0), 1)
        )
        assertEquals(1, decisionTree.getMajorityClass(data4))

        // テストケース5: 空のデータセット
        val data5 = emptyList<Data>()
        assertEquals(-1, decisionTree.getMajorityClass(data5))
    }
}
