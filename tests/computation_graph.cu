#include "utest.h"

#include "vika.cuh"

#define EXPECT_EXTENTS(actual, expected) EXPECT_TRUE((actual) == (expected))

UTEST(computation_graph, input_node)
{
    using namespace vika;

    ComputationGraph graph{4};
    const auto id = graph.input({8, 8, 1});

    EXPECT_EQ(graph.nodes.size(), 1u);
    EXPECT_EQ(id.value, 0u);

    const auto &node = graph.nodes[0];
    EXPECT_TRUE(std::holds_alternative<InputSpec>(node.spec));
    EXPECT_EQ(node.inputs.size(), 0u);
    EXPECT_EXTENTS(node.output_extents, Extents({4, 8, 8, 1}));
}

UTEST(computation_graph, input_node_id_increments)
{
    using namespace vika;

    ComputationGraph graph{2};
    const auto id0 = graph.input({4, 4, 3});
    const auto id1 = graph.input({4, 4, 3});

    EXPECT_EQ(graph.nodes.size(), 2u);
    EXPECT_EQ(id0.value, 0u);
    EXPECT_EQ(id1.value, 1u);
}

UTEST(computation_graph, xor_graph_shape)
{
    using namespace vika;

    // input [4, 2] -> dense [4, 8] -> sigmoid [4, 8] -> dense [4, 1] -> sigmoid [4, 1]
    ComputationGraph graph{4};
    auto x = graph.input({2});
    x = graph.dense(x, 8, 42).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, 43).unwrap();
    x = graph.sigmoid(x).unwrap();

    EXPECT_EQ(graph.nodes.size(), 5u);

    const Extents expected[] = {{4, 2}, {4, 8}, {4, 8}, {4, 1}, {4, 1}};
    for (usize i = 0; i < 5; ++i)
    {
        EXPECT_EXTENTS(graph.nodes[i].output_extents, expected[i]);
    }

    EXPECT_EQ(graph.nodes[0].inputs.size(), 0u);
    for (usize i = 1; i < 5; ++i)
    {
        EXPECT_EQ(graph.nodes[i].inputs.size(), 1u);
        EXPECT_EQ(graph.nodes[i].inputs[0].value, i - 1);
    }
}

UTEST(computation_graph, line_cnn_graph_shape)
{
    using namespace vika;

    // input [8, 8, 8, 1]
    // -> conv2d(3,3,8,stride=1,pad=0) -> [8, 6, 6, 8]
    // -> maxpool2d(2,2,stride=2)      -> [8, 3, 3, 8]
    // -> flatten                      -> [8, 72]
    // -> dense(16)                    -> [8, 16]
    // -> sigmoid                      -> [8, 16]
    // -> dense(1)                     -> [8, 1]
    // -> sigmoid                      -> [8, 1]
    ComputationGraph graph{8};
    auto x = graph.input({8, 8, 1});
    x = graph.conv2d(x, 3, 3, 8, 1, 0, 42).unwrap();
    x = graph.maxpool2d(x, 2, 2, 2).unwrap();
    x = graph.flatten(x).unwrap();
    x = graph.dense(x, 16, 43).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, 44).unwrap();
    x = graph.sigmoid(x).unwrap();

    EXPECT_EQ(graph.nodes.size(), 8u);

    const Extents expected[] = {
        {8, 8, 8, 1}, {8, 6, 6, 8}, {8, 3, 3, 8}, {8, 72}, {8, 16}, {8, 16}, {8, 1}, {8, 1},
    };
    for (usize i = 0; i < 8; ++i)
    {
        EXPECT_EXTENTS(graph.nodes[i].output_extents, expected[i]);
    }
}

UTEST(computation_graph, dense_invalid_node_id)
{
    using namespace vika;
    ComputationGraph graph{4};
    const auto result = graph.dense(NodeId{99}, 8, 42);
    EXPECT_TRUE(result.is_error());
}

UTEST(computation_graph, dense_wrong_rank)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({8, 8, 1}); // rank 4, not rank 2
    const auto result = graph.dense(x, 8, 42);
    EXPECT_TRUE(result.is_error());
}

UTEST(computation_graph, conv2d_kernel_too_large)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({4, 4, 1});
    const auto result = graph.conv2d(x, 8, 8, 16, 1, 0, 42);
    EXPECT_TRUE(result.is_error());
}

UTEST(computation_graph, maxpool2d_pool_too_large)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({4, 4, 1});
    const auto result = graph.maxpool2d(x, 8, 8, 1);
    EXPECT_TRUE(result.is_error());
}

UTEST(computation_graph, sigmoid_passthrough_extents)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({8, 8, 3});
    x = graph.sigmoid(x).unwrap();
    EXPECT_EXTENTS(graph.nodes[1].output_extents, graph.nodes[0].output_extents);
}

UTEST(computation_graph, flatten_shape)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({6, 6, 8}); // [4, 6, 6, 8]
    x = graph.flatten(x).unwrap();
    EXPECT_EXTENTS(graph.nodes[1].output_extents, Extents({4, 288})); // 6*6*8 = 288
}

UTEST(computation_graph, add_shape)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({8});
    auto a = graph.dense(x, 16, 42).unwrap();
    auto b = graph.dense(x, 16, 43).unwrap();
    auto sum = graph.add({a, b}).unwrap();

    EXPECT_EXTENTS(graph.nodes[sum.value].output_extents, Extents({4, 16}));
    EXPECT_EQ(graph.nodes[sum.value].inputs.size(), 2u);
    EXPECT_EQ(graph.nodes[sum.value].inputs[0].value, a.value);
    EXPECT_EQ(graph.nodes[sum.value].inputs[1].value, b.value);
}

UTEST(computation_graph, add_n_ary)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({8});
    auto a = graph.dense(x, 16, 42).unwrap();
    auto b = graph.dense(x, 16, 43).unwrap();
    auto c = graph.dense(x, 16, 44).unwrap();
    auto sum = graph.add({a, b, c}).unwrap();

    EXPECT_EXTENTS(graph.nodes[sum.value].output_extents, Extents({4, 16}));
    EXPECT_EQ(graph.nodes[sum.value].inputs.size(), 3u);
}

UTEST(computation_graph, add_too_few_inputs)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({8});
    auto a = graph.dense(x, 16, 42).unwrap();
    const auto result = graph.add({a});
    EXPECT_TRUE(result.is_error());
}

UTEST(computation_graph, add_invalid_node_id)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({8});
    auto a = graph.dense(x, 16, 42).unwrap();
    const auto result = graph.add({a, NodeId{99}});
    EXPECT_TRUE(result.is_error());
}

UTEST(computation_graph, add_shape_mismatch)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({8});
    auto a = graph.dense(x, 16, 42).unwrap();
    auto b = graph.dense(x, 8, 43).unwrap();
    const auto result = graph.add({a, b});
    EXPECT_TRUE(result.is_error());
}
