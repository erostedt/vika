#include "utest.h"

#include "vika.cuh"

UTEST(model, xor_compile_execution_order)
{
    using namespace vika;

    // input -> dense -> sigmoid -> dense -> sigmoid
    ComputationGraph graph{4};
    auto x = graph.input({2});
    x = graph.dense(x, 8, 42).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, 43).unwrap();
    x = graph.sigmoid(x).unwrap();

    auto model = graph.compile(x).unwrap();

    EXPECT_EQ(model.nodes.size(), 5u);
    EXPECT_EQ(model.execution_order.size(), 5u);
    EXPECT_EQ(model.input_node.value, 0u);
    EXPECT_EQ(model.output_node.value, 4u);

    // input node is first, output node is last
    EXPECT_EQ(model.execution_order.front().value, 0u);
    EXPECT_EQ(model.execution_order.back().value, 4u);

    // each node appears after all its predecessors
    for (usize i = 0; i < model.execution_order.size(); ++i)
    {
        const auto &node = model.nodes[model.execution_order[i].value];
        for (const auto &pred : node.inputs)
        {
            bool pred_seen = false;
            for (usize j = 0; j < i; ++j)
            {
                if (model.execution_order[j].value == pred.value)
                {
                    pred_seen = true;
                    break;
                }
            }
            EXPECT_TRUE(pred_seen);
        }
    }
}

UTEST(model, line_cnn_compile_execution_order)
{
    using namespace vika;

    ComputationGraph graph{8};
    auto x = graph.input({8, 8, 1});
    x = graph.conv2d(x, 3, 3, 8, 1, 0, 42).unwrap();
    x = graph.maxpool2d(x, 2, 2, 2).unwrap();
    x = graph.flatten(x).unwrap();
    x = graph.dense(x, 16, 43).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, 44).unwrap();
    x = graph.sigmoid(x).unwrap();

    auto model = graph.compile(x).unwrap();

    EXPECT_EQ(model.nodes.size(), 8u);
    EXPECT_EQ(model.execution_order.size(), 8u);
    EXPECT_EQ(model.input_node.value, 0u);
    EXPECT_EQ(model.output_node.value, 7u);

    EXPECT_EQ(model.execution_order.front().value, 0u);
    EXPECT_EQ(model.execution_order.back().value, 7u);
}

UTEST(model, compile_invalid_output_node)
{
    using namespace vika;
    ComputationGraph graph{4};
    graph.input({2});
    const auto result = graph.compile(NodeId{99});
    EXPECT_TRUE(result.is_error());
}

UTEST(model, compile_no_input_node)
{
    using namespace vika;
    // manually push a node with no InputSpec to simulate missing input
    ComputationGraph graph{4};
    auto x = graph.input({2});
    x = graph.dense(x, 4, 42).unwrap();
    // replace the input node spec with a DenseSpec to remove the InputSpec
    graph.nodes[0].spec = DenseSpec{2, 0};
    const auto result = graph.compile(x);
    EXPECT_TRUE(result.is_error());
}

UTEST(model, compile_multiple_input_nodes)
{
    using namespace vika;
    ComputationGraph graph{4};
    graph.input({2});
    auto x = graph.input({2});
    x = graph.dense(x, 4, 42).unwrap();
    const auto result = graph.compile(x);
    EXPECT_TRUE(result.is_error());
}

UTEST(model, compile_moves_nodes_out_of_graph)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({2});
    x = graph.dense(x, 4, 42).unwrap();
    auto model = graph.compile(x).unwrap();
    // nodes moved into model
    EXPECT_EQ(model.nodes.size(), 2u);
}
