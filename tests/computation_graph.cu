#include "utest.h"

#include "vika.cuh"

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
    EXPECT_EQ(node.output_extents.size(), 4u);
    EXPECT_EQ(node.output_extents[0], 4u);
    EXPECT_EQ(node.output_extents[1], 8u);
    EXPECT_EQ(node.output_extents[2], 8u);
    EXPECT_EQ(node.output_extents[3], 1u);
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
