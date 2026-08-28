#include "utest.h"

#include "comparison.cuh"
#include "vika.cuh"
#include <algorithm>
#include <memory>

UTEST(topological_sort, no_cyle)
{
    using namespace vika;
    using namespace std;

    const usize node_count = 6;
    unique_ptr<int[]> nodes = make_unique<int[]>(node_count);

    auto ptr = nodes.get();
    AdjacencyGraph<int *> adj;
    adj[ptr + 5] = {ptr + 2, ptr + 0};
    adj[ptr + 4] = {ptr + 0, ptr + 1};
    adj[ptr + 2] = {ptr + 3};
    adj[ptr + 3] = {ptr + 1};

    const auto actual = topological_sort(adj).unwrap();
    // 4 and 5 (both indegree 0), and 0 and 2 (once available), are unordered pairs in this DAG;
    // the min-heap always advances the smallest available node, so this is the one deterministic
    // order rather than one of several the old hash-bucket-ordered queue could have produced.
    const vector<int *> expected = {ptr + 4, ptr + 5, ptr + 0, ptr + 2, ptr + 3, ptr + 1};
    EXPECT_TRUE(equal(begin(actual), end(actual), begin(expected), end(expected)));
}

UTEST(topological_sort, cyle)
{
    using namespace vika;
    using namespace std;

    const usize node_count = 6;
    unique_ptr<int[]> nodes = make_unique<int[]>(node_count);

    auto ptr = nodes.get();
    AdjacencyGraph<int *> adj;
    adj[ptr + 5] = {ptr + 2, ptr + 0};
    adj[ptr + 4] = {ptr + 0, ptr + 1};
    adj[ptr + 2] = {ptr + 3};
    adj[ptr + 3] = {ptr + 1};
    adj[ptr + 1] = {ptr + 3};

    const auto actual = topological_sort(adj);
    EXPECT_TRUE(failed_with(actual, ErrorKind::Graph, "Cycle detected"));
}
