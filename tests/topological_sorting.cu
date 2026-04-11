#include "utest.h"

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
    AdjecencyGraph<int *> adj;
    adj[ptr + 5] = {ptr + 2, ptr + 0};
    adj[ptr + 4] = {ptr + 0, ptr + 1};
    adj[ptr + 2] = {ptr + 3};
    adj[ptr + 3] = {ptr + 1};

    const auto actual = topological_sort(adj).unwrap();
    const vector<int *> expected = {ptr + 5, ptr + 4, ptr + 2, ptr + 0, ptr + 3, ptr + 1};
    EXPECT_TRUE(equal(begin(actual), end(actual), begin(expected), end(expected)));
}

UTEST(topological_sort, cyle)
{
    using namespace vika;
    using namespace std;

    const usize node_count = 6;
    unique_ptr<int[]> nodes = make_unique<int[]>(node_count);

    auto ptr = nodes.get();
    AdjecencyGraph<int *> adj;
    adj[ptr + 5] = {ptr + 2, ptr + 0};
    adj[ptr + 4] = {ptr + 0, ptr + 1};
    adj[ptr + 2] = {ptr + 3};
    adj[ptr + 3] = {ptr + 1};
    adj[ptr + 1] = {ptr + 3};

    const auto actual = topological_sort(adj);
    EXPECT_TRUE(actual.is_error());
}
