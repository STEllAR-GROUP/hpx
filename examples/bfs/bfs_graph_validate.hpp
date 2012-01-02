//  Copyright (c) 2011 Matthew Anderson
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLES_BFS_GRAPH_VALIDATE_JAN_01_2012_0609PM)
#define HPX_EXAMPLES_BFS_GRAPH_VALIDATE_JAN_01_2012_0609PM

#include <vector>
#include <cstdlib>
#include <algorithm>

namespace bfs_graph
{
    ///////////////////////////////////////////////////////////////////////////
    // this routine validates the graph
    int validate_graph(std::size_t searchkey,
        std::vector<std::size_t> const& parents,
        std::vector<std::pair<std::size_t, std::size_t> > const& edgelist,
        std::size_t& num_nodes);

    ///////////////////////////////////////////////////////////////////////////
    void print_statistics(
        std::vector<double> const& kernel2_time,
        std::vector<std::size_t> const& kernel2_nedge);
}

#endif
