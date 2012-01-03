//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLES_BFS_GRAPH_SINGLE_LOCALITY_JAN_01_2012_0622PM)
#define HPX_EXAMPLES_BFS_GRAPH_SINGLE_LOCALITY_JAN_01_2012_0622PM

#include <vector>
#include <cstdlib>

namespace bfs_graph { namespace single_locality
{
    // run benchmarks for all single threaded graph variants
    void run_benchmarks(bool validate, std::size_t grainsize,
        std::vector<std::pair<std::size_t, std::size_t> > const& edgelist,
        std::vector<std::size_t> const& searchroots);
}}

#endif
