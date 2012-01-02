//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLES_BFS_GRAPH_UTIL_JAN_01_2012_0611PM)
#define HPX_EXAMPLES_BFS_GRAPH_UTIL_JAN_01_2012_0611PM

#include <fstream>
#include <algorithm>
#include <cstdlib>

namespace bfs_graph
{
    ///////////////////////////////////////////////////////////////////////////
    inline std::size_t
    max_node(std::size_t n1, std::pair<std::size_t, std::size_t> const& p)
    {
        return (std::max)((std::max)(n1, p.first), p.second);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline bool read_edge_list(std::string const& graphfile,
        std::vector<std::pair<std::size_t, std::size_t> >& edgelist)
    {
        std::ifstream edges(graphfile.c_str());
        if (edges.is_open()) {
            std::size_t node, neighbor;
            while (edges >> node >> neighbor)
                edgelist.push_back(std::make_pair(node+1, neighbor+1));
            return edges.eof();
        }

        std::cerr << " File " << graphfile
                  << " not found! Exiting... " << std::endl;
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline bool read_search_node_list(std::string const& searchfile,
        std::vector<std::size_t>& searchroots)
    {
        std::ifstream nodes(searchfile.c_str());
        if (nodes.is_open()) {
            std::size_t root;
            while (nodes >> root)
                searchroots.push_back(root+1);
            return nodes.eof();
        }

        std::cerr << " File " << searchfile
                  << " not found! Exiting... " << std::endl;
        return false;
    }
}

#endif
