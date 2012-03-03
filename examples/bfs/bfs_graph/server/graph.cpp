//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/util.hpp>

#include <vector>
#include <queue>

#include "graph.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace bfs { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    void graph::init(std::size_t idx, std::size_t grainsize,
        std::vector<std::pair<std::size_t, std::size_t> > const& edgelist)
    {
        idx_ = idx;

        parents_.resize(grainsize);
        std::fill(parents_.begin(), parents_.end(), 0);

        neighbors_.resize(grainsize);

        for (std::size_t i = 0; i < edgelist.size(); ++i)
        {
            std::size_t node = edgelist[i].first;
            std::size_t neighbor = edgelist[i].second;

            if (node != neighbor) {
                neighbors_[node].push_back(neighbor);
                neighbors_[neighbor].push_back(node);
            }
        }
    }

    double graph::bfs(std::size_t root)
    {
        hpx::util::high_resolution_timer t;

        std::queue<std::size_t> q;

        parents_[root] = root;
        q.push(root);

        while (!q.empty()) {
            std::size_t node = q.front(); q.pop();

            std::vector<std::size_t> const& node_neighbors = neighbors_[node];
            std::vector<std::size_t>::const_iterator end = node_neighbors.end();
            for (std::vector<std::size_t>::const_iterator it = node_neighbors.begin();
                 it != end; ++it)
            {
                std::size_t neighbor = *it;
                std::size_t& node_parent = parents_[neighbor];
                if (std::size_t(0) == node_parent) {
                    node_parent = node;
                    q.push(neighbor);
                }
            }
        }

        return t.elapsed();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<std::size_t> graph::get_parents()
    {
        return parents_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void graph::reset()
    {
        std::fill(parents_.begin(), parents_.end(), 0);
    }
}}

