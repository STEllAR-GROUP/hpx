//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/util.hpp>

#include <vector>
#include <queue>

#include <boost/foreach.hpp>

#include "bgl_graph.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace bfs { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    void bgl_graph::init(std::size_t idx, std::size_t grainsize,
        std::vector<std::pair<std::size_t, std::size_t> > const& edgelist)
    {
        idx_ = idx;

        parents_.resize(grainsize);
        std::fill(parents_.begin(), parents_.end(), 0);

        graph_ = graph_type(grainsize);

        typedef std::pair<std::size_t, std::size_t> edge_type;
        BOOST_FOREACH(edge_type const& e, edgelist)
            add_edge(e.first, e.second, graph_);
    }

    ///////////////////////////////////////////////////////////////////////////
    struct bfs_visitor : boost::default_bfs_visitor
    {
        bfs_visitor(std::vector<std::size_t>& parents)
          : parents_(parents)
        {}

        template <typename Edge, typename Graph>
        void tree_edge(Edge e, Graph& g) const
        {
            parents_[target(e, g)] = source(e, g);
        }

        std::vector<std::size_t>& parents_;
    };

    double bgl_graph::bfs(std::size_t root)
    {
        hpx::util::high_resolution_timer t;

        bfs_visitor vis(parents_);
        breadth_first_search(graph_, vertex(root, graph_), visitor(vis));
        parents_[root] = root;

        return t.elapsed();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<std::size_t> bgl_graph::get_parents()
    {
        return parents_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void bgl_graph::reset()
    {
        std::fill(parents_.begin(), parents_.end(), 0);
    }
}}

