//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/util.hpp>

#include <vector>
#include <queue>

#include <boost/foreach.hpp>

#include "concurrent_bgl/queue.hpp"
#include "concurrent_bgl/colormap.hpp"
#include "concurrent_bgl/breadth_first_search.hpp"
#include "concurrent_bgl_graph.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace bfs { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    void concurrent_bgl_graph::init(std::size_t idx, std::size_t grainsize,
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

    double concurrent_bgl_graph::bfs(std::size_t root)
    {
        hpx::util::high_resolution_timer t;

        typedef boost::graph_traits<graph_type> graph_traits;

        boost::identity_property_map pmap;
        concurrent_bgl::queue<graph_traits::vertex_descriptor> q;
        concurrent_bgl::breadth_first_search(
            graph_, vertex(root, graph_), q, bfs_visitor(parents_),
            concurrent_bgl::make_color_map(num_vertices(graph_), pmap));

        parents_[root] = root;
        return t.elapsed();
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<std::size_t> concurrent_bgl_graph::get_parents()
    {
        return parents_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void concurrent_bgl_graph::reset()
    {
        std::fill(parents_.begin(), parents_.end(), 0);
    }
}}

