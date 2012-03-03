//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>

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

    template <typename Graph, typename BFSVisitor, typename ColorMap>
    struct do_bfs
    {
        typedef typename boost::graph_traits<Graph>::edge_descriptor
            edge_type;

        do_bfs(Graph const& graph, edge_type e, BFSVisitor vis, ColorMap& cm,
                hpx::lcos::local_counting_semaphore& sem)
          : graph_(graph), e_(e), vis_(vis), cm_(cm), sem_(sem)
        {}

        void call()
        {
            typedef typename boost::graph_traits<Graph>::vertex_descriptor
                vertex_type;

            concurrent_bgl::queue<vertex_type> q;
            vertex_type v = target(e_, graph_);
            vis_.examine_edge(e_, graph_);

            typedef typename boost::property_traits<ColorMap>::value_type
                color_value_type;
            typedef boost::color_traits<color_value_type> color_type;

            vis_.tree_edge(e_, graph_);
            concurrent_bgl::breadth_first_visit(graph_, v, q, vis_, cm_);
            vis_.finish_vertex(v, graph_);

            sem_.signal();
        }

        Graph const& graph_;
        edge_type e_;
        BFSVisitor vis_;
        ColorMap& cm_;
        hpx::lcos::local_counting_semaphore& sem_;
    };

    ///////////////////////////////////////////////////////////////////////////
    double concurrent_bgl_graph::bfs(std::size_t root)
    {
        hpx::util::high_resolution_timer t;

        typedef boost::graph_traits<graph_type> graph_traits;
        typedef concurrent_bgl::color_map<boost::identity_property_map> colormap_type;

        boost::identity_property_map pmap;
        colormap_type cm (num_vertices(graph_), pmap);

        typedef boost::property_traits<colormap_type>::value_type color_value_type;
        typedef boost::color_traits<color_value_type> color_type;

        graph_traits::vertex_descriptor v = vertex(root, graph_);
        put(cm, v, color_type::gray(), boost::memory_order_release);

        std::size_t edge_count = 0;
        hpx::lcos::local_counting_semaphore sem;

        graph_traits::out_edge_iterator ei, ei_end;
        for (boost::tie(ei, ei_end) = out_edges(v, graph_); ei != ei_end; ++ei)
        {
            if (!cas(cm, target(*ei, graph_), color_type::white(), color_type::gray()))
                continue;
            ++edge_count;
            break;
        }

        for (boost::tie(ei, ei_end) = out_edges(v, graph_); ei != ei_end; ++ei)
        {
            typedef do_bfs<graph_type, bfs_visitor, colormap_type> do_bfs_type;
            do_bfs_type bfs(graph_, *ei, bfs_visitor(parents_), cm, sem);
            hpx::applier::register_thread_nullary(
                HPX_STD_BIND(&do_bfs_type::call, &bfs), "do_bfs");
            break;
        }

        put(cm, v, color_type::black(), boost::memory_order_release);
        parents_[root] = root;

        sem.wait(edge_count);           // wait for all threads to finish
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

