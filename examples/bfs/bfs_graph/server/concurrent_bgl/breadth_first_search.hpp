//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// The breadth_first_visit and breadth_first_search functions below have
// been taken from the Boost Graph Library. We have modified those
// functions to make them usable in multi-threaded environments.
//
// Original copyrights:
// Copyright 1997, 1998, 1999, 2000 University of Notre Dame.
// Authors: Andrew Lumsdaine, Lie-Quan Lee, Jeremy G. Siek

#if !defined(HPX_EXAMPLE_BFS_CONCURRENT_BGL_BFS_JAN_02_2012_0734PM)
#define HPX_EXAMPLE_BFS_CONCURRENT_BGL_BFS_JAN_02_2012_0734PM

#include <boost/concept/assert.hpp>
#include <boost/graph/breadth_first_search.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace concurrent_bgl
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename IncidenceGraph, typename Buffer, typename BFSVisitor,
        typename ColorMap>
    void breadth_first_visit(IncidenceGraph const& g,
        typename boost::graph_traits<IncidenceGraph>::vertex_descriptor s,
        Buffer& Q, BFSVisitor vis, ColorMap& color)
    {
        BOOST_CONCEPT_ASSERT((boost::IncidenceGraphConcept<IncidenceGraph>));
        BOOST_CONCEPT_ASSERT((boost::BFSVisitorConcept<BFSVisitor, IncidenceGraph>));

        typedef boost::graph_traits<IncidenceGraph> GTraits;
        typedef typename GTraits::vertex_descriptor Vertex;
        typedef typename GTraits::edge_descriptor Edge;

        BOOST_CONCEPT_ASSERT((boost::ReadWritePropertyMapConcept<ColorMap, Vertex>));

        typedef typename boost::property_traits<ColorMap>::value_type ColorValue;
        typedef boost::color_traits<ColorValue> Color;
        typename GTraits::out_edge_iterator ei, ei_end;

        put(color, s, Color::gray(), boost::memory_order_release);
                                                      vis.discover_vertex(s, g);
        Q.push(s);

        Vertex u;
        while (Q.pop(u)) {                            vis.examine_vertex(u, g);
            for (boost::tie(ei, ei_end) = out_edges(u, g); ei != ei_end; ++ei) {
                Vertex v = target(*ei, g);            vis.examine_edge(*ei, g);
                if (cas(color, v, Color::white(), Color::gray())) {
                                                      vis.tree_edge(*ei, g);
                                                      vis.discover_vertex(v, g);
                    Q.push(v);
                }
                else {                                vis.non_tree_edge(*ei, g);
                    if (get(color, v, boost::memory_order_acquire) == Color::gray())
                                                      vis.gray_target(*ei, g);
                    else
                                                      vis.black_target(*ei, g);
                }
            }
            put(color, u, Color::black(), boost::memory_order_release);
                                                      vis.finish_vertex(u, g);
        }
    }

    template <typename VertexListGraph, typename Buffer, typename BFSVisitor,
        typename ColorMap>
    void breadth_first_search(VertexListGraph const& g,
        typename boost::graph_traits<VertexListGraph>::vertex_descriptor s,
        Buffer& Q, BFSVisitor vis, ColorMap& color)
    {
        // Initialization
        typedef typename boost::property_traits<ColorMap>::value_type ColorValue;
        typedef boost::color_traits<ColorValue> Color;
        typename boost::graph_traits<VertexListGraph>::vertex_iterator i, i_end;

        for (boost::tie(i, i_end) = vertices(g); i != i_end; ++i) {
            vis.initialize_vertex(*i, g);
            put(color, *i, Color::white());
        }
        concurrent_bgl::breadth_first_visit(g, s, Q, vis, color);
    }
}

#endif

