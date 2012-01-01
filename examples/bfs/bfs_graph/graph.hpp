//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLE_BFS_GRAPH_DEC_31_2011_0435PM)
#define HPX_EXAMPLE_BFS_GRAPH_DEC_31_2011_0435PM

#include <hpx/runtime/components/client_base.hpp>
#include <boost/assert.hpp>

#include "stubs/graph.hpp"

namespace bfs
{
    ///////////////////////////////////////////////////////////////////////////
    /// The client side representation of a \a bfs::server::graph components.
    class graph : public hpx::components::client_base<graph, stubs::graph>
    {
        typedef hpx::components::client_base<graph, stubs::graph>
            base_type;

    public:
        /// Default construct an empty client side representation (not
        /// connected to any existing component).
        graph()
        {}

        /// Create a client side representation for the existing
        /// \a bfs::server::graph instance with the given GID.
        graph(hpx::naming::id_type const& gid)
          : base_type(gid)
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        // initialize the graph
        hpx::lcos::promise<void> init_async(
            std::size_t idx, std::size_t grainsize,
            std::vector<std::pair<std::size_t, std::size_t> > const& edgelist)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::init_async(gid_, idx, grainsize, edgelist);
        }
        void init(std::size_t idx, std::size_t grainsize,
            std::vector<std::pair<std::size_t, std::size_t> > const& edgelist)
        {
            BOOST_ASSERT(gid_);
            this->base_type::init_async(gid_, idx, grainsize, edgelist);
        }

        /// Perform a BFS on the graph.
        hpx::lcos::promise<double>
        bfs_async(std::size_t root)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::bfs_async(gid_, root);
        }
        double bfs(std::size_t root)
        {
            BOOST_ASSERT(gid_);
            return this->base_type::bfs(gid_, root);
        }

        /// validate the BFS on the graph.
        hpx::lcos::promise<std::vector<std::size_t> >
        get_parents_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_parents_async(gid_);
        }
        std::vector<std::size_t> get_parents()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::get_parents(gid_);
        }

        /// Reset for the next BFS
        hpx::lcos::promise<void>
        reset_async()
        {
            BOOST_ASSERT(gid_);
            return this->base_type::reset_async(gid_);
        }
        void reset()
        {
            BOOST_ASSERT(gid_);
            this->base_type::reset(gid_);
        }
    };
}

#endif

