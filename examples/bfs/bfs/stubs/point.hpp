//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_POINT)
#define HPX_COMPONENTS_STUBS_POINT

#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/point.hpp"

namespace hpx { namespace geometry { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct point : components::stub_base<server::point>
    {
        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        /// Initialize the \a hpx::geometry::server::point instance with the
        /// given graph file.  
        static lcos::promise<void>
        init_async(naming::id_type const& gid,std::size_t objectid,
            std::size_t max_num_neighbors,std::string graphfile)
        {
            typedef server::point::init_action action_type;
            return lcos::eager_future<action_type>(gid,objectid,max_num_neighbors,
                graphfile);
        }

        /// Initialize the \a hpx::geometry::server::point instance with the
        /// given graph file.  
        static void init(naming::id_type const& gid,std::size_t objectid,
            std::size_t max_num_neighbors,std::string const& graphfile)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            init_async(gid,objectid,max_num_neighbors,graphfile).get();
        }

        /// Traverse the graph. 
        static lcos::promise<std::vector<std::size_t> >
        traverse_async(naming::id_type const& gid,std::size_t level,std::size_t parent)
        {
            typedef server::point::traverse_action action_type;
            return lcos::eager_future<action_type>(gid,level,parent);
        }

        /// Traverse the graph. 
        static std::vector<std::size_t>
        traverse(naming::id_type const& gid,std::size_t level,std::size_t parent)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the promise
            return traverse_async(gid,level,parent).get();
        }

    };

}}}

#endif
