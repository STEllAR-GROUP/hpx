//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_226CC70A_D748_4ADA_BB55_70F85566B5CC)
#define HPX_226CC70A_D748_4ADA_BB55_70F85566B5CC

#include <vector>
#include <queue>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/unlock_lock.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace ep { namespace server
{

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT point
      : public hpx::components::managed_component_base<point>
    {
    public:
        point()
        {}

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality of this component.

        void bfs(std::size_t scale);

        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.

        /// Action codes.
        enum actions
        {
            point_bfs = 0
        };

        typedef hpx::actions::action1<
            // Component server type.
            point,
            // Action code.
            point_bfs,
            // Arguments of this action.
            std::size_t,
            // Method bound to this action.
            &point::bfs
        > bfs_action;

    private:
        hpx::lcos::local::mutex mtx_;
        double sum_;
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION(
    ep::server::point::bfs_action,
    ep_point_bfs_action);
#endif

