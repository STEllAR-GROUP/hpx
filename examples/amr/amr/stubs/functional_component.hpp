//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STUBS_FUNCTIONAL_COMPONENT_NOV_05_2008_0338PM)
#define HPX_COMPONENTS_AMR_STUBS_FUNCTIONAL_COMPONENT_NOV_05_2008_0338PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/functional_component.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace stubs 
{
    ///////////////////////////////////////////////////////////////////////////
    struct functional_component
      : components::stubs::stub_base<amr::server::functional_component>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        static lcos::future_value<bool> eval_async(naming::id_type const& gid, 
            naming::id_type const& result, 
            std::vector<naming::id_type> const& gids, int row, int column)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized future_value, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::functional_component::eval_action action_type;
            return lcos::eager_future<action_type>(gid, result, gids, row, column);
        }

        static bool eval(naming::id_type const& gid, 
            naming::id_type const& result, std::vector<naming::id_type> const& gids,
            int row, int column)
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the eager_future
            return eval_async(gid, result, gids, row, column).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future_value<naming::id_type> alloc_data_async(
            naming::id_type const& gid, int item = -1, int maxitems = -1,
            int row = -1)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized future_value, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::functional_component::alloc_data_action action_type;
            return lcos::eager_future<action_type>(gid, item, maxitems, row);
        }

        static naming::id_type alloc_data(naming::id_type const& gid, 
            int item = -1, int maxitems = -1, int row = -1)
        {
            return alloc_data_async(gid, item, maxitems, row).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static void init(naming::id_type const& gid, std::size_t numsteps, 
            naming::id_type const& val)
        {
            typedef amr::server::functional_component::init_action action_type;
            applier::apply<action_type>(gid, numsteps, val);
        }

        static void init_sync(naming::id_type const& gid, 
            std::size_t numsteps, naming::id_type const& val)
        {
            typedef amr::server::functional_component::init_action action_type;
            lcos::eager_future<action_type>(gid, numsteps, val).get();
        }
    };

}}}}

#endif
