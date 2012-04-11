//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STUBS_FUNCTIONAL_COMPONENT_NOV_05_2008_0338PM)
#define HPX_COMPONENTS_AMR_STUBS_FUNCTIONAL_COMPONENT_NOV_05_2008_0338PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/functional_component.hpp"
#include "../../parameter.hpp"

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
        static lcos::future<int> eval_async(naming::id_type const& gid,
            naming::id_type const& result,
            std::vector<naming::id_type> const& gids, std::size_t row, std::size_t column,
            double cycle_time,parameter const& par)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::functional_component::eval_action action_type;
            return lcos::async<action_type>(gid, result, gids, row, column,cycle_time,par);
        }

        static int eval(naming::id_type const& gid,
            naming::id_type const& result, std::vector<naming::id_type> const& gids,
            int row, int column,double cycle_time, parameter const& par)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return eval_async(gid, result, gids, row, column,cycle_time,par).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future<naming::id_type, naming::id_type> alloc_data_async(
            naming::id_type const& gid, int item, int maxitems,
            int row,std::vector<naming::id_type> const& interp_src_data,
            double time, parameter const& par)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::functional_component::alloc_data_action action_type;
            return lcos::async<action_type>(gid, item, maxitems,
                                                   row, interp_src_data,time,par);
        }

        static naming::id_type alloc_data(naming::id_type const& gid,
            int item, int maxitems, int row,
            std::vector<naming::id_type> const& interp_src_data,double time,
            parameter const& par)
        {
            return alloc_data_async(gid, item, maxitems, row,
                                    interp_src_data,time, par).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future<void>
        init_async(naming::id_type const& gid, std::size_t numsteps,
            naming::id_type const& val)
        {
            typedef amr::server::functional_component::init_action action_type;
            return lcos::async<action_type>(gid, numsteps, val);
        }

        static void init(naming::id_type const& gid,
            std::size_t numsteps, naming::id_type const& val)
        {
            init_async(gid, numsteps, val).get();
        }
    };

}}}}

#endif
