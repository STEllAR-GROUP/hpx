//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STUBS_RK_LEFT_FEB_17_2010_0448PM)
#define HPX_COMPONENTS_AMR_STUBS_RK_LEFT_FEB_17_2010_0448PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/rk_left.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace stubs 
{
    ///////////////////////////////////////////////////////////////////////////
    struct rk_left
      : components::stubs::stub_base<amr::server::rk_left>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        ///////////////////////////////////////////////////////////////////////
        static lcos::future_value<std::vector<naming::id_type> > 
        init_execute_async(naming::id_type const& gid, 
            components::component_type function_type, 
            components::component_type logging_type,
            std::size_t level,had_double_type x,Parameter const& par)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized future_value, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::rk_left::init_execute_action action_type;
            return lcos::eager_future<action_type>(gid, function_type,
                logging_type,level,x, par);
        }

        static std::vector<naming::id_type> init_execute(naming::id_type const& gid, 
            components::component_type function_type,
            components::component_type logging_type,
            std::size_t level, had_double_type x,Parameter const& par)
        {
            return init_execute_async(gid, function_type, logging_type,level,x,par).get();
        }

        ///////////////////////////////////////////////////////////////////////
        static lcos::future_value<std::vector<naming::id_type> > 
        execute_async(naming::id_type const& gid, 
            std::vector<naming::id_type> const& initial_data,
            components::component_type function_type, 
            components::component_type logging_type,
            Parameter const& par)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized future_value, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::rk_left::execute_action action_type;
            return lcos::eager_future<action_type>(gid, initial_data, 
                function_type, logging_type, par);
        }

        static std::vector<naming::id_type> execute(naming::id_type const& gid, 
            std::vector<naming::id_type> const& initial_data,
            components::component_type function_type, 
            components::component_type logging_type,
            Parameter const& par)
        {
            return execute_async(gid, initial_data, function_type,
                logging_type,par).get();
        }
    };

}}}}

#endif
