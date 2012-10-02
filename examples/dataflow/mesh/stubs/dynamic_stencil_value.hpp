//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STUBS_STENCIL_VALUE_NOV_02_2008_0447PM)
#define HPX_COMPONENTS_AMR_STUBS_STENCIL_VALUE_NOV_02_2008_0447PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include "../server/dynamic_stencil_value.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace stubs
{
    /// \class dynamic_stencil_value dynamic_stencil_value.hpp hpx/components/amr/stubs/dynamic_stencil_value.hpp
    struct dynamic_stencil_value
      : components::stub_base<amr::server::dynamic_stencil_value>
    {
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        ///////////////////////////////////////////////////////////////////////
        /// Invokes the time series evolution for this data point using the
        /// data referred to by the parameter \a initial. After finishing
        /// execution it returns a reference to the result as its return value
        /// (parameter \a result)
        static lcos::future<naming::id_type, naming::id_type> call_async(
            naming::id_type const& targetgid, naming::id_type const& initial)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::dynamic_stencil_value::call_action action_type;
            return hpx::async<action_type>(targetgid,initial);
        }

        static naming::id_type call(naming::id_type const& targetgid,
            naming::id_type const& initial)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return call_async(targetgid, initial).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Return the gid's of the output ports associated with this
        /// \a dynamic_stencil_value instance.
        static lcos::future<std::vector<naming::id_type> >
        get_output_ports_async(naming::id_type const& gid)
        {
            // Create a future, execute the required action,
            // we simply return the initialized future, the caller needs
            // to call get() on the return value to obtain the result
            typedef amr::server::dynamic_stencil_value::get_output_ports_action
                action_type;
            return hpx::async<action_type>(gid);
        }

        static std::vector<naming::id_type>
        get_output_ports(naming::id_type const& gid)
        {
            // The following get yields control while the action above
            // is executed and the result is returned to the future
            return get_output_ports_async(gid).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Connect the destinations given by the provided gid's with the
        /// corresponding input ports associated with this \a dynamic_stencil_value
        /// instance.
        static lcos::future<void>
        connect_input_ports_async(naming::id_type const& gid,
            std::vector<naming::id_type> const& gids)
        {
            typedef
                amr::server::dynamic_stencil_value::connect_input_ports_action
            action_type;
            return hpx::async<action_type>(gid, gids);
        }

        static void connect_input_ports(naming::id_type const& gid,
            std::vector<naming::id_type> const& gids)
        {
            connect_input_ports_async(gid, gids).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Set the gid of the component implementing the actual time evolution
        /// functionality
        static lcos::future<void>
        set_functional_component_async(naming::id_type const& gid,
            naming::id_type const& functiongid, int row, int column,
            int instencilsize, int outstencilsize, double cycle_time,parameter const& par)
        {
            typedef
                amr::server::dynamic_stencil_value::set_functional_component_action
            action_type;
            return hpx::async<action_type>(gid, functiongid, row,
                column, instencilsize, outstencilsize,cycle_time, par);
        }

        static void set_functional_component(naming::id_type const& gid,
            naming::id_type const& functiongid, int row, int column,
            int instencilsize, int outstencilsize, double cycle_time,parameter const& par)
        {
            set_functional_component_async(gid, functiongid, row,
                column, instencilsize, outstencilsize,cycle_time, par).get();
        }

        ///////////////////////////////////////////////////////////////////////
        /// Subset of set_functional_component functionality
        static lcos::future<void>
        start_async(naming::id_type const& gid)
        {
            typedef amr::server::dynamic_stencil_value::start_action
                action_type;
            return hpx::async<action_type>(gid);
        }

        static void start(naming::id_type const& gid)
        {
            start_async(gid).get();
        }
    };

}}}}

#endif

