//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STUBS_STENCIL_VALUE_NOV_02_2008_0447PM)
#define HPX_COMPONENTS_AMR_STUBS_STENCIL_VALUE_NOV_02_2008_0447PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/components/amr/server/stencil_value.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace stubs 
{
    /// \class stencil_value stencil_value.hpp hpx/components/amr/stubs/stencil_value.hpp
    template <int N>
    class stencil_value 
      : public components::stubs::stub_base<amr::server::stencil_value<N> >
    {
    private:
        typedef 
            components::stubs::stub_base<amr::server::stencil_value<N> > 
        base_type;

    public:
        /// Construct a new stencil_value instance
        stencil_value(applier::applier& appl)
          : base_type(appl)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        ///////////////////////////////////////////////////////////////////////
        /// Invokes the time series evolution for this data point using the
        /// data referred to by the parameter \a initial. After finishing 
        /// execution it returns a reference to the result as its return value
        /// (parameter \a result)
        static lcos::future_value<naming::id_type> call_async(
            applier::applier& appl, naming::id_type const& targetgid, 
            naming::id_type const& initial)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized future_value, the caller needs
            // to call get() on the return value to obtain the result
            typedef typename amr::server::stencil_value<N>::call_action action_type;
            return lcos::eager_future<action_type>(appl, targetgid, initial);
        }

        static naming::id_type call(threads::thread_self& self, 
            applier::applier& appl, naming::id_type const& targetgid, 
            naming::id_type const& initial)
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the eager_future
            return call_async(appl, targetgid, initial).get(self);
        }

        lcos::future_value<naming::id_type> call_async(
            naming::id_type const& targetgid, naming::id_type const& initial)
        {
            return call_async(this->appl_, targetgid, initial);
        }

        naming::id_type call(threads::thread_self& self, 
            naming::id_type const& targetgid, naming::id_type const& initial)
        {
            return call(self, this->appl_, targetgid, initial);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Return the gid's of the output ports associated with this 
        /// \a stencil_value instance.
        static lcos::future_value<std::vector<naming::id_type> > 
        get_output_ports_async(applier::applier& appl, naming::id_type const& gid)
        {
            // Create an eager_future, execute the required action,
            // we simply return the initialized future_value, the caller needs
            // to call get() on the return value to obtain the result
            typedef 
                typename amr::server::stencil_value<N>::get_output_ports_action 
            action_type;
            typedef std::vector<naming::id_type> return_type;
            return lcos::eager_future<action_type, return_type>(appl, gid);
        }

        static std::vector<naming::id_type> 
        get_output_ports(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& gid)
        {
            // The following get yields control while the action above 
            // is executed and the result is returned to the eager_future
            return get_output_ports_async(appl, gid).get(self);
        }

        lcos::future_value<std::vector<naming::id_type> > 
        get_output_ports_async(naming::id_type const& gid)
        {
            return get_output_ports_async(this->appl_, gid);
        }

        std::vector<naming::id_type> 
        get_output_ports(threads::thread_self& self, naming::id_type const& gid)
        {
            return get_output_ports(self, this->appl_, gid);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Connect the destinations given by the provided gid's with the 
        /// corresponding input ports associated with this \a stencil_value 
        /// instance.
        static void connect_input_ports(applier::applier& appl, 
            naming::id_type const& gid, std::vector<naming::id_type> const& gids)
        {
            typedef 
                typename amr::server::stencil_value<N>::connect_input_ports_action 
            action_type;
            appl.apply<action_type>(gid, gids);
        }

        void connect_input_ports(naming::id_type const& gid, 
            std::vector<naming::id_type> const& gids)
        {
            connect_input_ports(this->appl_, gid, gids);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Set the gid of the component implementing the actual time evolution
        /// functionality
        static void set_functional_component(applier::applier& appl, 
            naming::id_type const& gid, naming::id_type const& functiongid)
        {
            typedef 
                typename amr::server::stencil_value<N>::set_functional_component_action 
            action_type;
            appl.apply<action_type>(gid, functiongid);
        }

        void set_functional_component(naming::id_type const& gid, 
            naming::id_type const& functiongid)
        {
            set_functional_component(this->appl_, gid, functiongid);
        }
    };

}}}}

#endif

