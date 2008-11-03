//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_SERVER_STENCIL_VALUE_OCT_17_2008_0848AM)
#define HPX_COMPONENTS_AMR_SERVER_STENCIL_VALUE_OCT_17_2008_0848AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/action.hpp>
#include <hpx/lcos/counting_semaphore.hpp>

#include <hpx/components/amr/server/stencil_value_in_adaptor.hpp>
#include <hpx/components/amr/server/stencil_value_out_adaptor.hpp>

#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/bind.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    /// \class stencil_value stencil_value.hpp hpx/components/amr/server/stencil_value.hpp
    template <int N>
    class HPX_COMPONENT_EXPORT stencil_value 
      : public components::detail::managed_component_base<stencil_value<N> >
    {
    protected:
        // the in_adaptors_type is the concrete stencil_value_in_adaptor
        // of the proper type
        typedef amr::server::stencil_value_in_adaptor in_adaptor_type;

        // the out_adaptors_type is the concrete stencil_value_out_adaptor
        // of the proper type
        typedef 
            managed_component<amr::server::stencil_value_out_adaptor>
        out_adaptor_type;

    public:
        /// Construct a new stencil_value instance
        stencil_value(applier::applier& appl);

        /// The function get will be called by the out-ports whenever 
        /// the current value has been requested.
        void get_value(threads::thread_self& self, naming::id_type*);

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            stencil_value_call = 0,
            stencil_value_get_output_ports = 1,
            stencil_value_connect_input_ports = 2,
            stencil_value_set_functional_component = 3,
        };

        /// Main thread function looping through all timesteps
        threads::thread_state  
        main(threads::thread_self& self, applier::applier& appl);

        /// This is the main entry point of this component. Calling this 
        /// function (by applying the call_action) will trigger the repeated 
        /// execution of the whole time step evolution functionality.
        ///
        /// It invokes the time series evolution for this data point using the
        /// data referred to by the parameter \a initial. After finishing 
        /// execution it returns a reference to the result as its return value
        /// (parameter \a result)
        threads::thread_state 
        call (threads::thread_self&, applier::applier&, naming::id_type* result, 
            naming::id_type const& initial);

        /// Return the gid's of the output ports associated with this 
        /// \a stencil_value instance.
        threads::thread_state 
        get_output_ports(threads::thread_self&, applier::applier&, 
            std::vector<naming::id_type> *gids);

        /// Connect the destinations given by the provided gid's with the 
        /// corresponding input ports associated with this \a stencil_value 
        /// instance.
        threads::thread_state 
        connect_input_ports(threads::thread_self&, applier::applier&, 
            std::vector<naming::id_type> const& gids);

        /// Set the gid of the component implementing the actual time evolution
        /// functionality
        threads::thread_state 
        set_functional_component(threads::thread_self&, applier::applier&, 
            naming::id_type const& gid);

        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            stencil_value, naming::id_type, stencil_value_call, 
            naming::id_type const&, &stencil_value::call
        > call_action;

        typedef hpx::actions::result_action0<
            stencil_value, std::vector<naming::id_type>, 
            stencil_value_get_output_ports, &stencil_value::get_output_ports
        > get_output_ports_action;

        typedef hpx::actions::action1<
            stencil_value, stencil_value_connect_input_ports, 
            std::vector<naming::id_type> const&,
            &stencil_value::connect_input_ports
        > connect_input_ports_action;

        typedef hpx::actions::action1<
            stencil_value, stencil_value_set_functional_component, 
            naming::id_type const&, &stencil_value::set_functional_component
        > set_functional_component_action;

    private:
        lcos::counting_semaphore sem_in_;
        lcos::counting_semaphore sem_out_;
        lcos::counting_semaphore sem_result_;

        boost::scoped_ptr<in_adaptor_type> in_[N];    // adaptors used to gather input
        boost::scoped_ptr<out_adaptor_type> out_[N];  // adaptors used to provide result

        naming::id_type value_gid_;                   // reference to current value
        naming::id_type backup_value_gid_;            // reference to previous value
        naming::id_type functional_gid_;              // reference to functional code
    };

}}}}

#endif

