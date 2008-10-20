//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_FUNCTIONAL_TIMESTEP_OCT_19_2008_0931PM)
#define HPX_COMPONENTS_AMR_FUNCTIONAL_TIMESTEP_OCT_19_2008_0931PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    /// This component is responsible for the creation an the deletion of a
    /// appropriate number of stencil_value components, connecting them, and 
    /// generally controlling the flow of execution.
    class functional_timestep
      : public simple_component_base<functional_timestep>
    {
    public:
        /// Construct a new stencil_value instance
        functional_timestep(applier::applier& appl);

        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            functional_timestep_initialize = 0,
            functional_timestep_execute = 1,
        };

        /// This is the main entry point of this component. Calling this 
        /// function (by applying the call_action) will trigger the repeated 
        /// execution of the whole time step evolution functionality.
        threads::thread_state 
        initialize(threads::thread_self&, applier::applier&);

        /// This is called to execute a full time step based evolution based on
        /// a network of \a stencil_value components set up during a previous 
        /// call to initialize.
        threads::thread_state 
        execute(threads::thread_self&, applier::applier&);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action0<
            functional_timestep, functional_timestep_initialize, 
            &functional_timestep::initialize
        > initialize_action;

        typedef hpx::actions::action0<
            functional_timestep, functional_timestep_execute, 
            &functional_timestep::execute
        > execute_action;
    };

}}}}

#endif

