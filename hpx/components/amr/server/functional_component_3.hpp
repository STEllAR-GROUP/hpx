//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_3_OCT_21_2008_1216PM)
#define HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_3_OCT_21_2008_1216PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/components/amr/server/functional_component.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    template <>
    class functional_component<3>
      : public simple_component_base<functional_component<3> >
    {
    public:
        functional_component(applier::applier& appl)
          : simple_component_base<functional_component>(appl)
        {}

        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this 
        // component
        typedef amr::server::functional_component<3> wrapping_type;

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        virtual threads::thread_state eval(threads::thread_self&, 
            applier::applier&, naming::id_type*, naming::id_type const&, 
            naming::id_type const&, naming::id_type const&)
        {
            // This shouldn't ever be called. If you're seeing this assertion 
            // you probably forgot to overload this function in your stencil 
            // class.
            BOOST_ASSERT(false);
            return threads::terminated;
        }

        virtual threads::thread_state is_last_timestep(threads::thread_self&, 
            applier::applier&, bool*) 
        {
            // This shouldn't ever be called. If you're seeing this assertion 
            // you probably forgot to overload this function in your stencil 
            // class.
            BOOST_ASSERT(false);
            return threads::terminated;
        }

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            functional_component_eval = 0,
            functional_component_is_last_timestep = 1
        };

        /// This is the main entry point of this component. Calling this 
        /// function (by applying the eval_action) will compute the next 
        /// time step value based on the result values of the previous time 
        /// steps.
        threads::thread_state eval_nv(threads::thread_self& self, 
            applier::applier& appl, naming::id_type* result, 
            naming::id_type const& gid1, naming::id_type const& gid2, 
            naming::id_type const& gid3)
        {
            return eval(self, appl, result, gid1, gid2, gid3);
        }

        threads::thread_state is_last_timestep_nv(threads::thread_self& self, 
            applier::applier& appl, bool* result)
        {
            return is_last_timestep(self, appl, result);
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action0<
            functional_component, bool, functional_component_is_last_timestep, 
            &functional_component::is_last_timestep_nv
        > is_last_timestep_action;

        typedef hpx::actions::result_action3<
            functional_component, naming::id_type, functional_component_eval, 
            naming::id_type const&, naming::id_type const&, 
            naming::id_type const&, &functional_component::eval_nv
        > eval_action;
    };

}}}}

#endif
