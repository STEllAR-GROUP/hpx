//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_OCT_19_2008_1234PM)
#define HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_OCT_19_2008_1234PM

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
    class HPX_COMPONENT_EXPORT functional_component
      : public simple_component_base<functional_component>
    {
    public:
        functional_component(applier::applier& appl)
          : simple_component_base<functional_component>(appl)
        {}

        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this 
        // component
        typedef amr::server::functional_component wrapping_type;

        // The eval and is_last_timestep functions have to be overloaded by any
        // functional component derived from this class
        virtual threads::thread_state eval(threads::thread_self&, 
            applier::applier&, bool*, naming::id_type const&, 
            std::vector<naming::id_type> const&)
        {
            // This shouldn't ever be called. If you're seeing this assertion 
            // you probably forgot to overload this function in your stencil 
            // class.
            BOOST_ASSERT(false);
            return threads::terminated;
        }

        virtual threads::thread_state init(threads::thread_self&, 
            applier::applier&, naming::id_type*)
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
            functional_component_init = 0,
            functional_component_eval = 1
        };

        /// This is the main entry point of this component. Calling this 
        /// function (by applying the eval_action) will compute the next 
        /// time step value based on the result values of the previous time 
        /// steps.
        threads::thread_state eval_nv(threads::thread_self& self, 
            applier::applier& appl, bool* retval, naming::id_type const& result, 
            std::vector<naming::id_type> const& gids)
        {
            return eval(self, appl, retval, result, gids);
        }

        threads::thread_state init_nv(threads::thread_self& self, 
            applier::applier& appl, naming::id_type* result)
        {
            return init(self, appl, result);
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action0<
            functional_component, naming::id_type, functional_component_init, 
            &functional_component::init_nv
        > init_action;

        typedef hpx::actions::result_action2<
            functional_component, bool, functional_component_eval, 
            naming::id_type const&, std::vector<naming::id_type> const&, 
            &functional_component::eval_nv
        > eval_action;
    };

}}}}

#endif
