//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_BASE_OCT_19_2008_1234PM)
#define HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_BASE_OCT_19_2008_1234PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, int N>
    class functional_component_base;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class functional_component_base<T, 1>
      : public simple_component_base<functional_component_base<T, 1> >
    {
    public:
        functional_component_base(applier::applier& appl)
          : simple_component_base<functional_component_base<T, 1> >(appl)
        {}

        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            functional_component_eval = 0,
            functional_component_is_last_timestep = 1
        };

        /// The is_last_timestep action decides whether the current time step 
        /// is the last one (the computation has reached the final time step).
        virtual threads::thread_state 
        is_last_timestep(threads::thread_self&, applier::applier&, bool*);

        virtual threads::thread_state 
        eval(threads::thread_self&, applier::applier&, T*, T const&);

        ///////////////////////////////////////////////////////////////////////
        threads::thread_state 
        is_last_timestep_nonvirt(threads::thread_self&, applier::applier&, bool*);

        threads::thread_state 
        eval_nonvirt(threads::thread_self&, applier::applier&, T*, T const&);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action0<
            functional_component_base, bool, 
            functional_component_is_last_timestep, 
            &functional_component_base::is_last_timestep_nonvirt
        > is_last_timestep_action;

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action1<
            functional_component_base, T, functional_component_eval, 
            T const&, &functional_component_base::eval_nonvirt
        > eval_action;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class functional_component_base<T, 3>
      : public simple_component_base<functional_component_base<T, 3> >
    {
    public:
        functional_component_base(applier::applier& appl)
          : simple_component_base<functional_component_base<T, 3> >(appl)
        {}

        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            functional_component_eval = 0,
            functional_component_is_last_timestep = 1
        };

        /// The is_last_timestep action decides whether the current time step 
        /// is the last one (the computation has reached the final time step).
        virtual threads::thread_state 
        is_last_timestep(threads::thread_self&, applier::applier&, bool*);

        virtual threads::thread_state 
        eval(threads::thread_self&, applier::applier&, T*, T const&, T const&, 
            T const&);

        ///////////////////////////////////////////////////////////////////////
        threads::thread_state 
        is_last_timestep_nonvirt(threads::thread_self&, applier::applier&, bool*);

        threads::thread_state 
        eval_nonvirt(threads::thread_self&, applier::applier&, T*, T const&, T const&, 
            T const&);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action0<
            functional_component_base, bool, 
            functional_component_is_last_timestep, 
            &functional_component_base::is_last_timestep_nonvirt
        > is_last_timestep_action;

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action3<
            functional_component_base, T, functional_component_eval, 
            T const&, T const&, T const&, 
            &functional_component_base::eval_nonvirt
        > eval_action;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class functional_component_base<T, 5>
      : public simple_component_base<functional_component_base<T, 5> >
    {
    public:
        functional_component_base(applier::applier& appl)
          : simple_component_base<functional_component_base<T, 5> >(appl)
        {}

        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            functional_component_eval = 0,
            functional_component_is_last_timestep = 1
        };

        /// The is_last_timestep action decides whether the current time step 
        /// is the last one (the computation has reached the final time step).
        virtual threads::thread_state 
        is_last_timestep(threads::thread_self&, applier::applier&, bool*);

        virtual threads::thread_state 
        eval(threads::thread_self&, applier::applier&, T*, T const&, T const&, 
            T const&, T const&, T const&);

        ///////////////////////////////////////////////////////////////////////
        threads::thread_state 
        is_last_timestep_nonvirt(threads::thread_self&, applier::applier&, bool*);

        threads::thread_state 
        eval_nonvirt(threads::thread_self&, applier::applier&, T*, T const&, T const&, 
            T const&, T const&, T const&);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action0<
            functional_component_base, bool, 
            functional_component_is_last_timestep, 
            &functional_component_base::is_last_timestep_nonvirt
        > is_last_timestep_action;

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action5<
            functional_component_base, T, functional_component_eval, 
            T const&, T const&, T const&, T const&, T const&, 
            &functional_component_base::eval_nonvirt
        > eval_action;
    };

}}}}

#endif
