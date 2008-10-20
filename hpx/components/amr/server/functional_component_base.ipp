//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_BASE_IMPL_OCT_19_2008_1234PM)
#define HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_BASE_IMPL_OCT_19_2008_1234PM

#include <hpx/components/amr/server/functional_component_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    threads::thread_state 
    functional_component_base<T, 1>::is_last_timestep(
        threads::thread_self&, applier::applier&, bool* result)
    {
        // FIXME: throw exception
        return threads::terminated;
    }

    template <typename T>
    threads::thread_state 
    functional_component_base<T, 1>::eval(threads::thread_self&, 
        applier::applier&, T*, T const&)
    {
        // FIXME: throw exception
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    threads::thread_state 
    functional_component_base<T, 3>::is_last_timestep(
        threads::thread_self&, applier::applier&, bool* result)
    {
        // FIXME: throw exception
        return threads::terminated;
    }

    template <typename T>
    threads::thread_state 
    functional_component_base<T, 3>::eval(threads::thread_self&, 
        applier::applier&, T*, T const&, T const&, T const&)
    {
        // FIXME: throw exception
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    threads::thread_state 
    functional_component_base<T, 5>::is_last_timestep(
        threads::thread_self&, applier::applier&, bool* result)
    {
        // FIXME: throw exception
        return threads::terminated;
    }

    template <typename T>
    threads::thread_state 
    functional_component_base<T, 5>::eval(threads::thread_self&, 
        applier::applier&, T*, T const&, T const&, T const&, T const&, T const&)
    {
        // FIXME: throw exception
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    threads::thread_state 
    functional_component_base<T, 1>::is_last_timestep_nonvirt(
        threads::thread_self& self, applier::applier& appl, bool* result)
    {
        return is_last_timestep(self, appl, result);
    }

    template <typename T>
    threads::thread_state 
    functional_component_base<T, 1>::eval_nonvirt(threads::thread_self& self, 
        applier::applier& appl, T* result, T const& val1)
    {
        return eval(self, appl, result, val1);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    threads::thread_state 
    functional_component_base<T, 3>::is_last_timestep_nonvirt(
        threads::thread_self& self, applier::applier& appl, bool* result)
    {
        return is_last_timestep(self, appl, result);
    }

    template <typename T>
    threads::thread_state 
    functional_component_base<T, 3>::eval_nonvirt(threads::thread_self& self, 
        applier::applier& appl, T* result, T const& val1, T const& val2, 
        T const& val3)
    {
        return eval(self, appl, result, val1, val2, val3);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    threads::thread_state 
    functional_component_base<T, 5>::is_last_timestep_nonvirt(
        threads::thread_self& self, applier::applier& appl, bool* result)
    {
        return is_last_timestep(self, appl, result);
    }

    template <typename T>
    threads::thread_state 
    functional_component_base<T, 5>::eval_nonvirt(threads::thread_self& self, 
        applier::applier& appl, T* result, T const& val1, T const& val2, 
        T const& val3, T const& val4, T const& val5)
    {
        return eval(self, appl, result, val1, val2, val3, val4, val5);
    }

    ///////////////////////////////////////////////////////////////////////////

}}}}

#endif


