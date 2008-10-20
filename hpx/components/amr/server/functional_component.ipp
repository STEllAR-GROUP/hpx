//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_IMPL_OCT_19_2008_1234PM)
#define HPX_COMPONENTS_AMR_FUNCTIONAL_COMPONENT_IMPL_OCT_19_2008_1234PM

#include <hpx/components/amr/server/functional_component.hpp>
#include <hpx/components/amr/server/functional_component_base.ipp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename T>
    threads::thread_state 
    functional_component<Derived, T, 1>::is_last_timestep(
        threads::thread_self&, applier::applier&, bool* result)
    {
        *result = derived().is_last_timestep();
        return threads::terminated;
    }

    template <typename Derived, typename T>
    threads::thread_state 
    functional_component<Derived, T, 1>::eval(threads::thread_self&, 
        applier::applier&, T* result, T const& val1)
    {
        *result = derived().eval(val1);
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename T>
    threads::thread_state 
    functional_component<Derived, T, 3>::is_last_timestep(
        threads::thread_self&, applier::applier&, bool* result)
    {
        *result = derived().is_last_timestep();
        return threads::terminated;
    }

    template <typename Derived, typename T>
    threads::thread_state 
    functional_component<Derived, T, 3>::eval(threads::thread_self&, 
        applier::applier&, T* result, T const& val1, T const& val2, 
        T const& val3)
    {
        *result = derived().eval(val1, val2, val3);
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename T>
    threads::thread_state 
    functional_component<Derived, T, 5>::is_last_timestep(
        threads::thread_self&, applier::applier&, bool* result)
    {
        *result = derived().is_last_timestep();
        return threads::terminated;
    }

    template <typename Derived, typename T>
    threads::thread_state 
    functional_component<Derived, T, 5>::eval(threads::thread_self&, 
        applier::applier&, T* result, T const& val1, T const& val2, 
        T const& val3, T const& val4, T const& val5)
    {
        *result = derived().eval(val1, val2, val3, val4, val5);
        return threads::terminated;
    }

}}}}

#endif


