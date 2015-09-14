//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_DECORATE_FUNCTION_MAR_30_2014_1054AM)
#define HPX_TRAITS_ACTION_DECORATE_FUNCTION_MAR_30_2014_1054AM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/always_void.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    template <typename Action, typename Enable>
    struct action_decorate_function
    {
        template <typename F>
        static threads::thread_function_type
        call(naming::address_type lva, F && f)
        {
            // by default we forward this to the component type
            typedef typename Action::component_type component_type;
            return component_type::decorate_action(lva, std::forward<F>(f));
        }
    };

    template <typename Action>
    struct action_decorate_function<Action
      , typename util::always_void<typename Action::type>::type>
      : action_decorate_function<typename Action::type>
    {};
}}

#endif

