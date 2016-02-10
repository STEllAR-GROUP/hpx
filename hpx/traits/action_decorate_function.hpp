//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_DECORATE_FUNCTION_MAR_30_2014_1054AM)
#define HPX_TRAITS_ACTION_DECORATE_FUNCTION_MAR_30_2014_1054AM

#include <hpx/traits.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>

#include <utility>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    namespace detail
    {
        struct decorate_function_helper
        {
            // by default we return the unchanged function
            template <typename Action, typename F>
            static threads::thread_function_type
            call(wrap_int, naming::address_type lva, F && f)
            {
                return std::forward<F>(f);
            }

            // forward the call if the component implements the function
            template <typename Action, typename F>
            static auto
            call(int, naming::address_type lva, F && f)
            ->  decltype(
                    Action::component_type::decorate_action(
                        lva, std::forward<F>(f))
                )
            {
                typedef typename Action::component_type component_type;
                return component_type::decorate_action(lva, std::forward<F>(f));
            }
        };

        template <typename Action, typename F>
        threads::thread_function_type
        call_decorate_function(naming::address_type lva, F && f)
        {
            return decorate_function_helper::template call<Action>(
                0, lva, std::forward<F>(f));
        }
    }

    template <typename Action, typename Enable>
    struct action_decorate_function
    {
        template <typename F>
        static threads::thread_function_type
        call(naming::address_type lva, F && f)
        {
            return detail::call_decorate_function<Action>(
                lva, std::forward<F>(f));
        }
    };
}}

#endif

