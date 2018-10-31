//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_DECORATE_FUNCTION_MAR_30_2014_1054AM)
#define HPX_TRAITS_ACTION_DECORATE_FUNCTION_MAR_30_2014_1054AM

#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/traits/has_xxx.hpp>
#include <hpx/util/unique_function.hpp>

#include <type_traits>
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
            template <typename Component, typename F>
            static threads::thread_function_type
            call(wrap_int, naming::address_type /*lva*/, F && f)
            {
                return std::forward<F>(f);
            }

            // forward the call if the component implements the function
            template <typename Component, typename F>
            static auto
            call(int, naming::address_type lva, F && f)
            ->  decltype(Component::decorate_action(lva, std::forward<F>(f)))
            {
                return Component::decorate_action(lva, std::forward<F>(f));
            }
        };

        template <typename Component, typename F>
        threads::thread_function_type
        call_decorate_function(naming::address_type lva, F && f)
        {
            return decorate_function_helper::template call<Component>(
                0, lva, std::forward<F>(f));
        }

        HPX_HAS_XXX_TRAIT_DEF(decorates_action);
    }

    template <typename Action, typename Enable = void>
    struct has_decorates_action
      : detail::has_decorates_action<
            typename std::decay<Action>::type::component_type>
    {};

    template <typename Action, typename Enable = void>
    struct action_decorate_function
    {
        static constexpr bool value = has_decorates_action<Action>::value;

        template <typename F>
        static threads::thread_function_type
        call(naming::address_type lva, F && f)
        {
            typedef typename std::decay<Action>::type::component_type
                component_type;
            return detail::call_decorate_function<component_type>(
                lva, std::forward<F>(f));
        }
    };

    template <typename Action, typename Enable = void>
    struct component_decorates_action
      : detail::has_decorates_action<typename std::decay<Action>::type>
    {};

    template <typename Component, typename Enable = void>
    struct component_decorate_function
    {
        template <typename F>
        static threads::thread_function_type
        call(naming::address_type lva, F && f)
        {
            return detail::call_decorate_function<Component>(
                lva, std::forward<F>(f));
        }
    };
}}

#endif

