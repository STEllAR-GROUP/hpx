//  Copyright (c) 2015-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/basic_action.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace components { namespace server {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        // simple utility action which invoke an arbitrary function
        template <typename F, typename... Ts>
        struct invoke_function
        {
            static typename util::invoke_result<F, Ts...>::type call(
                F f, Ts... ts)
            {
                return f(std::move(ts)...);
            }
        };

        // simple utility action which invoke an arbitrary global function
        template <typename F>
        struct invoke_function_ptr;

        template <typename R, typename... Ts>
        struct invoke_function_ptr<R (*)(Ts...)>
        {
            static R call(std::size_t f, Ts... ts)
            {
                return reinterpret_cast<R (*)(Ts...)>(f)(std::move(ts)...);
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // action definition exposing invoke_function<>
    template <typename F, typename... Ts>
    struct invoke_function_action
      : ::hpx::actions::action<typename util::invoke_result<F, Ts...>::type (*)(
                                   F, Ts...),
            &detail::invoke_function<F, Ts...>::call,
            invoke_function_action<F, Ts...>>
    {
    };

    // action definition exposing invoke_function_ptr<> that binds a global
    // function (Note: this assumes global function addresses are the same on
    // all localities)
    template <typename R, typename... Ts>
    struct invoke_function_action<R (*)(Ts...), Ts...>
      : ::hpx::actions::action<R (*)(std::size_t, Ts...),
            &detail::invoke_function_ptr<R (*)(Ts...)>::call,
            invoke_function_action<R (*)(Ts...), Ts...>>
    {
    };
}}}    // namespace hpx::components::server
