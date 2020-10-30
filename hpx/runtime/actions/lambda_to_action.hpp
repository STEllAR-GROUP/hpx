//  Copyright (c) 2017 Antoine TRAN TAN
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/actions/make_action.hpp

#pragma once

#include <hpx/include/plain_actions.hpp>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Helpers to actionize a lambda

        template <typename ... T>
        struct sequence
        {};

        struct addr_add
        {
            template<class T>
            friend typename std::remove_reference<T>::type *
            operator+(addr_add, T &&t)
            {
                return &t;
            }
        };


        template<typename F, typename ReturnType, typename... Args>
        struct lambda_action
          : basic_action<detail::plain_function, ReturnType(Args...),
            lambda_action<F, ReturnType, Args...>>
        {
            typedef lambda_action derived_type;

            static std::string get_action_name(naming::address::address_type /*lva*/)
            {
                return "lambda action()";
            }

            template <typename... Ts>
            static ReturnType invoke(naming::address::address_type /*lva*/,
                naming::address::component_type /*comptype*/, Ts&&... vs)
            {
                int* dummy = nullptr;
                return reinterpret_cast<const F&>(*dummy)(    // -V522
                    std::forward<Ts>(vs)...);
            }
        };

        template <typename F, typename T>
        struct extract_lambda_action
        {};

        // Specialization for lambdas
        template <typename F, typename ClassType, typename ReturnType, typename... Args>
        struct extract_lambda_action<F, ReturnType(ClassType::*)(Args...) const>
        {
            using type = lambda_action<F, ReturnType, Args...>;
        };

        template <typename F>
        struct action_from_lambda
        {
            using type =
                typename extract_lambda_action<F, decltype(&F::operator())>::type;
        };

        struct action_maker
        {
            template<typename F>
            constexpr typename hpx::actions::detail::action_from_lambda<F>::type
            operator += (F*) const
            {
                static_assert(
                    //!std::is_assignable<F,F>::value &&
                    std::is_empty<F>::value,
                    "lambda_to_action() needs and only needs a lambda with empty " \
                    "capture list");

                return typename
                    hpx::actions::detail::action_from_lambda<F>::type();
            }
        };
    }

    template <typename F>
    auto lambda_to_action(F&& f)
        -> decltype(hpx::actions::detail::action_maker() +=
            true ? nullptr : hpx::actions::detail::addr_add() + f)
    {
        return hpx::actions::detail::action_maker() +=
            true ? nullptr : hpx::actions::detail::addr_add() + f;
    }
}}

