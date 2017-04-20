//  Copyright (c) 2017 Antoine TRAN TAN
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/actions/lambda_to_action.hpp

#ifndef HPX_RUNTIME_ACTIONS_LAMBDA_TO_ACTION_HPP
#define HPX_RUNTIME_ACTIONS_LAMBDA_TO_ACTION_HPP

#include <hpx/include/plain_actions.hpp>
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

        template<typename F, typename ReturnType, typename ... Args>
        struct caller{
          static inline ReturnType call(Args && ... args)
          {
             int * dummy = nullptr;
             return reinterpret_cast<const F&>(*dummy)( std::forward<Args>(args)... );
          }
        };

        template<class F, typename ReturnType, typename Sequence>
        struct make_action_using_sequence
        {};

        template<class F, typename ReturnType,
          template <typename...> class T, typename ... Args>
        struct make_action_using_sequence< F, ReturnType, T<Args...> >
        {
          using type =
              typename hpx::actions::make_action<
                decltype(&caller<F,ReturnType,Args...>::call),
                &caller<F,ReturnType,Args...>::call >::type;
        };

        template <typename T>
        struct extract_parameters
        {};

        // Specialization for lambdas
        template <typename ClassType, typename ReturnType, typename... Args>
        struct extract_parameters<ReturnType(ClassType::*)(Args...) const>
        {
            using type = hpx::actions::detail::sequence< Args... >;
        };

        template <typename T>
        struct extract_return_type
        {};

        template <typename ClassType, typename ReturnType, typename... Args>
        struct extract_return_type<ReturnType(ClassType::*)(Args...) const>
        {
            using type = ReturnType;
        };

        template <typename F>
        struct action_from_lambda
        {
            using sequence_type =
                typename hpx::actions::detail::extract_parameters<
                    decltype(&F::operator())>::type;
            using return_type =
                typename hpx::actions::detail::extract_return_type<
                    decltype(&F::operator())>::type;
            using type =
                typename hpx::actions::detail::make_action_using_sequence<
                    F,return_type,sequence_type>::type;
        };

#if defined(HPX_MSVC)
// suppress: 'operator +=': in C++14 'constexpr' will not imply 'const';
// consider explicitly specifying 'const'
#pragma warning(push)
#pragma warning(disable: 4814)
#endif

        struct action_maker
        {
            template<typename F>
            HPX_CONSTEXPR typename hpx::actions::detail::action_from_lambda<F>::type
            operator += (F* f)
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

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
    }

    template<typename F>
    auto lambda_to_action(F&& f)
    -> decltype( hpx::actions::detail::action_maker() += true
        ? nullptr
        : hpx::actions::detail::addr_add() +  f)
    {
        HPX_CONSTEXPR auto act =
            hpx::actions::detail::action_maker() += true
            ? nullptr
            : hpx::actions::detail::addr_add() +  f;

        return act;
    }
}}

#endif /*HPX_RUNTIME_ACTIONS_LAMBDA_TO_ACTION_HPP*/
