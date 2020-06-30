//  Copyright (c) 2015 Hartmut Kaiser
//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions_base/traits/extract_action.hpp>
#include <hpx/async_distributed/applier/apply.hpp>
#include <hpx/async_local/async_fwd.hpp>
#include <hpx/datastructures/member_pack.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/functional/traits/is_bind_expression.hpp>
#include <hpx/functional/traits/is_placeholder.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/promise_local_result.hpp>
#include <hpx/traits/is_continuation.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename Action, typename Is, typename... Ts>
        class bound_action;

        template <typename Action, std::size_t... Is, typename... Ts>
        class bound_action<Action, index_pack<Is...>, Ts...>
        {
        public:
            typedef typename traits::promise_local_result<
                typename hpx::traits::extract_action<
                    Action>::remote_result_type>::type result_type;

        public:
            // default constructor is needed for serialization
            bound_action() = default;

            template <typename Derived, typename... Ts_>
            explicit bound_action(Derived /*action*/, Ts_&&... vs)
              : _args(std::piecewise_construct, std::forward<Ts_>(vs)...)
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            bound_action(bound_action const&) = default;
            bound_action(bound_action&&) = default;
#else
            HPX_HOST_DEVICE bound_action(bound_action const& other)
              : _args(other._args)
            {
            }

            HPX_HOST_DEVICE bound_action(bound_action&& other)
              : _args(std::move(other._args))
            {
            }
#endif

            bound_action& operator=(bound_action const&) = delete;

            template <typename... Us>
            HPX_FORCEINLINE bool apply(Us&&... vs) const
            {
                return hpx::apply<Action>(detail::bind_eval<Ts const&>::call(
                    _args.template get<Is>(), std::forward<Us>(vs)...)...);
            }

            template <typename... Us>
            HPX_FORCEINLINE bool apply_c(
                naming::id_type const& cont, Us&&... vs) const
            {
                return hpx::apply_c<Action>(cont,
                    detail::bind_eval<Ts const&>::call(
                        _args.template get<Is>(), std::forward<Us>(vs)...)...);
            }

            template <typename Continuation, typename... Us>
            HPX_FORCEINLINE typename std::enable_if<
                traits::is_continuation<Continuation>::value, bool>::type
            apply_c(Continuation&& cont, Us&&... vs) const
            {
                return hpx::apply<Action>(std::forward<Continuation>(cont),
                    detail::bind_eval<Ts const&>::call(
                        _args.template get<Is>(), std::forward<Us>(vs)...)...);
            }

            template <typename... Us>
            HPX_FORCEINLINE hpx::lcos::future<result_type> async(
                Us&&... vs) const
            {
                return hpx::async<Action>(detail::bind_eval<Ts const&>::call(
                    _args.template get<Is>(), std::forward<Us>(vs)...)...);
            }

            template <typename... Us>
            HPX_FORCEINLINE result_type operator()(Us&&... vs) const
            {
                return async(std::forward<Us>(vs)...).get();
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                ar& _args;
            }

        private:
            util::member_pack_for<Ts...> _args;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename... Ts,
        typename Enable = typename std::enable_if<
            traits::is_action<typename std::decay<Action>::type>::value>::type>
    detail::bound_action<typename std::decay<Action>::type,
        typename util::make_index_pack<sizeof...(Ts)>::type,
        typename std::decay<Ts>::type...>
    bind(Ts&&... vs)
    {
        typedef detail::bound_action<typename std::decay<Action>::type,
            typename util::make_index_pack<sizeof...(Ts)>::type,
            typename std::decay<Ts>::type...>
            result_type;

        return result_type(Action(), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename... Ts>
    detail::bound_action<Derived,
        typename util::make_index_pack<sizeof...(Ts)>::type,
        typename std::decay<Ts>::type...>
    bind(hpx::actions::basic_action<Component, Signature, Derived> action,
        Ts&&... vs)
    {
        typedef detail::bound_action<Derived,
            typename util::make_index_pack<sizeof...(Ts)>::type,
            typename std::decay<Ts>::type...>
            result_type;

        return result_type(
            static_cast<Derived const&>(action), std::forward<Ts>(vs)...);
    }
}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Is, typename... Ts>
    struct is_bind_expression<util::detail::bound_action<Action, Is, Ts...>>
      : std::true_type
    {
    };

    template <typename Action, typename Is, typename... Ts>
    struct is_bound_action<util::detail::bound_action<Action, Is, Ts...>>
      : std::true_type
    {
    };
}}    // namespace hpx::traits

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization {
    // serialization of the bound action object
    template <typename Archive, typename F, typename... Ts>
    void serialize(Archive& ar,
        ::hpx::util::detail::bound_action<F, Ts...>& bound,
        unsigned int const version = 0)
    {
        bound.serialize(ar, version);
    }
}}    // namespace hpx::serialization

