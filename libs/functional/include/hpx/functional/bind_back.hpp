//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/member_pack.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/one_shot.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    namespace detail {
        template <typename F, typename Ts, typename... Us>
        struct invoke_bound_back_result;

        template <typename F, typename... Ts, typename... Us>
        struct invoke_bound_back_result<F, util::pack<Ts...>, Us...>
          : util::invoke_result<F, Us..., Ts...>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Is, typename... Ts>
        class bound_back;

        template <typename F, std::size_t... Is, typename... Ts>
        class bound_back<F, index_pack<Is...>, Ts...>
        {
        public:
            bound_back() = default;    // needed for serialization

            template <typename F_, typename... Ts_,
                typename = typename std::enable_if<
                    std::is_constructible<F, F_>::value>::type>
            constexpr explicit bound_back(F_&& f, Ts_&&... vs)
              : _f(std::forward<F_>(f))
              , _args(std::piecewise_construct, std::forward<Ts_>(vs)...)
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            bound_back(bound_back const&) = default;
            bound_back(bound_back&&) = default;
#else
            constexpr HPX_HOST_DEVICE bound_back(bound_back const& other)
              : _f(other._f)
              , _args(other._args)
            {
            }

            constexpr HPX_HOST_DEVICE bound_back(bound_back&& other)
              : _f(std::move(other._f))
              , _args(std::move(other._args))
            {
            }
#endif

            bound_back& operator=(bound_back const&) = delete;

            template <typename... Us>
            constexpr HPX_HOST_DEVICE typename invoke_bound_back_result<F&,
                util::pack<Ts&...>, Us&&...>::type
            operator()(Us&&... vs) &
            {
                return HPX_INVOKE(
                    _f, std::forward<Us>(vs)..., _args.template get<Is>()...);
            }

            template <typename... Us>
            constexpr HPX_HOST_DEVICE
                typename invoke_bound_back_result<F const&,
                    util::pack<Ts const&...>, Us&&...>::type
                operator()(Us&&... vs) const&
            {
                return HPX_INVOKE(
                    _f, std::forward<Us>(vs)..., _args.template get<Is>()...);
            }

            template <typename... Us>
            constexpr HPX_HOST_DEVICE typename invoke_bound_back_result<F&&,
                util::pack<Ts&&...>, Us&&...>::type
            operator()(Us&&... vs) &&
            {
                return HPX_INVOKE(std::move(_f), std::forward<Us>(vs)...,
                    std::move(_args).template get<Is>()...);
            }

            template <typename... Us>
            constexpr HPX_HOST_DEVICE
                typename invoke_bound_back_result<F const&&,
                    util::pack<Ts const&&...>, Us&&...>::type
                operator()(Us&&... vs) const&&
            {
                return HPX_INVOKE(std::move(_f), std::forward<Us>(vs)...,
                    std::move(_args).template get<Is>()...);
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                ar& _f;
                ar& _args;
            }

            std::size_t get_function_address() const
            {
                return traits::get_function_address<F>::call(_f);
            }

            char const* get_function_annotation() const
            {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation<F>::call(_f);
#else
                return nullptr;
#endif
            }

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            util::itt::string_handle get_function_annotation_itt() const
            {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation_itt<F>::call(_f);
#else
                static util::itt::string_handle sh("bound_back");
                return sh;
#endif
            }
#endif

        private:
            F _f;
            util::member_pack_for<Ts...> _args;
        };
    }    // namespace detail

    template <typename F, typename... Ts>
    constexpr detail::bound_back<typename std::decay<F>::type,
        typename util::make_index_pack<sizeof...(Ts)>::type,
        typename util::decay_unwrap<Ts>::type...>
    bind_back(F&& f, Ts&&... vs)
    {
        typedef detail::bound_back<typename std::decay<F>::type,
            typename util::make_index_pack<sizeof...(Ts)>::type,
            typename util::decay_unwrap<Ts>::type...>
            result_type;

        return result_type(std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    // nullary functions do not need to be bound again
    template <typename F>
    constexpr typename std::decay<F>::type bind_back(F&& f)
    {
        return std::forward<F>(f);
    }
}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    template <typename F, typename... Ts>
    struct get_function_address<util::detail::bound_back<F, Ts...>>
    {
        static std::size_t call(
            util::detail::bound_back<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_annotation<util::detail::bound_back<F, Ts...>>
    {
        static char const* call(
            util::detail::bound_back<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename F, typename... Ts>
    struct get_function_annotation_itt<util::detail::bound_back<F, Ts...>>
    {
        static util::itt::string_handle call(
            util::detail::bound_back<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
#endif
}}    // namespace hpx::traits

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization {
    // serialization of the bound_back object
    template <typename Archive, typename F, typename... Ts>
    void serialize(Archive& ar,
        ::hpx::util::detail::bound_back<F, Ts...>& bound,
        unsigned int const version = 0)
    {
        bound.serialize(ar, version);
    }
}}    // namespace hpx::serialization
