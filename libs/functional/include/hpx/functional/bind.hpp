//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/member_pack.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/one_shot.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/functional/traits/is_bind_expression.hpp>
#include <hpx/functional/traits/is_placeholder.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <std::size_t I>
        struct placeholder
        {
            static std::size_t const value = I;
        };

        template <>
        struct placeholder<0>;    // not a valid placeholder
    }                             // namespace detail

    namespace placeholders {
        HPX_STATIC_CONSTEXPR detail::placeholder<1> _1 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<2> _2 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<3> _3 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<4> _4 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<5> _5 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<6> _6 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<7> _7 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<8> _8 = {};
        HPX_STATIC_CONSTEXPR detail::placeholder<9> _9 = {};
    }    // namespace placeholders

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <std::size_t I>
        struct bind_eval_placeholder
        {
            template <typename T, typename... Us>
            static constexpr HPX_HOST_DEVICE decltype(auto) call(
                T&& /*t*/, Us&&... vs)
            {
                return util::member_pack_for<Us&&...>(
                    std::piecewise_construct, std::forward<Us>(vs)...)
                    .template get<I>();
            }
        };

        template <typename T, typename TD = typename std::decay<T>::type,
            typename Enable = void>
        struct bind_eval
        {
            template <typename... Us>
            static constexpr HPX_HOST_DEVICE T&& call(T&& t, Us&&... /*vs*/)
            {
                return std::forward<T>(t);
            }
        };

        template <typename T, typename TD>
        struct bind_eval<T, TD,
            typename std::enable_if<traits::is_placeholder<TD>::value !=
                0>::type>
          : bind_eval_placeholder<
                (std::size_t) traits::is_placeholder<TD>::value - 1>
        {
        };

        template <typename T, typename TD>
        struct bind_eval<T, TD,
            typename std::enable_if<
                traits::is_bind_expression<TD>::value>::type>
        {
            template <typename... Us>
            static constexpr HPX_HOST_DEVICE
                typename util::invoke_result<T, Us...>::type
                call(T&& t, Us&&... vs)
            {
                return HPX_INVOKE(std::forward<T>(t), std::forward<Us>(vs)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Ts, typename... Us>
        struct invoke_bound_result;

        template <typename F, typename... Ts, typename... Us>
        struct invoke_bound_result<F, util::pack<Ts...>, Us...>
          : util::invoke_result<F,
                decltype(bind_eval<Ts>::call(
                    std::declval<Ts>(), std::declval<Us>()...))...>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Is, typename... Ts>
        class bound;

        template <typename F, std::size_t... Is, typename... Ts>
        class bound<F, index_pack<Is...>, Ts...>
        {
        public:
            bound() = default;    // needed for serialization

            template <typename F_, typename... Ts_,
                typename = typename std::enable_if<
                    std::is_constructible<F, F_>::value>::type>
            constexpr explicit bound(F_&& f, Ts_&&... vs)
              : _f(std::forward<F_>(f))
              , _args(std::piecewise_construct, std::forward<Ts_>(vs)...)
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            bound(bound const&) = default;
            bound(bound&&) = default;
#else
            constexpr HPX_HOST_DEVICE bound(bound const& other)
              : _f(other._f)
              , _args(other._args)
            {
            }

            constexpr HPX_HOST_DEVICE bound(bound&& other)
              : _f(std::move(other._f))
              , _args(std::move(other._args))
            {
            }
#endif

            bound& operator=(bound const&) = delete;

            template <typename... Us>
            constexpr HPX_HOST_DEVICE typename invoke_bound_result<F&,
                util::pack<Ts&...>, Us&&...>::type
            operator()(Us&&... vs) &
            {
                return HPX_INVOKE(_f,
                    detail::bind_eval<Ts&>::call(
                        _args.template get<Is>(), std::forward<Us>(vs)...)...);
            }

            template <typename... Us>
            constexpr HPX_HOST_DEVICE typename invoke_bound_result<F const&,
                util::pack<Ts const&...>, Us&&...>::type
            operator()(Us&&... vs) const&
            {
                return HPX_INVOKE(_f,
                    detail::bind_eval<Ts const&>::call(
                        _args.template get<Is>(), std::forward<Us>(vs)...)...);
            }

            template <typename... Us>
            constexpr HPX_HOST_DEVICE typename invoke_bound_result<F&&,
                util::pack<Ts&&...>, Us&&...>::type
            operator()(Us&&... vs) &&
            {
                return HPX_INVOKE(std::move(_f),
                    detail::bind_eval<Ts>::call(
                        std::move(_args).template get<Is>(),
                        std::forward<Us>(vs)...)...);
            }

            template <typename... Us>
            constexpr HPX_HOST_DEVICE typename invoke_bound_result<F const&&,
                util::pack<Ts const&&...>, Us&&...>::type
            operator()(Us&&... vs) const&&
            {
                return HPX_INVOKE(std::move(_f),
                    detail::bind_eval<Ts const>::call(
                        std::move(_args).template get<Is>(),
                        std::forward<Us>(vs)...)...);
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
                static util::itt::string_handle sh("bound");
                return sh;
#endif
            }
#endif

        private:
            F _f;
            util::member_pack_for<Ts...> _args;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts,
        typename Enable = typename std::enable_if<
            !traits::is_action<typename std::decay<F>::type>::value>::type>
    constexpr detail::bound<typename std::decay<F>::type,
        typename util::make_index_pack<sizeof...(Ts)>::type,
        typename util::decay_unwrap<Ts>::type...>
    bind(F&& f, Ts&&... vs)
    {
        typedef detail::bound<typename std::decay<F>::type,
            typename util::make_index_pack<sizeof...(Ts)>::type,
            typename util::decay_unwrap<Ts>::type...>
            result_type;

        return result_type(std::forward<F>(f), std::forward<Ts>(vs)...);
    }
}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct is_bind_expression<util::detail::bound<F, Ts...>> : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t I>
    struct is_placeholder<util::detail::placeholder<I>>
      : std::integral_constant<int, I>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    template <typename F, typename... Ts>
    struct get_function_address<util::detail::bound<F, Ts...>>
    {
        static std::size_t call(util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_annotation<util::detail::bound<F, Ts...>>
    {
        static char const* call(util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename F, typename... Ts>
    struct get_function_annotation_itt<util::detail::bound<F, Ts...>>
    {
        static util::itt::string_handle call(
            util::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
#endif
}}    // namespace hpx::traits

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization {
    // serialization of the bound object
    template <typename Archive, typename F, typename... Ts>
    void serialize(Archive& ar, ::hpx::util::detail::bound<F, Ts...>& bound,
        unsigned int const version = 0)
    {
        bound.serialize(ar, version);
    }

    // serialization of placeholders is trivial, just provide empty functions
    template <typename Archive, std::size_t I>
    void serialize(Archive& ar,
        ::hpx::util::detail::placeholder<I>& /*placeholder*/
        ,
        unsigned int const /*version*/ = 0)
    {
    }
}}    // namespace hpx::serialization
