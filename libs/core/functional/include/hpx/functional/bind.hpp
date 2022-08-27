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

namespace hpx {

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

        inline constexpr detail::placeholder<1> _1 = {};
        inline constexpr detail::placeholder<2> _2 = {};
        inline constexpr detail::placeholder<3> _3 = {};
        inline constexpr detail::placeholder<4> _4 = {};
        inline constexpr detail::placeholder<5> _5 = {};
        inline constexpr detail::placeholder<6> _6 = {};
        inline constexpr detail::placeholder<7> _7 = {};
        inline constexpr detail::placeholder<8> _8 = {};
        inline constexpr detail::placeholder<9> _9 = {};
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
                    std::piecewise_construct, HPX_FORWARD(Us, vs)...)
                    .template get<I>();
            }
        };

        template <typename T, std::size_t NumUs, typename TD = std::decay_t<T>,
            typename Enable = void>
        struct bind_eval
        {
            template <typename... Us>
            static constexpr HPX_HOST_DEVICE T&& call(T&& t, Us&&... /*vs*/)
            {
                return HPX_FORWARD(T, t);
            }
        };

        template <typename T, std::size_t NumUs, typename TD>
        struct bind_eval<T, NumUs, TD,
            std::enable_if_t<hpx::is_placeholder_v<TD> != 0 &&
                (hpx::is_placeholder_v<TD> <= NumUs)>>
          : bind_eval_placeholder<(std::size_t) hpx::is_placeholder_v<TD> - 1>
        {
        };

        template <typename T, std::size_t NumUs, typename TD>
        struct bind_eval<T, NumUs, TD,
            std::enable_if_t<hpx::is_bind_expression_v<TD>>>
        {
            template <typename... Us>
            static constexpr HPX_HOST_DEVICE util::invoke_result_t<T, Us...>
            call(T&& t, Us&&... vs)
            {
                return HPX_INVOKE(HPX_FORWARD(T, t), HPX_FORWARD(Us, vs)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Ts, typename... Us>
        struct invoke_bound_result;

        template <typename F, typename... Ts, typename... Us>
        struct invoke_bound_result<F, util::pack<Ts...>, Us...>
          : util::invoke_result<F,
                decltype(bind_eval<Ts, sizeof...(Us)>::call(
                    std::declval<Ts>(), std::declval<Us>()...))...>
        {
        };

        template <typename F, typename Ts, typename... Us>
        using invoke_bound_result_t =
            typename invoke_bound_result<F, Ts, Us...>::type;

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Is, typename... Ts>
        class bound;

        template <typename F, std::size_t... Is, typename... Ts>
        class bound<F, util::index_pack<Is...>, Ts...>
        {
        public:
            bound() = default;    // needed for serialization

            template <typename F_, typename... Ts_,
                typename = std::enable_if_t<std::is_constructible_v<F, F_>>>
            constexpr explicit bound(F_&& f, Ts_&&... vs)
              : _f(HPX_FORWARD(F_, f))
              , _args(std::piecewise_construct, HPX_FORWARD(Ts_, vs)...)
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
              : _f(HPX_MOVE(other._f))
              , _args(HPX_MOVE(other._args))
            {
            }
#endif

            bound& operator=(bound const&) = delete;

            template <typename... Us>
            constexpr HPX_HOST_DEVICE
                invoke_bound_result_t<F&, util::pack<Ts&...>, Us&&...>
                operator()(Us&&... vs) &
            {
                return HPX_INVOKE(_f,
                    detail::bind_eval<Ts&, sizeof...(Us)>::call(
                        _args.template get<Is>(), HPX_FORWARD(Us, vs)...)...);
            }

            template <typename... Us>
            constexpr HPX_HOST_DEVICE invoke_bound_result_t<F const&,
                util::pack<Ts const&...>, Us&&...>
            operator()(Us&&... vs) const&
            {
                return HPX_INVOKE(_f,
                    detail::bind_eval<Ts const&, sizeof...(Us)>::call(
                        _args.template get<Is>(), HPX_FORWARD(Us, vs)...)...);
            }

            template <typename... Us>
            constexpr HPX_HOST_DEVICE
                invoke_bound_result_t<F&&, util::pack<Ts&&...>, Us&&...>
                operator()(Us&&... vs) &&
            {
                return HPX_INVOKE(HPX_MOVE(_f),
                    detail::bind_eval<Ts, sizeof...(Us)>::call(
                        HPX_MOVE(_args).template get<Is>(),
                        HPX_FORWARD(Us, vs)...)...);
            }

            template <typename... Us>
            constexpr HPX_HOST_DEVICE invoke_bound_result_t<F const&&,
                util::pack<Ts const&&...>, Us&&...>
            operator()(Us&&... vs) const&&
            {
                return HPX_INVOKE(HPX_MOVE(_f),
                    detail::bind_eval<Ts const, sizeof...(Us)>::call(
                        HPX_MOVE(_args).template get<Is>(),
                        HPX_FORWARD(Us, vs)...)...);
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                // clang-format off
                ar & _f;
                ar & _args;
                // clang-format on
            }

            constexpr std::size_t get_function_address() const noexcept
            {
                return traits::get_function_address<F>::call(_f);
            }

            constexpr char const* get_function_annotation() const noexcept
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
        typename Enable =
            std::enable_if_t<!traits::is_action_v<std::decay_t<F>>>>
    constexpr detail::bound<std::decay_t<F>,
        util::make_index_pack_t<sizeof...(Ts)>, util::decay_unwrap_t<Ts>...>
    bind(F&& f, Ts&&... vs)
    {
        using result_type = detail::bound<std::decay_t<F>,
            util::make_index_pack_t<sizeof...(Ts)>,
            util::decay_unwrap_t<Ts>...>;

        return result_type(HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...);
    }
}    // namespace hpx

namespace hpx::util {

    template <typename F, typename... Ts>
    HPX_DEPRECATED_V(
        1, 8, "hpx::util::bind is deprecated, use hpx::bind instead")
    constexpr decltype(auto) bind(F&& f, Ts&&... ts)
    {
        return hpx::bind(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }

    namespace placeholders {

        HPX_DEPRECATED_V(1, 8,
            "hpx::placeholders::_1 is deprecated, use "
            "hpx::placeholders::_1 instead")
        inline constexpr hpx::detail::placeholder<1> _1 = {};
        HPX_DEPRECATED_V(1, 8,
            "hpx::placeholders::_2 is deprecated, use "
            "hpx::placeholders::_2 instead")
        inline constexpr hpx::detail::placeholder<2> _2 = {};
        HPX_DEPRECATED_V(1, 8,
            "hpx::placeholders::_3 is deprecated, use "
            "hpx::placeholders::_3 instead")
        inline constexpr hpx::detail::placeholder<3> _3 = {};
        HPX_DEPRECATED_V(1, 8,
            "hpx::placeholders::_4 is deprecated, use "
            "hpx::placeholders::_4 instead")
        inline constexpr hpx::detail::placeholder<4> _4 = {};
        HPX_DEPRECATED_V(1, 8,
            "hpx::placeholders::_5 is deprecated, use "
            "hpx::placeholders::_5 instead")
        inline constexpr hpx::detail::placeholder<5> _5 = {};
        HPX_DEPRECATED_V(1, 8,
            "hpx::placeholders::_6 is deprecated, use "
            "hpx::placeholders::_6 instead")
        inline constexpr hpx::detail::placeholder<6> _6 = {};
        HPX_DEPRECATED_V(1, 8,
            "hpx::placeholders::_7 is deprecated, use "
            "hpx::placeholders::_7 instead")
        inline constexpr hpx::detail::placeholder<7> _7 = {};
        HPX_DEPRECATED_V(1, 8,
            "hpx::placeholders::_8 is deprecated, use "
            "hpx::placeholders::_8 instead")
        inline constexpr hpx::detail::placeholder<8> _8 = {};
        HPX_DEPRECATED_V(1, 8,
            "hpx::placeholders::_9 is deprecated, use "
            "hpx::placeholders::_9 instead")
        inline constexpr hpx::detail::placeholder<9> _9 = {};
    }    // namespace placeholders
}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct is_bind_expression<hpx::detail::bound<F, Ts...>> : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <std::size_t I>
    struct is_placeholder<hpx::detail::placeholder<I>>
      : std::integral_constant<int, I>
    {
    };
}    // namespace hpx

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    template <typename F, typename... Ts>
    struct get_function_address<hpx::detail::bound<F, Ts...>>
    {
        static constexpr std::size_t call(
            hpx::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_annotation<hpx::detail::bound<F, Ts...>>
    {
        static constexpr char const* call(
            hpx::detail::bound<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename F, typename... Ts>
    struct get_function_annotation_itt<hpx::detail::bound<F, Ts...>>
    {
        static util::itt::string_handle call(
            hpx::detail::bound<F, Ts...> const& f) noexcept
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
    void serialize(Archive& ar, ::hpx::detail::bound<F, Ts...>& bound,
        unsigned int const version = 0)
    {
        bound.serialize(ar, version);
    }

    // serialization of placeholders is trivial, just provide empty functions
    template <typename Archive, std::size_t I>
    constexpr void serialize(Archive& /* ar */,
        ::hpx::detail::placeholder<I>& /*placeholder*/,
        unsigned int const /*version*/ = 0) noexcept
    {
    }
}}    // namespace hpx::serialization
