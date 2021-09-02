//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/datastructures/member_pack.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits { namespace detail {

    template <typename F, typename... Ts>
    struct is_deferred_invocable
      : hpx::is_invocable<util::decay_unwrap_t<F>, util::decay_unwrap_t<Ts>...>
    {
    };

    template <typename F, typename... Ts>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_deferred_invocable_v =
        is_deferred_invocable<F, Ts...>::value;

}}}    // namespace hpx::traits::detail

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename F, typename... Ts>
        struct invoke_deferred_result
          : util::invoke_result<util::decay_unwrap_t<F>,
                util::decay_unwrap_t<Ts>...>
        {
        };

        template <typename F, typename... Ts>
        using invoke_deferred_result_t =
            typename invoke_deferred_result<F, Ts...>::type;

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Is, typename... Ts>
        class deferred;

        template <typename F, std::size_t... Is, typename... Ts>
        class deferred<F, index_pack<Is...>, Ts...>
        {
        public:
            deferred() = default;    // needed for serialization

            template <typename F_, typename... Ts_,
                typename = std::enable_if_t<std::is_constructible_v<F, F_&&>>>
            explicit constexpr HPX_HOST_DEVICE deferred(F_&& f, Ts_&&... vs)
              : _f(std::forward<F_>(f))
              , _args(std::piecewise_construct, std::forward<Ts_>(vs)...)
            {
            }

#if !defined(__NVCC__) && !defined(__CUDACC__)
            deferred(deferred&&) = default;
#else
            constexpr HPX_HOST_DEVICE deferred(deferred&& other)
              : _f(std::move(other._f))
              , _args(std::move(other._args))
            {
            }
#endif

            deferred(deferred const&) = delete;
            deferred& operator=(deferred const&) = delete;

            HPX_HOST_DEVICE HPX_FORCEINLINE util::invoke_result_t<F, Ts...>
            operator()()
            {
                return HPX_INVOKE(
                    std::move(_f), std::move(_args).template get<Is>()...);
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                // clang-format off
                ar & _f;
                ar & _args;
                // clang-format on
            }

            constexpr std::size_t get_function_address() const
            {
                return traits::get_function_address<F>::call(_f);
            }

            constexpr char const* get_function_annotation() const
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
                static util::itt::string_handle sh("deferred");
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
    detail::deferred<std::decay_t<F>, util::make_index_pack_t<sizeof...(Ts)>,
        util::decay_unwrap_t<Ts>...>
    deferred_call(F&& f, Ts&&... vs)
    {
        static_assert(traits::detail::is_deferred_invocable_v<F, Ts...>,
            "F shall be Callable with decay_t<Ts> arguments");

        using result_type = detail::deferred<std::decay_t<F>,
            util::make_index_pack_t<sizeof...(Ts)>,
            util::decay_unwrap_t<Ts>...>;

        return result_type(std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    // nullary functions do not need to be bound again
    template <typename F>
    inline std::decay_t<F> deferred_call(F&& f)
    {
        static_assert(traits::detail::is_deferred_invocable_v<F>,
            "F shall be Callable with no arguments");

        return std::forward<F>(f);
    }
}}    // namespace hpx::util

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_address<util::detail::deferred<F, Ts...>>
    {
        static constexpr std::size_t call(
            util::detail::deferred<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_annotation<util::detail::deferred<F, Ts...>>
    {
        static constexpr char const* call(
            util::detail::deferred<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename F, typename... Ts>
    struct get_function_annotation_itt<util::detail::deferred<F, Ts...>>
    {
        static util::itt::string_handle call(
            util::detail::deferred<F, Ts...> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
}}    // namespace hpx::traits
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive, typename F, typename... Ts>
    HPX_FORCEINLINE void serialize(Archive& ar,
        ::hpx::util::detail::deferred<F, Ts...>& d,
        unsigned int const version = 0)
    {
        d.serialize(ar, version);
    }
}}    // namespace hpx::serialization
