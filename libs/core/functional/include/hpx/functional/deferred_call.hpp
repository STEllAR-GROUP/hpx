//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
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
    template <typename T>
    struct HPX_DEPRECATED_V(1, 5,
        "is_deferred_callable is deprecated, use is_deferred_invocable "
        "instead.") is_deferred_callable;

#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    template <typename F, typename... Ts>
    struct is_deferred_callable<F(Ts...)>
      : hpx::is_invocable<typename util::decay_unwrap<F>::type,
            typename util::decay_unwrap<Ts>::type...>
    {
    };
#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic pop
#endif

    template <typename F, typename... Ts>
    struct is_deferred_invocable
      : hpx::is_invocable<typename util::decay_unwrap<F>::type,
            typename util::decay_unwrap<Ts>::type...>
    {
    };

}}}    // namespace hpx::traits::detail

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename T>
        struct HPX_DEPRECATED_V(1, 5,
            "deferred_result_of is deprecated, use invoke_deferred_result "
            "instead.") deferred_result_of;

#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        template <typename F, typename... Ts>
        struct deferred_result_of<F(Ts...)>
          : util::invoke_result<typename util::decay_unwrap<F>::type,
                typename util::decay_unwrap<Ts>::type...>
        {
        };
#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic pop
#endif

        template <typename F, typename... Ts>
        struct invoke_deferred_result
          : util::invoke_result<typename util::decay_unwrap<F>::type,
                typename util::decay_unwrap<Ts>::type...>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Is, typename... Ts>
        class deferred;

        template <typename F, std::size_t... Is, typename... Ts>
        class deferred<F, index_pack<Is...>, Ts...>
        {
        public:
            deferred() = default;    // needed for serialization

            template <typename F_, typename... Ts_,
                typename = typename std::enable_if<
                    std::is_constructible<F, F_&&>::value>::type>
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

            HPX_HOST_DEVICE HPX_FORCEINLINE
                typename util::invoke_result<F, Ts...>::type
                operator()()
            {
                return HPX_INVOKE(
                    std::move(_f), std::move(_args).template get<Is>()...);
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
    detail::deferred<typename std::decay<F>::type,
        typename util::make_index_pack<sizeof...(Ts)>::type,
        typename util::decay_unwrap<Ts>::type...>
    deferred_call(F&& f, Ts&&... vs)
    {
        static_assert(traits::detail::is_deferred_invocable<F, Ts...>::value,
            "F shall be Callable with decay_t<Ts> arguments");

        typedef detail::deferred<typename std::decay<F>::type,
            typename util::make_index_pack<sizeof...(Ts)>::type,
            typename util::decay_unwrap<Ts>::type...>
            result_type;

        return result_type(std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    // nullary functions do not need to be bound again
    template <typename F>
    inline typename std::decay<F>::type deferred_call(F&& f)
    {
        static_assert(traits::detail::is_deferred_invocable<F>::value,
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
        static std::size_t call(
            util::detail::deferred<F, Ts...> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct get_function_annotation<util::detail::deferred<F, Ts...>>
    {
        static char const* call(
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
