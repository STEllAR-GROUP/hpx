//  Copyright (c) 2017-2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <functional>
#include <type_traits>
#include <utility>

namespace hpx::util::detail {

    ///////////////////////////////////////////////////////////////////////////
    // when `pm` is a pointer to member of a class `C` and
    // `is_base_of_v<C, remove_reference_t<T>>` is `true`;
    template <typename C, typename T,
        typename =
            std::enable_if_t<std::is_base_of_v<C, std::remove_reference_t<T>>>>
    static constexpr T&& mem_ptr_target(T&& v) noexcept
    {
        return HPX_FORWARD(T, v);
    }

    // when `pm` is a pointer to member of a class `C` and
    // `remove_cvref_t<T>` is a specialization of `reference_wrapper`;
    template <typename C, typename T>
    static constexpr T& mem_ptr_target(std::reference_wrapper<T> v) noexcept
    {
        return v.get();
    }

    // when `pm` is a pointer to member of a class `C` and `T` does not
    // satisfy the previous two items;
    //
    // Note: NVCC requires to use std::forward below
    template <typename C, typename T>
    static constexpr auto mem_ptr_target(T&& v) noexcept(
        noexcept(*std::forward<T>(v))) -> decltype(*std::forward<T>(v))
    {
        return *std::forward<T>(v);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename C>
    struct invoke_mem_obj
    {
        T C::*pm;

    public:
        explicit constexpr invoke_mem_obj(T C::*pm) noexcept
          : pm(pm)
        {
        }

        template <typename T1>
        constexpr auto operator()(T1&& t1) const noexcept(
            noexcept(detail::mem_ptr_target<C>(HPX_FORWARD(T1, t1)).*pm))
            -> decltype(detail::mem_ptr_target<C>(HPX_FORWARD(T1, t1)).*pm)
        {
            // This seems to trigger a bogus warning in GCC 11 with
            // optimizations enabled (possibly the same as this:
            // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98503) so we disable
            // the warning locally.
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
            return detail::mem_ptr_target<C>(HPX_FORWARD(T1, t1)).*pm;
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif
        }
    };

    template <typename T, typename C>
    struct invoke_mem_fun
    {
        T C::*pm;

    public:
        explicit constexpr invoke_mem_fun(T C::*pm) noexcept
          : pm(pm)
        {
        }

        template <typename T1, typename... Tn>
        constexpr auto operator()(T1&& t1, Tn&&... tn) const
            noexcept(noexcept((detail::mem_ptr_target<C>(HPX_FORWARD(T1, t1)).*
                pm)(HPX_FORWARD(Tn, tn)...)))
                -> decltype((detail::mem_ptr_target<C>(HPX_FORWARD(T1, t1)).*
                    pm)(HPX_FORWARD(Tn, tn)...))
        {
            // This seems to trigger a bogus warning in GCC 11 with
            // optimizations enabled (possibly the same as this:
            // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98503) so we disable
            // the warning locally.
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable : 26800)    //  Use of a moved from object: '(*t1)'
#endif

            return (detail::mem_ptr_target<C>(HPX_FORWARD(T1, t1)).*pm)(
                HPX_FORWARD(Tn, tn)...);

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F,
        typename FD = std::remove_cv_t<std::remove_reference_t<F>>>
    struct dispatch_invoke
    {
        using type = F&&;
    };

    template <typename F, typename T, typename C>
    struct dispatch_invoke<F, T C::*>
    {
        using type = std::conditional_t<std::is_function_v<T>,
            invoke_mem_fun<T, C>, invoke_mem_obj<T, C>>;
    };

    template <typename F>
    using invoke = typename dispatch_invoke<F>::type;

#define HPX_INVOKE(F, ...)                                                     \
    (::hpx::util::detail::invoke<decltype((F))>(F)(__VA_ARGS__))

}    // namespace hpx::util::detail
