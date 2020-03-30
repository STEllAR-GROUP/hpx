//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/invoke_fused.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util { namespace detail {
    // Hand-crafted function object allowing to replace a more complex
    // bind(functional::invoke_fused(), f1, _1)
    template <typename Result, typename F>
    struct partitioner_iteration
    {
        typename std::decay<F>::type f_;

        template <typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE Result operator()(T&& t)
        {
            return hpx::util::invoke_fused_r<Result>(f_, std::forward<T>(t));
        }
    };
}}}}    // namespace hpx::parallel::util::detail

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>

namespace hpx { namespace traits {
    template <typename Result, typename F>
    struct get_function_address<
        parallel::util::detail::partitioner_iteration<Result, F>>
    {
        static std::size_t call(
            parallel::util::detail::partitioner_iteration<Result, F> const&
                f) noexcept
        {
            return get_function_address<typename std::decay<F>::type>::call(
                f.f_);
        }
    };

    template <typename Result, typename F>
    struct get_function_annotation<
        parallel::util::detail::partitioner_iteration<Result, F>>
    {
        static char const* call(
            parallel::util::detail::partitioner_iteration<Result, F> const&
                f) noexcept
        {
            return get_function_annotation<typename std::decay<F>::type>::call(
                f.f_);
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Result, typename F>
    struct get_function_annotation_itt<
        parallel::util::detail::partitioner_iteration<Result, F>>
    {
        static util::itt::string_handle call(
            parallel::util::detail::partitioner_iteration<Result, F> const&
                f) noexcept
        {
            return get_function_annotation_itt<
                typename std::decay<F>::type>::call(f.f_);
        }
    };
#endif
}}    // namespace hpx::traits
#endif
