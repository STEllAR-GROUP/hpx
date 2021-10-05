//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/functional/tag_dispatch.hpp>
#include <hpx/parallel/algorithms/detail/generate.hpp>
#include <hpx/parallel/datapar/loop.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    struct datapar_generate
    {
        template <typename ExPolicy, typename Iter, typename Sent, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            util::detail::iterator_datapar_compatible<Iter>::value, Iter>::type
        call(ExPolicy&& policy, Iter first, Sent last, F&& f)
        {
            hpx::parallel::util::loop_ind(std::forward<ExPolicy>(policy), first,
                last, [f = std::forward<F>(f)](auto& v) mutable { v = f(); });
            return first;
        }

        template <typename ExPolicy, typename Iter, typename Sent, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            !util::detail::iterator_datapar_compatible<Iter>::value, Iter>::type
        call(ExPolicy&&, Iter first, Sent last, F&& f)
        {
            return sequential_generate_helper(first, last, std::forward<F>(f));
        }
    };

    template <typename ExPolicy, typename Iter, typename Sent, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iter>::type
    tag_dispatch(
        sequential_generate_t, ExPolicy&& policy, Iter first, Sent last, F&& f)
    {
        return datapar_generate::call(
            std::forward<ExPolicy>(policy), first, last, std::forward<F>(f));
    }

    ///////////////////////////////////////////////////////////////////////////
    struct datapar_generate_n
    {
        template <typename ExPolicy, typename Iter, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            util::detail::iterator_datapar_compatible<Iter>::value, Iter>::type
        call(ExPolicy&&, Iter first, std::size_t count, F&& f)
        {
            hpx::parallel::util::loop_n_ind<std::decay_t<ExPolicy>>(first,
                count, [f = std::forward<F>(f)](auto& v) mutable { v = f(); });
            return first;
        }

        template <typename ExPolicy, typename Iter, typename F>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            !util::detail::iterator_datapar_compatible<Iter>::value, Iter>::type
        call(ExPolicy&&, Iter first, std::size_t count, F&& f)
        {
            return sequential_generate_n_helper(
                first, count, std::forward<F>(f));
        }
    };

    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE typename std::enable_if<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iter>::type
    tag_dispatch(sequential_generate_n_t, ExPolicy&& policy, Iter first,
        std::size_t count, F&& f)
    {
        return datapar_generate_n::call(
            std::forward<ExPolicy>(policy), first, count, std::forward<F>(f));
    }
}}}}    // namespace hpx::parallel::v1::detail
#endif
