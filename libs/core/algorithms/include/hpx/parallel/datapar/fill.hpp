//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/functional/tag_dispatch.hpp>
#include <hpx/parallel/algorithms/detail/fill.hpp>
#include <hpx/parallel/datapar/loop.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    struct datapar_fill
    {
        template <typename ExPolicy, typename Iter, typename Sent, typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            util::detail::iterator_datapar_compatible<Iter>::value, Iter>::type
        call(ExPolicy&& policy, Iter first, Sent last, T const& val)
        {
            hpx::parallel::util::loop_ind(std::forward<ExPolicy>(policy), first,
                last, [&val](auto& v) { v = val; });
            return first;
        }
    };

    template <typename ExPolicy, typename Iter, typename Sent, typename T,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_vectorpack_execution_policy<ExPolicy>::value&&
                hpx::parallel::util::detail::iterator_datapar_compatible<
                    Iter>::value)>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter tag_dispatch(sequential_fill_t,
        ExPolicy&& policy, Iter first, Sent last, T const& value)
    {
        return datapar_fill::call(
            std::forward<ExPolicy>(policy), first, last, value);
    }

    ///////////////////////////////////////////////////////////////////////////
    struct datapar_fill_n
    {
        template <typename ExPolicy, typename Iter, typename T>
        HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
            util::detail::iterator_datapar_compatible<Iter>::value, Iter>::type
        call(ExPolicy&&, Iter first, std::size_t count, T const& val)
        {
            hpx::parallel::util::loop_n_ind<std::decay_t<ExPolicy>>(
                first, count, [&val](auto& v) { v = val; });
            return first;
        }
    };

    template <typename ExPolicy, typename Iter, typename T,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_vectorpack_execution_policy<ExPolicy>::value&&
                hpx::parallel::util::detail::iterator_datapar_compatible<
                    Iter>::value)>
    HPX_HOST_DEVICE HPX_FORCEINLINE Iter tag_dispatch(sequential_fill_n_t,
        ExPolicy&& policy, Iter first, std::size_t count, T const& value)
    {
        return datapar_fill_n::call(
            std::forward<ExPolicy>(policy), first, count, value);
    }
}}}}    // namespace hpx::parallel::v1::detail
#endif
