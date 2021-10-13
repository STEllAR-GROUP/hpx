//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/functional/tag_dispatch.hpp>
#include <hpx/iterator_support/zip_iterator.hpp>
#include <hpx/parallel/algorithms/detail/adjacent_difference.hpp>
#include <hpx/parallel/datapar/iterator_helpers.hpp>
#include <hpx/parallel/datapar/loop.hpp>
#include <hpx/parallel/datapar/zip_iterator.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_adjacent_difference
    {
        template <typename InIter, typename OutIter, typename Op>
        static inline OutIter call(
            InIter first, InIter last, OutIter dest, Op&& op)
        {
            if (first == last)
                return dest;
            auto count = std::distance(first, last) - 1;

            InIter prev = first;
            *dest++ = *first++;

            if (count == 0)
            {
                return dest;
            }

            using hpx::get;
            using hpx::util::make_zip_iterator;
            util::loop_n<std::decay_t<ExPolicy>>(
                make_zip_iterator(first, prev, dest), count,
                [op](auto&& it) mutable {
                    get<2>(*it) = HPX_INVOKE(op, get<0>(*it), get<1>(*it));
                });
            std::advance(dest, count);
            return dest;
        }
    };

    template <typename ExPolicy, typename InIter, typename OutIter, typename Op,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_vectorpack_execution_policy<ExPolicy>::value&&
                hpx::parallel::util::detail::iterator_datapar_compatible<
                    InIter>::value)>
    inline OutIter tag_dispatch(sequential_adjacent_difference_t<ExPolicy>,
        InIter first, InIter last, OutIter dest, Op&& op)
    {
        return datapar_adjacent_difference<ExPolicy>::call(
            first, last, dest, std::forward<Op>(op));
    }
}}}}    // namespace hpx::parallel::v1::detail
#endif
