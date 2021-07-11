//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/rotate.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/parallel/util/tagged_pair.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/transfer.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // shift_left
    namespace detail {
        /*
        template <typename ExPolicy, typename FwdIter>
        hpx::future<util::in_out_result<FwdIter, FwdIter>> rotate_helper(
            ExPolicy policy, FwdIter first, FwdIter new_first, FwdIter last)
        {
            using non_seq = std::false_type;

            auto p = hpx::execution::parallel_task_policy()
                         .on(policy.executor())
                         .with(policy.parameters());

            detail::reverse<FwdIter> r;
            return dataflow(
                [=](hpx::future<FwdIter>&& f1,
                    hpx::future<FwdIter>&& f2) mutable
                -> hpx::future<util::in_out_result<FwdIter, FwdIter>> {
                    // propagate exceptions
                    f1.get();
                    f2.get();

                    hpx::future<FwdIter> f = r.call2(p, non_seq(), first, last);
                    return f.then([=](hpx::future<FwdIter>&& f) mutable
                        -> util::in_out_result<FwdIter, FwdIter> {
                        f.get();    // propagate exceptions
                        std::advance(first, std::distance(new_first, last));
                        return util::in_out_result<FwdIter, FwdIter>{
                            first, last};
                    });
                },
                r.call2(p, non_seq(), first, new_first),
                r.call2(p, non_seq(), new_first, last));
        }*/

        /* Sequential shift_left implementation borrowed
        from https://github.com/danra/shift_proposal */

        template <class FwdIter, class N>
        constexpr N bounded_advance(FwdIter& i, N n, FwdIter const bound)
        {
            for (; n > 0 && i != bound; --n, void(++i))
            {
                ;
            }

            return n;
        }

        template <typename FwdIter, typename Size>
        static constexpr std::enable_if_t<
            hpx::traits::is_random_access_iterator_v<FwdIter>, FwdIter>
        sequential_shift_left(FwdIter first, FwdIter last, Size n)
        {
            if (n <= 0)
            {
                return first;
            }

            auto mid = first;
            if (detail::bounded_advance(mid, n, last))
            {
                return first;
            }

            return parallel::util::get_second_element(util::move_n(
                mid, detail::distance(mid, last), std::move(first)));
        }

        template <typename FwdIter, typename Size>
        static constexpr std::enable_if_t<
            !hpx::traits::is_random_access_iterator_v<FwdIter>, FwdIter>
        sequential_shift_left(FwdIter first, FwdIter last, Size n)
        {
            if (n <= 0)
            {
                return first;
            }

            auto mid = first;
            if (detail::bounded_advance(mid, n, last))
            {
                return first;
            }

            return parallel::util::get_second_element(util::move(
                std::move(mid), std::move(last), std::move(first)));
        }

        template <typename FwdIter2>
        struct shift_left
          : public detail::algorithm<shift_left<FwdIter2>, FwdIter2>
        {
            shift_left()
              : shift_left::algorithm("shift_left")
            {
            }

            template <typename ExPolicy, typename FwdIter, typename Size>
            static FwdIter sequential(
                ExPolicy, FwdIter first, FwdIter last, Size n)
            {
                return detail::sequential_shift_left(first, last, n);
            }

            template <typename ExPolicy>
            static typename util::detail::algorithm_result<ExPolicy,
                FwdIter2>::type
            parallel(ExPolicy&& policy, FwdIter2 first, FwdIter2 new_first,
                FwdIter2 last)
            {
                return util::detail::algorithm_result<ExPolicy, FwdIter2>::get(
                    first);
            }
        };
        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::shift_left
    HPX_INLINE_CONSTEXPR_VARIABLE struct shift_left_t final
      : hpx::functional::tag_fallback<shift_left_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter>::value)>
        // clang-format on
        friend FwdIter tag_fallback_dispatch(
            shift_left_t, FwdIter first, FwdIter last, Size n)
        {
            return hpx::parallel::v1::detail::shift_left<FwdIter>().call(
                hpx::execution::seq, first, last, n);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value)>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_dispatch(shift_left_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, Size n)
        {
            return hpx::parallel::v1::detail::shift_left<FwdIter>().call(
                std::forward<ExPolicy>(policy), first, last, n);
        }
    } shift_left{};
}    // namespace hpx
