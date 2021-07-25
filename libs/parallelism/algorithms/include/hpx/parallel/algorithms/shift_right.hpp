//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/shift_right.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_local/dataflow.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/parallel/util/tagged_pair.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/reverse.hpp>
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
    // shift_right
    namespace detail {
        template <typename ExPolicy, typename FwdIter, typename Sent>
        hpx::future<FwdIter> shift_right_helper(
            ExPolicy policy, FwdIter first, Sent last, FwdIter new_first)
        {
            using non_seq = std::false_type;

            auto p = hpx::execution::parallel_task_policy()
                         .on(policy.executor())
                         .with(policy.parameters());

            detail::reverse<FwdIter> r;
            return dataflow(
                [=](hpx::future<FwdIter>&& f1) mutable -> hpx::future<FwdIter> {
                    // propagate exceptions
                    f1.get();

                    hpx::future<FwdIter> f = r.call2(p, non_seq(), first, last);
                    return f.then(
                        [=](hpx::future<FwdIter>&& f) mutable -> FwdIter {
                            f.get();    // propagate exceptions
                            return new_first;
                        });
                },
                r.call2(p, non_seq(), first, new_first));
        }

        /* Sequential shift_right implementation borrowed
        from https://github.com/danra/shift_proposal */

        template <class I>
        using difference_type_t =
            typename std::iterator_traits<I>::difference_type;

        template <class I>
        using iterator_category_t =
            typename std::iterator_traits<I>::iterator_category;

        template <class I, class Tag, class = void>
        constexpr bool is_category = false;
        template <class I, class Tag>
        constexpr bool is_category<I, Tag,
            std::enable_if_t<
                std::is_convertible_v<iterator_category_t<I>, Tag>>> = true;

        template <class FwdIter>
        constexpr difference_type_t<FwdIter> bounded_advance_r(
            FwdIter& i, difference_type_t<FwdIter> n, FwdIter const bound)
        {
            if constexpr (is_category<FwdIter, std::bidirectional_iterator_tag>)
            {
                for (; n < 0 && i != bound; ++n, void(--i))
                {
                    ;
                }
            }

            for (; n > 0 && i != bound; --n, void(++i))
            {
                ;
            }

            return n;
        }

        template <typename FwdIter>
        FwdIter sequential_shift_right(
            FwdIter first, FwdIter last, difference_type_t<FwdIter> n)
        {
            if (n <= 0)
            {
                return first;
            }

            if constexpr (is_category<FwdIter, std::bidirectional_iterator_tag>)
            {
                auto mid = last;
                if (bounded_advance_r(mid, -n, first))
                {
                    return last;
                }
                return std::move_backward(
                    std::move(first), std::move(mid), std::move(last));
            }
            else
            {
                auto result = first;
                if (bounded_advance_r(result, n, last))
                {
                    return last;
                }

                auto lead = result;
                auto trail = first;

                for (; trail != result; ++lead, void(++trail))
                {
                    if (lead == last)
                    {
                        util::move(std::move(first), std::move(trail),
                            std::move(result));
                        return result;
                    }
                }

                for (;;)
                {
                    for (auto mid = first; mid != result;
                         ++lead, void(++trail), ++mid)
                    {
                        if (lead == last)
                        {
                            trail = util::move(mid, result, std::move(trail));
                            util::move(std::move(first), std::move(mid),
                                std::move(trail));
                            return result;
                        }
                        std::iter_swap(mid, trail);
                    }
                }
            }
        }

        template <typename FwdIter2>
        struct shift_right
          : public detail::algorithm<shift_right<FwdIter2>, FwdIter2>
        {
            shift_right()
              : shift_right::algorithm("shift_right")
            {
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename Size>
            static FwdIter sequential(
                ExPolicy, FwdIter first, Sent last, Size n)
            {
                if (n <= 0 || n >= detail::distance(first, last))
                {
                    return first;
                }

                auto last_iter = detail::advance_to_sentinel(first, last);
                return detail::sequential_shift_right(
                    first, last_iter, difference_type_t<FwdIter>(n));
            }

            template <typename ExPolicy, typename Sent, typename Size>
            static typename util::detail::algorithm_result<ExPolicy,
                FwdIter2>::type
            parallel(ExPolicy&& policy, FwdIter2 first, Sent last, Size n)
            {
                auto dist = detail::distance(first, last);
                if (n <= 0 || n >= dist)
                {
                    return parallel::util::detail::algorithm_result<ExPolicy,
                        FwdIter2>::get(std::move(first));
                }

                auto new_first = std::next(first, dist - n);
                return util::detail::algorithm_result<ExPolicy, FwdIter2>::get(
                    shift_right_helper(policy, first, last, new_first));
            }
        };
        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::shift_right
    HPX_INLINE_CONSTEXPR_VARIABLE struct shift_right_t final
      : hpx::functional::tag_fallback<shift_right_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter>::value)>
        // clang-format on
        friend FwdIter tag_fallback_dispatch(
            shift_right_t, FwdIter first, FwdIter last, Size n)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::shift_right<FwdIter>().call(
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
        tag_fallback_dispatch(shift_right_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, Size n)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::shift_right<FwdIter>().call(
                std::forward<ExPolicy>(policy), first, last, n);
        }
    } shift_right{};
}    // namespace hpx
