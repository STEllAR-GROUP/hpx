//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/swap_ranges.hpp

#if defined(DOXYGEN)

namespace hpx {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Exchanges elements between range [first1, last1) and another range
    /// starting at \a first2.
    ///
    /// \note   Complexity: Linear in the distance between \a first1 and \a
    ///  last1
    ///
    /// \tparam FwdIter1    The type of the first range of iterators to swap
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the second range of iterators to swap
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second  sequence of
    ///                     elements the algorithm will be applied to.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked without an execution policy object  execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a swap_ranges algorithm returns \a FwdIter2.
    ///           The \a swap_ranges algorithm returns iterator to the element
    ///           past the last element exchanged in the range beginning with
    ///           \a first2.
    ///
    template <typename FwdIter1, typename FwdIter2>
    FwdIter2 swap_ranges(FwdIter1 first1, FwdIter1 last1, FwdIter2 first2);

    ///////////////////////////////////////////////////////////////////////////
    /// Exchanges elements between range [first1, last1) and another range
    /// starting at \a first2.
    ///
    /// \note   Complexity: Linear in the distance between \a first1 and \a last1
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the swap operations.
    /// \tparam FwdIter1    The type of the first range of iterators to swap
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the second range of iterators to swap
    ///                     (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second  sequence of
    ///                     elements the algorithm will be applied to.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in
    /// the calling thread.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a swap_ranges algorithm returns a
    ///           \a hpx::future<FwdIter2>  if the execution policy is of
    ///           type \a parallel_task_policy and returns \a FwdIter2
    ///           otherwise.
    ///           The \a swap_ranges algorithm returns iterator to the element
    ///           past the last element exchanged in the range beginning with
    ///           \a first2.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    swap_ranges(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
        FwdIter2 first2);

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    template <typename Iter1, typename Iter2>
    using swap_ranges_result = hpx::parallel::util::in_in_result<Iter1, Iter2>;

    ///////////////////////////////////////////////////////////////////////////
    // swap ranges
    namespace detail {
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Size>
        typename util::detail::algorithm_result<ExPolicy,
            swap_ranges_result<FwdIter1, FwdIter2>>::type
        parallel_swap_ranges(
            ExPolicy&& policy, FwdIter1 first1, FwdIter2 first2, Size n)
        {
            using zip_iterator = hpx::util::zip_iterator<FwdIter1, FwdIter2>;
            using reference = typename zip_iterator::reference;

            return get_iter_in_in_result(for_each_n<zip_iterator>().call(
                std::forward<ExPolicy>(policy),
                hpx::util::make_zip_iterator(first1, first2), n,
                [](reference t) -> void {
                    using hpx::get;
                    std::swap(get<0>(t), get<1>(t));
                },
                util::projection_identity()));
        }

        template <typename IterPair>
        struct swap_ranges
          : public detail::algorithm<swap_ranges<IterPair>, IterPair>
        {
            swap_ranges()
              : swap_ranges::algorithm("swap_ranges")
            {
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent,
                typename FwdIter2>
            static FwdIter2 sequential(
                ExPolicy, FwdIter1 first1, Sent last1, FwdIter2 first2)
            {
                while (first1 != last1)
                {
#if defined(HPX_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                    std::ranges::iter_swap(first1++, first2++);
#else
                    std::iter_swap(first1++, first2++);
#endif
                }
                return first2;
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent1,
                typename FwdIter2, typename Sent2>
            static swap_ranges_result<FwdIter1, FwdIter2> sequential(ExPolicy,
                FwdIter1 first1, Sent1 last1, FwdIter2 first2, Sent2 last2)
            {
                while (first1 != last1 && first2 != last2)
                {
                    std::swap(*first1, *first2);
                    first1++;
                    first2++;
                }
                return util::in_in_result<FwdIter1, FwdIter2>{first1, first2};
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent,
                typename FwdIter2>
            static typename util::detail::algorithm_result<ExPolicy,
                FwdIter2>::type
            parallel(
                ExPolicy&& policy, FwdIter1 first1, Sent last1, FwdIter2 first2)
            {
                return util::get_in2_element(
                    parallel_swap_ranges(std::forward<ExPolicy>(policy), first1,
                        first2, detail::distance(first1, last1)));
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent1,
                typename FwdIter2, typename Sent2>
            static typename util::detail::algorithm_result<ExPolicy,
                swap_ranges_result<FwdIter1, FwdIter2>>::type
            parallel(ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
                FwdIter2 first2, Sent2 last2)
            {
                auto dist1 = detail::distance(first1, last1);
                auto dist2 = detail::distance(first2, last2);
                return parallel_swap_ranges(std::forward<ExPolicy>(policy),
                    first1, first2, dist1 < dist2 ? dist1 : dist2);
            }
        };
        /// \endcond
    }    // namespace detail

    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::transform_exclusive_scan is deprecated, use "
        "hpx::transform_exclusive_scan instead")
    inline typename std::enable_if<hpx::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type>::type
        swap_ranges(
            ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1, FwdIter2 first2)
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::swap_ranges<FwdIter2>().call(
            std::forward<ExPolicy>(policy), first1, last1, first2);
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::swap_ranges
    HPX_INLINE_CONSTEXPR_VARIABLE struct swap_ranges_t final
      : hpx::functional::tag_fallback<swap_ranges_t>
    {
        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter1>::value &&
                hpx::traits::is_forward_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend FwdIter2 tag_fallback_dispatch(hpx::swap_ranges_t,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2)
        {
            static_assert(hpx::traits::is_input_iterator<FwdIter1>::value,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_output_iterator<FwdIter2>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::swap_ranges<FwdIter2>().call(
                hpx::execution::seq, first1, last1, first2);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter1>::value &&
                hpx::traits::is_forward_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_dispatch(hpx::swap_ranges_t, ExPolicy&& policy,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2)
        {
            static_assert(hpx::traits::is_input_iterator<FwdIter1>::value,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_output_iterator<FwdIter2>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::swap_ranges<FwdIter2>().call(
                std::forward<ExPolicy>(policy), first1, last1, first2);
        }
    } swap_ranges{};
}    // namespace hpx

#endif    // DOXYGEN
