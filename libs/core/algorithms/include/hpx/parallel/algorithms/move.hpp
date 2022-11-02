//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2016-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/move.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Moves the elements in the range [first, last), to another range
    /// beginning at \a dest. After this operation the elements in the
    /// moved-from range will still contain valid values of the appropriate
    /// type, but not necessarily the same values as before the move.
    /// Executed according to the policy.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first move assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the move assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The move assignments in the parallel \a move algorithm invoked
    /// with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in
    /// the calling thread.
    ///
    /// The move assignments in the parallel \a move algorithm invoked
    /// with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a move algorithm returns a
    ///           \a  hpx::future<FwdIter2>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a move algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           moved.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    move(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest);

    /// Moves the elements in the range [first, last), to another range
    /// beginning at \a dest. After this operation the elements in the
    /// moved-from range will still contain valid values of the appropriate
    /// type, but not necessarily the same values as before the move.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first move assignments.
    ///
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// \returns  The \a move algorithm returns a \a FwdIter2.
    ///           The \a move algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           moved.
    ///
    template <typename FwdIter1, typename FwdIter2>
    FwdIter2 move(FwdIter1 first, FwdIter1 last, FwdIter2 dest);

    // clang-format off
}   // namespace hpx

#else // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/transfer.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/transfer.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // move
    namespace detail {
        /// \cond NOINTERNAL

        template <typename IterPair>
        struct move_pair
          : public detail::algorithm<detail::move_pair<IterPair>, IterPair>
        {
            move_pair()
              : move_pair::algorithm("move")
            {
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename OutIter>
            static constexpr std::enable_if_t<
                !hpx::traits::is_random_access_iterator_v<InIter>,
                util::in_out_result<InIter, OutIter>>
            sequential(ExPolicy, InIter first, Sent last, OutIter dest)
            {
                return util::move(first, last, dest);
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename OutIter>
            static constexpr std::enable_if_t<
                hpx::traits::is_random_access_iterator_v<InIter>,
                util::in_out_result<InIter, OutIter>>
            sequential(ExPolicy, InIter first, Sent last, OutIter dest)
            {
                return util::move_n(first, detail::distance(first, last), dest);
            }

            template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<FwdIter1, FwdIter2>>::type
            parallel(
                ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest)
            {
                using zip_iterator =
                    hpx::util::zip_iterator<FwdIter1, FwdIter2>;

                return util::detail::get_in_out_result(
                    util::foreach_partitioner<ExPolicy>::call(
                        HPX_FORWARD(ExPolicy, policy),
                        zip_iterator(first, dest),
                        detail::distance(first, last),
                        [](zip_iterator part_begin, std::size_t part_size,
                            std::size_t) {
                            auto iters = part_begin.get_iterator_tuple();
                            util::move_n(hpx::get<0>(iters), part_size,
                                hpx::get<1>(iters));
                        },
                        [](zip_iterator&& last) -> zip_iterator {
                            return HPX_MOVE(last);
                        }));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter1, typename FwdIter2, typename Enable = void>
        struct move;

        template <typename FwdIter1, typename FwdIter2>
        struct move<FwdIter1, FwdIter2,
            typename std::enable_if<
                iterators_are_segmented<FwdIter1, FwdIter2>::value>::type>
          : public move_pair<util::in_out_result<
                typename hpx::traits::segmented_iterator_traits<
                    FwdIter1>::local_iterator,
                typename hpx::traits::segmented_iterator_traits<
                    FwdIter2>::local_iterator>>
        {
        };

        template <typename FwdIter1, typename FwdIter2>
        struct move<FwdIter1, FwdIter2,
            typename std::enable_if<
                iterators_are_not_segmented<FwdIter1, FwdIter2>::value>::type>
          : public move_pair<util::in_out_result<FwdIter1, FwdIter2>>
        {
        };
        /// \endcond
    }    // namespace detail
}}}    // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::move
    inline constexpr struct move_t final
      : hpx::detail::tag_parallel_algorithm<move_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value)>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_invoke(move_t, ExPolicy&& policy, FwdIter1 first,
            FwdIter1 last, FwdIter2 dest)
        {
            return hpx::parallel::util::get_second_element(
                hpx::parallel::v1::detail::transfer<
                    hpx::parallel::v1::detail::move<FwdIter1, FwdIter2>>(
                    HPX_FORWARD(ExPolicy, policy), first, last, dest));
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value)>
        // clang-format on
        friend FwdIter2 tag_fallback_invoke(
            move_t, FwdIter1 first, FwdIter1 last, FwdIter2 dest)
        {
            return std::move(first, last, dest);
        }
    } move{};
}    // namespace hpx

#endif    // DOXYGEN
