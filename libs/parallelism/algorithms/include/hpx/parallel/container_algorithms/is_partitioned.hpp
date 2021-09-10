//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2015 Daniel Bourgeois
//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    namespace ranges {
        /// Determines if the range [first, last) is partitioned.
        ///
        /// \note   Complexity: at most (N) predicate evaluations where
        ///         \a N = distance(first, last).
        ///
        /// \tparam FwdIter     The type of the source iterators used for the
        ///                     This iterator type must meet the requirements of a
        ///                     forward iterator.
        /// \tparam Sent        The type of the source sentinel (deduced). This
        ///                     sentinel type must be a sentinel for FwdIter.
        /// \tparam Proj        The type of an optional projection function. This
        ///                     defaults to \a hpx::parallel::util::projection_identity.
        /// \tparam Pred        The type of the function/function object to use
        ///                     (deduced).
        /// \param first        Refers to the beginning of the sequence of elements
        ///                     of that the algorithm will be applied to.
        /// \param last         Refers to the end of the sequence of elements of
        ///                     that the algorithm will be applied to.
        /// \param pred         Refers to the unary predicate which returns true
        ///                     for elements expected to be found in the beginning
        ///                     of the range. The signature of the function
        ///                     should be equivalent to
        ///                     \code
        ///                     bool pred(const Type &a);
        ///                     \endcode \n
        ///                     The signature does not need to have const &, but
        ///                     the function must not modify the objects passed to
        ///                     it. The type \a Type must be such that objects of
        ///                     types \a FwdIter can be dereferenced and then
        ///                     implicitly converted to Type.
        /// \param proj         Specifies the function (or function object) which
        ///                     will be invoked for each of the elements as a
        ///                     projection operation before the actual predicate
        ///                     \a is invoked.
        ///
        /// \returns  The \a is_partitioned algorithm returns \a bool.
        ///           The \a is_partitioned algorithm returns true if each element
        ///           in the sequence for which pred returns true precedes those for
        ///           which pred returns false. Otherwise is_partitioned returns
        ///           false. If the range [first, last) contains less than two
        ///           elements, the function is always true.
        ///
        template <typename FwdIter, typename Sent, typename Pred,
            typename Proj = hpx::parallel::util::projection_identity>
        bool is_partitioned(
            FwdIter first, Sent last, Pred&& pred, Proj&& proj = Proj());

        /// Determines if the range [first, last) is partitioned.
        ///
        /// \note   Complexity: at most (N) predicate evaluations where
        ///         \a N = distance(first, last).
        ///
        /// \tparam ExPolicy    The type of the execution policy to use (deduced).
        ///                     It describes the manner in which the execution
        ///                     of the algorithm may be parallelized and the manner
        ///                     in which it executes the assignments.
        /// \tparam FwdIter     The type of the source iterators used for the
        ///                     This iterator type must meet the requirements of a
        ///                     forward iterator.
        /// \tparam Sent        The type of the source sentinel (deduced). This
        ///                     sentinel type must be a sentinel for FwdIter.
        /// \tparam Proj        The type of an optional projection function. This
        ///                     defaults to \a hpx::parallel::util::projection_identity.
        /// \tparam Pred        The type of the function/function object to use
        ///                     (deduced). \a Pred must be \a CopyConstructible
        ///                     when using a parallel policy.
        /// \param policy       The execution policy to use for the scheduling of
        ///                     the iterations.
        /// \param first        Refers to the beginning of the sequence of elements
        ///                     of that the algorithm will be applied to.
        /// \param last         Refers to the end of the sequence of elements of
        ///                     that the algorithm will be applied to.
        /// \param pred         Refers to the unary predicate which returns true
        ///                     for elements expected to be found in the beginning
        ///                     of the range. The signature of the function
        ///                     should be equivalent to
        ///                     \code
        ///                     bool pred(const Type &a);
        ///                     \endcode \n
        ///                     The signature does not need to have const &, but
        ///                     the function must not modify the objects passed to
        ///                     it. The type \a Type must be such that objects of
        ///                     types \a FwdIter can be dereferenced and then
        ///                     implicitly converted to Type.
        /// \param proj         Specifies the function (or function object) which
        ///                     will be invoked for each of the elements as a
        ///                     projection operation before the actual predicate
        ///                     \a is invoked.
        ///
        /// The predicate operations in the parallel \a is_partitioned algorithm invoked
        /// with an execution policy object of type \a sequenced_policy
        /// executes in sequential order in the calling thread.
        ///
        /// The comparison operations in the parallel \a is_partitioned algorithm invoked
        /// with an execution policy object of type \a parallel_policy
        /// or \a parallel_task_policy are permitted to execute in an unordered
        /// fashion in unspecified threads, and indeterminately sequenced
        /// within each thread.
        ///
        /// \returns  The \a is_partitioned algorithm returns a \a hpx::future<bool>
        ///           if the execution policy is of type \a task_execution_policy
        ///           and returns \a bool otherwise.
        ///           The \a is_partitioned algorithm returns true if each element
        ///           in the sequence for which pred returns true precedes those for
        ///           which pred returns false. Otherwise is_partitioned returns
        ///           false. If the range [first, last) contains less than two
        ///           elements, the function is always true.
        ///
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Pred,
            typename Proj = hpx::parallel::util::projection_identity>
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        is_partitioned(ExPolicy&& policy, FwdIter first, Sent last, Pred&& pred,
            Proj&& proj = Proj());

        /// Determines if the range rng is partitioned.
        ///
        /// \note   Complexity: at most (N) predicate evaluations where
        ///         \a N = std::size(rng).
        ///
        /// \tparam Rng         The type of the source range used (deduced).
        ///                     The iterators extracted from this range type must
        ///                     meet the requirements of an forward iterator.
        /// \tparam Proj        The type of an optional projection function. This
        ///                     defaults to \a hpx::parallel::util::projection_identity.
        /// \tparam Pred        The type of the function/function object to use
        ///                     (deduced).
        /// \param rng          Refers to the sequence of elements the algorithm
        ///                     will be applied to.
        /// \param pred         Refers to the unary predicate which returns true
        ///                     for elements expected to be found in the beginning
        ///                     of the range. The signature of the function
        ///                     should be equivalent to
        ///                     \code
        ///                     bool pred(const Type &a);
        ///                     \endcode \n
        ///                     The signature does not need to have const &, but
        ///                     the function must not modify the objects passed to
        ///                     it. The type \a Type must be such that objects of
        ///                     types \a FwdIter can be dereferenced and then
        ///                     implicitly converted to Type.
        /// \param proj         Specifies the function (or function object) which
        ///                     will be invoked for each of the elements as a
        ///                     projection operation before the actual predicate
        ///                     \a is invoked.
        ///
        /// \returns  The \a is_partitioned algorithm returns \a bool.
        ///           The \a is_partitioned algorithm returns true if each element
        ///           in the sequence for which pred returns true precedes those for
        ///           which pred returns false. Otherwise is_partitioned returns
        ///           false. If the range rng contains less than two
        ///           elements, the function is always true.
        ///
        template <typename Rng, typename Pred,
            typename Proj = hpx::parallel::util::projection_identity>
        bool is_partitioned(Rng&& rng, Pred&& pred, Proj&& proj = Proj());

        /// Determines if the range [first, last) is partitioned.
        ///
        /// \note   Complexity: at most (N) predicate evaluations where
        ///         \a N = std::size(rng).
        ///
        /// \tparam ExPolicy    The type of the execution policy to use (deduced).
        ///                     It describes the manner in which the execution
        ///                     of the algorithm may be parallelized and the manner
        ///                     in which it executes the assignments.
        /// \tparam Rng         The type of the source range used (deduced).
        ///                     The iterators extracted from this range type must
        ///                     meet the requirements of an forward iterator.
        /// \tparam Proj        The type of an optional projection function. This
        ///                     defaults to \a hpx::parallel::util::projection_identity.
        /// \tparam Pred        The type of the function/function object to use
        ///                     (deduced). \a Pred must be \a CopyConstructible
        ///                     when using a parallel policy.
        /// \param policy       The execution policy to use for the scheduling of
        ///                     the iterations.
        /// \param rng          Refers to the sequence of elements the algorithm
        ///                     will be applied to.
        /// \param pred         Refers to the unary predicate which returns true
        ///                     for elements expected to be found in the beginning
        ///                     of the range. The signature of the function
        ///                     should be equivalent to
        ///                     \code
        ///                     bool pred(const Type &a);
        ///                     \endcode \n
        ///                     The signature does not need to have const &, but
        ///                     the function must not modify the objects passed to
        ///                     it. The type \a Type must be such that objects of
        ///                     types \a FwdIter can be dereferenced and then
        ///                     implicitly converted to Type.
        /// \param proj         Specifies the function (or function object) which
        ///                     will be invoked for each of the elements as a
        ///                     projection operation before the actual predicate
        ///                     \a is invoked.
        ///
        /// The predicate operations in the parallel \a is_partitioned algorithm invoked
        /// with an execution policy object of type \a sequenced_policy
        /// executes in sequential order in the calling thread.
        ///
        /// The comparison operations in the parallel \a is_partitioned algorithm invoked
        /// with an execution policy object of type \a parallel_policy
        /// or \a parallel_task_policy are permitted to execute in an unordered
        /// fashion in unspecified threads, and indeterminately sequenced
        /// within each thread.
        ///
        /// \returns  The \a is_partitioned algorithm returns a \a hpx::future<bool>
        ///           if the execution policy is of type \a task_execution_policy
        ///           and returns \a bool otherwise.
        ///           The \a is_partitioned algorithm returns true if each element
        ///           in the sequence for which pred returns true precedes those for
        ///           which pred returns false. Otherwise is_partitioned returns
        ///           false. If the range rng contains less than two
        ///           elements, the function is always true.
        ///
        template <typename ExPolicy, typename Rng, typename Pred,
            typename Proj = hpx::parallel::util::projection_identity>
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        is_partitioned(
            ExPolicy&& policy, Rng&& rng, Pred&& pred, Proj&& proj = Proj());
    }    // namespace ranges
#else

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/parallel/algorithms/is_partitioned.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace ranges {
    HPX_INLINE_CONSTEXPR_VARIABLE struct is_partitioned_t final
      : hpx::detail::tag_parallel_algorithm<is_partitioned_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent,
            typename Pred,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj, FwdIter>>::value
            )>
        // clang-format on
        friend bool tag_fallback_dispatch(hpx::ranges::is_partitioned_t,
            FwdIter first, Sent last, Pred&& pred, Proj&& proj = Proj())
        {
            return hpx::parallel::v1::detail::is_partitioned<FwdIter, Sent>()
                .call(hpx::execution::seq, first, last,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Pred,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj, FwdIter>>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_dispatch(hpx::ranges::is_partitioned_t, ExPolicy&& policy,
            FwdIter first, Sent last, Pred&& pred, Proj&& proj = Proj())
        {
            return hpx::parallel::v1::detail::is_partitioned<FwdIter, Sent>()
                .call(std::forward<ExPolicy>(policy), first, last,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng,
            typename Pred,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>>::value
            )>
        // clang-format on
        friend bool tag_fallback_dispatch(hpx::ranges::is_partitioned_t,
            Rng&& rng, Pred&& pred, Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            return hpx::parallel::v1::detail::is_partitioned<iterator_type,
                iterator_type>()
                .call(hpx::execution::seq, std::begin(rng), std::end(rng),
                    std::forward<Pred>(pred), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Pred,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_dispatch(hpx::ranges::is_partitioned_t, ExPolicy&& policy,
            Rng&& rng, Pred&& pred, Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            return hpx::parallel::v1::detail::is_partitioned<iterator_type,
                iterator_type>()
                .call(std::forward<ExPolicy>(policy), std::begin(rng),
                    std::end(rng), std::forward<Pred>(pred),
                    std::forward<Proj>(proj));
        }
    } is_partitioned{};
}}    // namespace hpx::ranges

#endif
