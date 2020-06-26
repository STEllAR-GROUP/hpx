//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/for_each.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {
    /// Applies \a f to the result of dereferencing every iterator in the
    /// range [first, last).
    ///
    /// \note   Complexity: Applies \a f exactly \a last - \a first times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// \tparam InIter      The type of the source begin iterator used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns            {last, std::move(f)} where last is the iterator
    ///                     corresponding to the input sentinel last.
    ///
    template <typename InIter, typename Sent, typename F,
        typename Proj = util::projection_identity>
    hpx::ranges::for_each_result<InIter, F> for_each(
        InIter first, Sent last, F&& f, Proj&& proj = Proj());

    /// Applies \a f to the result of dereferencing every iterator in the
    /// range [first, last).
    ///
    /// \note   Complexity: Applies \a f exactly \a last - \a first times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIter     The type of the source begin iterator used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a for_each algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a FwdIter
    ///           otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent, typename F,
        typename Proj = util::projection_identity>
    FwdIter for_each(ExPolicy&& policy, FwdIter first, Sent last, F&& f,
        Proj&& proj = Proj());

    /// Applies \a f to the result of dereferencing every iterator in the
    /// given range \a rng.
    ///
    /// \note   Complexity: Applies \a f exactly \a size(rng) times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns            {std::end(rng), std::move(f)}
    ///
    template <typename Rng, typename F,
        typename Proj = util::projection_identity>
    hpx::ranges::for_each_result<
        typename hpx::traits::range_iterator<Rng>::type, F>
    for_each(ExPolicy&& policy, Rng&& rng, F&& f, Proj&& proj = Proj());

    /// Applies \a f to the result of dereferencing every iterator in the
    /// given range \a rng.
    ///
    /// \note   Complexity: Applies \a f exactly \a size(rng) times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a for_each algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a FwdIter
    ///           otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Rng, typename F,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
    for_each(ExPolicy&& policy, Rng&& rng, F&& f, Proj&& proj = Proj());

    /////////////////////////////////////////////////////
    /// Applies \a f to the result of dereferencing every iterator in the range
    /// [first, first + count), starting from first and proceeding to
    /// first + count - 1.
    ///
    /// \note   Complexity: Applies \a f exactly \a last - \a first times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// \tparam InIter      The type of the source begin iterator used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns            {first + count, std::move(f)}
    ///
    template <typename InIter, typename Sent, typename F,
        typename Proj = util::projection_identity>
    hpx::ranges::for_each_result<InIter, F> for_each(
        InIter first, Sent last, F&& f, Proj&& proj = Proj());

    /// Applies \a f to the result of dereferencing every iterator in the range
    /// [first, first + count), starting from first and proceeding to
    /// first + count - 1.
    ///
    /// \note   Complexity: Applies \a f exactly \a count times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIter     The type of the source begin iterator used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// \returns  The \a for_each algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a FwdIter
    ///           otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size, typename F,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type for_each_n(
        ExPolicy&& policy, FwdIter first, Size count, F&& f,
        Proj&& proj = Proj());
}}    // namespace hpx::ranges
#else
#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    // clang-format off
    template <typename ExPolicy, typename Rng, typename F,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value &&
            hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
            hpx::parallel::traits::is_indirect_callable<ExPolicy, F,
                hpx::parallel::traits::projected_range<Proj, Rng>>::value)>
    // clang-format on
    typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
        HPX_DEPRECATED_V(1, 6,
            "hpx::parallel::for_each is deprecated, use "
            "hpx::ranges::for_each instead")
            for_each(ExPolicy&& policy, Rng&& rng, F&& f, Proj&& proj = Proj())
    {
        return for_each(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
            hpx::util::end(rng), std::forward<F>(f), std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1

namespace hpx { namespace ranges {
    template <typename I, typename F>
    using for_each_result = in_fun_result<I, F>;

    template <typename I, typename F>
    using for_each_n_result = in_fun_result<I, F>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::for_each
    HPX_INLINE_CONSTEXPR_VARIABLE struct for_each_t final
      : hpx::functional::tag<for_each_t>
    {
        // clang-format off
        template <typename InIter, typename Sent, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_input_iterator<InIter>::value &&
                hpx::traits::is_sentinel_for<Sent, InIter>::value &&
                hpx::parallel::traits::is_projected<Proj, InIter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::parallel::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected<Proj, InIter>>::value)>
        // clang-format on
        friend for_each_result<InIter, F> tag_invoke(hpx::ranges::for_each_t,
            InIter first, Sent last, F&& f, Proj&& proj = Proj())
        {
            using is_segmented = hpx::traits::is_segmented_iterator<InIter>;

            auto it =
                parallel::v1::detail::for_each_(hpx::parallel::execution::seq,
                    first, last, f, std::forward<Proj>(proj), is_segmented());
            return {std::move(it), std::forward<F>(f)};
        }

        // clang-format off
        template <typename Rng, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::parallel::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>>::value)>
        // clang-format on
        friend for_each_result<typename hpx::traits::range_iterator<Rng>::type,
            F>
        tag_invoke(
            hpx::ranges::for_each_t, Rng&& rng, F&& f, Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;
            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            auto it =
                parallel::v1::detail::for_each_(hpx::parallel::execution::seq,
                    hpx::util::begin(rng), hpx::util::end(rng), f,
                    std::forward<Proj>(proj), is_segmented());
            return {std::move(it), std::forward<F>(f)};
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, F,
                    hpx::parallel::traits::projected<Proj, FwdIter>>::value)>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_invoke(hpx::ranges::for_each_t, ExPolicy&& policy, FwdIter first,
            Sent last, F&& f, Proj&& proj = Proj())
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return parallel::v1::detail::for_each_(
                std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
                std::forward<Proj>(proj), is_segmented());
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, F,
                    hpx::parallel::traits::projected_range<Proj, Rng>>::value)>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng>::type>::type
        tag_invoke(hpx::ranges::for_each_t, ExPolicy&& policy, Rng&& rng, F&& f,
            Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;
            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            return parallel::v1::detail::for_each_(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), std::forward<F>(f),
                std::forward<Proj>(proj), is_segmented());
        }
    } for_each{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::for_each_n
    HPX_INLINE_CONSTEXPR_VARIABLE struct for_each_n_t final
      : hpx::functional::tag<for_each_n_t>
    {
        // clang-format off
        template <typename InIter, typename Size, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_input_iterator<InIter>::value &&
                hpx::parallel::traits::is_projected<Proj, InIter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::parallel::execution::sequenced_policy, F,
                    hpx::parallel::traits::projected<Proj, InIter>>::value)>
        // clang-format on
        friend for_each_n_result<InIter, F> tag_invoke(
            hpx::ranges::for_each_n_t, InIter first, Size count, F&& f,
            Proj&& proj = Proj())
        {
            // if count is representing a negative value, we do nothing
            if (parallel::v1::detail::is_negative(count))
            {
                return {std::move(first), std::forward<F>(f)};
            }

            auto it = for_each_n(hpx::parallel::execution::seq, first, count, f,
                std::forward<Proj>(proj));
            return {std::move(it), std::forward<F>(f)};
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, F,
                    hpx::parallel::traits::projected<Proj, FwdIter>>::value)>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_invoke(hpx::ranges::for_each_n_t, ExPolicy&& policy, FwdIter first,
            Size count, F&& f, Proj&& proj = Proj())
        {
            // if count is representing a negative value, we do nothing
            if (parallel::v1::detail::is_negative(count))
            {
                return parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(std::move(first));
            }

            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return parallel::v1::detail::for_each_n_(
                std::forward<ExPolicy>(policy), first, count,
                std::forward<F>(f), std::forward<Proj>(proj), is_segmented());
        }
    } for_each_n{};
}}    // namespace hpx::ranges
#endif
