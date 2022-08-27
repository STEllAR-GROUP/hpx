//  Copyright (c) 2015 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/transform.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {

    /// Applies the given function \a f to the given range \a rng and stores
    /// the result in another range, beginning at dest.
    ///
    /// \note   Complexity: Exactly size(rng) applications of \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The type \a Type must be such that an object of
    ///                     type \a range_iterator<Rng>::type can be dereferenced and then
    ///                     implicitly converted to \a Type. The type \a Ret
    ///                     must be such that an object of type \a OutIter can
    ///                     be dereferenced and assigned a value of type
    ///                     \a Ret.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform algorithm returns a
    ///           \a hpx::future<ranges::unary_transform_result<range_iterator<Rng>::type, OutIter> >
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns \a ranges::unary_transform_result<range_iterator<Rng>::type, OutIter>
    ///           otherwise.
    ///           The \a transform algorithm returns a tuple holding an iterator
    ///           referring to the first element after the input sequence and
    ///           the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename Rng, typename OutIter, typename F,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        ranges::unary_transform_result<
            typename hpx::traits::range_iterator<Rng>::type, OutIter>>::type
    transform(ExPolicy&& policy, Rng&& rng, OutIter dest, F&& f,
        Proj&& proj = Proj());

    /// Applies the given function \a f to the given range \a rng and stores
    /// the result in another range, beginning at dest.
    ///
    /// \note   Complexity: Exactly size(rng) applications of \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam FwdIter1    The type of the source iterators for the first
    ///                     range used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for FwdIter1.
    /// \tparam FwdIter2    The type of the source iterators for the first
    ///                     range used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to \a Type. The type \a Ret
    ///                     must be such that an object of type \a FwdIter2 can
    ///                     be dereferenced and assigned a value of type
    ///                     \a Ret.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform algorithm returns a
    ///           \a hpx::future<ranges::unary_transform_result<FwdIter1, FwdIter2> >
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns \a ranges::unary_transform_result<FwdIter1, FwdIter2>
    ///           otherwise.
    ///           The \a transform algorithm returns a tuple holding an iterator
    ///           referring to the first element after the input sequence and
    ///           the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter2, typename F,
        typename Proj = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        ranges::unary_transform_result<FwdIter1, FwdIter2>>::type
    transform(ExPolicy&& policy, FwdIter1 first, Sent1 last, FwdIter2 dest,
        F&& f, Proj&& proj = Proj());

    /// Applies the given function \a f to pairs of elements from two ranges:
    /// one defined by \a rng and the other beginning at first2, and
    /// stores the result in another range, beginning at dest.
    ///
    /// \note   Complexity: Exactly size(rng) applications of \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam FwdIter1    The type of the source iterators for the first
    ///                     range used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for FwdIter1.
    /// \tparam FwdIter2    The type of the source iterators for the first
    ///                     range used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for FwdIter2.
    /// \tparam FwdIter3    The type of the source iterators for the first
    ///                     range used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj1       The type of an optional projection function to be
    ///                     used for elements of the first sequence. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function to be
    ///                     used for elements of the second sequence. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the first sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last1        Refers to the end of the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the algorithm will be applied to.
    /// \param last2        Refers to the end of the second sequence of elements
    ///                     the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The types \a Type1 and \a Type2 must be such that
    ///                     objects of types FwdIter1 and FwdIter2 can be
    ///                     dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively. The type \a Ret
    ///                     must be such that an object of type \a FwdIter3 can
    ///                     be dereferenced and assigned a value of type
    ///                     \a Ret.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first sequence as a projection operation before the
    ///                     actual predicate \a f is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second sequence as a projection operation before
    ///                     the actual predicate \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform algorithm returns a
    /// \a hpx::future<ranges::binary_transform_result<FwdIter1, FwdIter2, FwdIter3> >
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns
    /// \a ranges::binary_transform_result<FwdIter1, FwdIter2, FwdIter3>
    ///           otherwise.
    ///           The \a transform algorithm returns a tuple holding an iterator
    ///           referring to the first element after the first input sequence,
    ///           an iterator referring to the first element after the second
    ///           input sequence, and the output iterator referring to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter2, typename Sent2, typename FwdIter3, typename F,
        typename Proj1 = hpx::parallel::util::projection_identity,
        typename Proj2 = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        ranges::binary_transform_result<FwdIter1, FwdIter2, FwdIter3>>::type
    transform(ExPolicy&& policy, FwdIter1 first1, Sent1 last1, FwdIter2 first2,
        Sent2 last2, FwdIter3 dest, F&& f, Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    /// Applies the given function \a f to pairs of elements from two ranges:
    /// one defined by [first1, last1) and the other beginning at first2, and
    /// stores the result in another range, beginning at dest.
    ///
    /// \note   Complexity: Exactly min(last2-first2, last1-first1)
    ///         applications of \a f
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam Rng1        The type of the first source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the second source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj1       The type of an optional projection function to be
    ///                     used for elements of the first sequence. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function to be
    ///                     used for elements of the second sequence. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the first sequence of elements the
    ///                     algorithm will be applied to.
    /// \param rng2         Refers to the second sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The types \a Type1 and \a Type2 must be such that
    ///                     objects of types range_iterator<Rng1>::type and
    ///                     range_iterator<Rng2>::type can be
    ///                     dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively. The type \a Ret
    ///                     must be such that an object of type \a FwdIter can
    ///                     be dereferenced and assigned a value of type
    ///                     \a Ret.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first sequence as a projection operation before the
    ///                     actual predicate \a f is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second sequence as a projection operation before
    ///                     the actual predicate \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The invocations of \a f in the parallel \a transform algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The algorithm will invoke the binary predicate until it reaches
    ///       the end of the shorter of the two given input sequences
    ///
    /// \returns  The \a transform algorithm returns a
    /// \a hpx::future<ranges::binary_transform_result<
    ///           typename hpx::traits::range_iterator<Rng1>::type,
    ///           typename hpx::traits::range_iterator<Rng2>::type,
    ///           FwdIter> >
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns
    /// \a ranges::binary_transform_result<
    ///           typename hpx::traits::range_iterator<Rng1>::type,
    ///           typename hpx::traits::range_iterator<Rng2>::type,
    ///           FwdIter>
    ///           otherwise.
    ///           The \a transform algorithm returns a tuple holding an iterator
    ///           referring to the first element after the first input sequence,
    ///           an iterator referring to the first element after the second
    ///           input sequence, and the output iterator referring to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2, typename FwdIter,
        typename F, typename Proj1 = hpx::parallel::util::projection_identity,
        typename Proj2 = hpx::parallel::util::projection_identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        ranges::binary_transform_result<
            typename hpx::traits::range_iterator<Rng1>::type,
            typename hpx::traits::range_iterator<Rng2>::type, FwdIter>>::type
    tag_fallback_invoke(hpx::ranges::transform_t, ExPolicy&& policy,
        Rng1&& rng1, Rng2&& rng2, FwdIter dest, F&& f, Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2())

}}       // namespace hpx::ranges
#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/transform.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace ranges {

    template <typename I, typename O>
    using unary_transform_result = parallel::util::in_out_result<I, O>;

    template <typename I1, typename I2, typename O>
    using binary_transform_result = parallel::util::in_in_out_result<I1, I2, O>;

    ///////////////////////////////////////////////////////////////////////////
    // a for hpx::ranges::transform
    inline constexpr struct transform_t final
      : hpx::detail::tag_parallel_algorithm<transform_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent1,
            typename FwdIter2, typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            ranges::unary_transform_result<FwdIter1, FwdIter2>>::type
        tag_fallback_invoke(hpx::ranges::transform_t, ExPolicy&& policy,
            FwdIter1 first, Sent1 last, FwdIter2 dest, F&& f,
            Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter1>::value,
                "Requires at least forward iterator.");

            return parallel::v1::detail::transform<
                unary_transform_result<FwdIter1, FwdIter2>>()
                .call(HPX_FORWARD(ExPolicy, policy), first, last, dest,
                    HPX_FORWARD(F, f), HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename FwdIter,
            typename F, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            ranges::unary_transform_result<
                typename hpx::traits::range_iterator<Rng>::type, FwdIter>>::type
        tag_fallback_invoke(hpx::ranges::transform_t, ExPolicy&& policy,
            Rng&& rng, FwdIter dest, F&& f, Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return parallel::v1::detail::transform<
                unary_transform_result<iterator_type, FwdIter>>()
                .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                    hpx::util::end(rng), dest, HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent1,
            typename FwdIter2, typename Sent2, typename FwdIter3, typename F,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_sentinel_for<Sent2, FwdIter2>::value &&
                hpx::traits::is_iterator<FwdIter3>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            ranges::binary_transform_result<FwdIter1, FwdIter2, FwdIter3>>::type
        tag_fallback_invoke(hpx::ranges::transform_t, ExPolicy&& policy,
            FwdIter1 first1, Sent1 last1, FwdIter2 first2, Sent2 last2,
            FwdIter3 dest, F&& f, Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter1>::value &&
                    hpx::traits::is_forward_iterator<FwdIter2>::value,
                "Requires at least forward iterator.");

            return parallel::v1::detail::transform_binary2<
                binary_transform_result<FwdIter1, FwdIter2, FwdIter3>>()
                .call(HPX_FORWARD(ExPolicy, policy), first1, last1, first2,
                    last2, dest, HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                    HPX_FORWARD(Proj2, proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng1, typename Rng2, typename FwdIter,
            typename F, typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            ranges::binary_transform_result<
                typename hpx::traits::range_iterator<Rng1>::type,
                typename hpx::traits::range_iterator<Rng2>::type,
                FwdIter>>::type
        tag_fallback_invoke(hpx::ranges::transform_t, ExPolicy&& policy,
            Rng1&& rng1, Rng2&& rng2, FwdIter dest, F&& f,
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            using iterator_type1 =
                typename hpx::traits::range_traits<Rng1>::iterator_type;
            using iterator_type2 =
                typename hpx::traits::range_traits<Rng2>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type1>::value &&
                    hpx::traits::is_forward_iterator<iterator_type2>::value,
                "Requires at least forward iterator.");

            return parallel::v1::detail::transform_binary2<
                binary_transform_result<iterator_type1, iterator_type2,
                    FwdIter>>()
                .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng1),
                    hpx::util::end(rng1), hpx::util::begin(rng2),
                    hpx::util::end(rng2), dest, HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));
        }

        // clang-format off
        template <typename FwdIter1, typename Sent1, typename FwdIter2,
            typename F,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend ranges::unary_transform_result<FwdIter1, FwdIter2>
        tag_fallback_invoke(hpx::ranges::transform_t, FwdIter1 first,
            Sent1 last, FwdIter2 dest, F&& f, Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_input_iterator<FwdIter1>::value,
                "Requires at least input iterator.");

            return parallel::v1::detail::transform<
                unary_transform_result<FwdIter1, FwdIter2>>()
                .call(hpx::execution::seq, first, last, dest, HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng, typename FwdIter,
            typename F, typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend ranges::unary_transform_result<
            typename hpx::traits::range_iterator<Rng>::type, FwdIter>
        tag_fallback_invoke(hpx::ranges::transform_t, Rng&& rng, FwdIter dest,
            F&& f, Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(hpx::traits::is_input_iterator<iterator_type>::value,
                "Requires at least input iterator.");

            return parallel::v1::detail::transform<
                unary_transform_result<iterator_type, FwdIter>>()
                .call(hpx::execution::seq, hpx::util::begin(rng),
                    hpx::util::end(rng), dest, HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename FwdIter1, typename Sent1,
            typename FwdIter2, typename Sent2, typename FwdIter3, typename F,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_sentinel_for<Sent2, FwdIter2>::value &&
                hpx::traits::is_iterator<FwdIter3>::value
            )>
        // clang-format on
        friend ranges::binary_transform_result<FwdIter1, FwdIter2, FwdIter3>
        tag_fallback_invoke(hpx::ranges::transform_t, FwdIter1 first1,
            Sent1 last1, FwdIter2 first2, Sent2 last2, FwdIter3 dest, F&& f,
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            static_assert(hpx::traits::is_input_iterator<FwdIter1>::value &&
                    hpx::traits::is_input_iterator<FwdIter2>::value,
                "Requires at least input iterator.");

            return parallel::v1::detail::transform_binary2<
                binary_transform_result<FwdIter1, FwdIter2, FwdIter3>>()
                .call(hpx::execution::seq, first1, last1, first2, last2, dest,
                    HPX_FORWARD(F, f), HPX_FORWARD(Proj1, proj1),
                    HPX_FORWARD(Proj2, proj2));
        }

        // clang-format off
        template <typename Rng1, typename Rng2, typename FwdIter,
            typename F, typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend ranges::binary_transform_result<
            typename hpx::traits::range_iterator<Rng1>::type,
            typename hpx::traits::range_iterator<Rng2>::type, FwdIter>
        tag_fallback_invoke(hpx::ranges::transform_t, Rng1&& rng1, Rng2&& rng2,
            FwdIter dest, F&& f, Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            using iterator_type1 =
                typename hpx::traits::range_traits<Rng1>::iterator_type;
            using iterator_type2 =
                typename hpx::traits::range_traits<Rng2>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type1>::value &&
                    hpx::traits::is_forward_iterator<iterator_type2>::value,
                "Requires at least forward iterator.");

            return parallel::v1::detail::transform_binary2<
                binary_transform_result<iterator_type1, iterator_type2,
                    FwdIter>>()
                .call(hpx::execution::seq, hpx::util::begin(rng1),
                    hpx::util::end(rng1), hpx::util::begin(rng2),
                    hpx::util::end(rng2), dest, HPX_FORWARD(F, f),
                    HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));
        }

    } transform{};
}}    // namespace hpx::ranges

#endif    // DOXYGEN
