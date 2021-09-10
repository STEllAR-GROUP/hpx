//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/transform_inclusive_scan.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, conv(*first), ...,
    /// conv(*(first + (i - result) - 1))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicates \a op and \a conv.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Conv        The type of the unary function object used for
    ///                     the conversion operation.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param conv         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is a
    ///                     unary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     R fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    ///                     The type \a R must be such that an object of this
    ///                     type can be implicitly converted to \a T.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 and \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a transform_inclusive_scan
    /// algorithm invoked without an execution policy object execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a transform_inclusive_scan algorithm returns a
    ///           returns \a OutIter.
    ///           The \a transform_inclusive_scan algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// Neither \a conv nor \a op shall invalidate iterators or subranges, or
    /// modify elements in the ranges [first,last) or [result,result +
    /// (last - first)).
    ///
    /// The behavior of transform_inclusive_scan may be non-deterministic for
    /// a non-associative predicate.
    ///
    template <typename InIter, typename Sent, typename OutIter, typename T,
        typename BinOp, typename UnOp>
    OutIter transform_inclusive_scan(InIter first, Sent last, OutIter dest,
        T init, BinOp&& binary_op, UnOp&& unary_op);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, conv(*first), ...,
    /// conv(*(first + (i - result) - 1))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicates \a op and \a conv.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Conv        The type of the unary function object used for
    ///                     the conversion operation.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param conv         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is a
    ///                     unary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     R fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    ///                     The type \a R must be such that an object of this
    ///                     type can be implicitly converted to \a T.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 and \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a transform_inclusive_scan
    /// algorithm invoked with an execution policy object of type \a
    /// sequenced_policy execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a transform_inclusive_scan
    /// algorithm invoked with an execution policy object of type \a
    /// parallel_policy or \a parallel_task_policy are permitted to execute
    /// in an unordered fashion in unspecified threads, and indeterminately
    /// sequenced within each thread.
    ///
    /// \returns  The \a transform_inclusive_scan algorithm returns a
    ///           \a hpx::future<FwdIter2> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a transform_inclusive_scan algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// Neither \a conv nor \a op shall invalidate iterators or subranges, or
    /// modify elements in the ranges [first,last) or [result,result +
    /// (last - first)).
    ///
    /// The behavior of transform_inclusive_scan may be non-deterministic for
    /// a non-associative predicate.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent,
        typename FwdIter2, typename T, typename BinOp, typename UnOp>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    transform_inclusive_scan(ExPolicy&& policy, FwdIter1 first, Sent last,
        FwdIter2 dest, T init, BinOp&& binary_op, UnOp&& unary_op);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, conv(*first), ...,
    /// conv(*(first + (i - result) - 1))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicates \a op and \a conv.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    /// \tparam Conv        The type of the unary function object used for
    ///                     the conversion operation.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param conv         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is a
    ///                     unary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     R fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    ///                     The type \a R must be such that an object of this
    ///                     type can be implicitly converted to \a T.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 and \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a transform_inclusive_scan
    /// algorithm invoked without an execution policy object execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a transform_inclusive_scan algorithm returns a
    ///           returns \a O.
    ///           The \a transform_inclusive_scan algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// Neither \a conv nor \a op shall invalidate iterators or subranges, or
    /// modify elements in the ranges [first,last) or [result,result +
    /// (last - first)).
    ///
    /// The behavior of transform_inclusive_scan may be non-deterministic for
    /// a non-associative predicate.
    ///
    template <typename Rng, typename O, typename T, typename BinOp,
        typename UnOp>
    O transform_inclusive_scan(Rng&& rng, O dest, T init, BinOp&& binary_op,
        UnOp&& unary_op);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, conv(*first), ...,
    /// conv(*(first + (i - result) - 1))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicates \a op and \a conv.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam O           The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Conv        The type of the unary function object used for
    ///                     the conversion operation.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param conv         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is a
    ///                     unary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     R fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    ///                     The type \a R must be such that an object of this
    ///                     type can be implicitly converted to \a T.
    /// \param init         The initial value for the generalized sum.
    /// \param op           Specifies the function (or function object) which
    ///                     will be invoked for each of the values of the input
    ///                     sequence. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1 and \a Ret must be
    ///                     such that an object of a type as given by the input
    ///                     sequence can be implicitly converted to any
    ///                     of those types.
    ///
    /// The reduce operations in the parallel \a transform_inclusive_scan
    /// algorithm invoked with an execution policy object of type \a
    /// sequenced_policy execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a transform_inclusive_scan
    /// algorithm invoked with an execution policy object of type \a
    /// parallel_policy or \a parallel_task_policy are permitted to execute
    /// in an unordered fashion in unspecified threads, and indeterminately
    /// sequenced within each thread.
    ///
    /// \returns  The \a transform_inclusive_scan algorithm returns a
    ///           \a hpx::future<O> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a O otherwise.
    ///           The \a transform_inclusive_scan algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// Neither \a conv nor \a op shall invalidate iterators or subranges, or
    /// modify elements in the ranges [first,last) or [result,result +
    /// (last - first)).
    ///
    /// The behavior of transform_inclusive_scan may be non-deterministic for
    /// a non-associative predicate.
    ///
    template <typename ExPolicy, typename Rng,  typename O, typename T,
        typename BinOp, typename UnOp>
    typename parallel::util::detail::algorithm_result<ExPolicy, O>::type
    transform_inclusive_scan(ExPolicy&& policy, Rng&& rng, O dest, T init,
        BinOp&& binary_op, UnOp&& unary_op);

    // clang-format on
}}    // namespace hpx::ranges
#else

#include <hpx/config.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/transform_inclusive_scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace ranges {
    HPX_INLINE_CONSTEXPR_VARIABLE struct transform_inclusive_scan_t final
      : hpx::functional::tag_fallback<transform_inclusive_scan_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Sent, typename OutIter,
             typename BinOp, typename UnOp,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value &&
                hpx::traits::is_sentinel_for<Sent, InIter>::value &&
                hpx::traits::is_iterator<OutIter>::value &&
                hpx::is_invocable_v<UnOp,
                    typename std::iterator_traits<InIter>::value_type> &&
                hpx::is_invocable_v<BinOp,
                    typename hpx::util::invoke_result<UnOp,
                        typename std::iterator_traits<InIter>::value_type>::type,
                    typename hpx::util::invoke_result<UnOp,
                        typename std::iterator_traits<InIter>::value_type>::type
                >
            )>
        // clang-format on
        friend OutIter tag_fallback_dispatch(
            hpx::ranges::transform_inclusive_scan_t, InIter first, Sent last,
            OutIter dest, BinOp&& binary_op, UnOp&& unary_op)
        {
            static_assert(hpx::traits::is_input_iterator<InIter>::value,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_output_iterator<OutIter>::value,
                "Requires at least output iterator.");

            return hpx::parallel::v1::detail::transform_inclusive_scan<
                OutIter>()
                .call(hpx::execution::seq, first, last, dest,
                    std::forward<UnOp>(unary_op),
                    std::forward<BinOp>(binary_op));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent,
            typename FwdIter2, typename BinOp, typename UnOp,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<UnOp,
                    typename std::iterator_traits<FwdIter1>::value_type> &&
                hpx::is_invocable_v<BinOp,
                    typename hpx::util::invoke_result<UnOp,
                        typename std::iterator_traits<FwdIter1>::value_type>::type,
                    typename hpx::util::invoke_result<UnOp,
                        typename std::iterator_traits<FwdIter1>::value_type>::type
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_dispatch(hpx::ranges::transform_inclusive_scan_t,
            ExPolicy&& policy, FwdIter1 first, Sent last, FwdIter2 dest,
            BinOp&& binary_op, UnOp&& unary_op)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter1>::value,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<FwdIter2>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::transform_inclusive_scan<
                FwdIter2>()
                .call(std::forward<ExPolicy>(policy), first, last, dest,
                    std::forward<UnOp>(unary_op),
                    std::forward<BinOp>(binary_op));
        }

        // clang-format off
        template <typename Rng, typename O,
            typename BinOp, typename UnOp,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::is_invocable_v<UnOp,
                    typename hpx::traits::range_traits<Rng>::value_type> &&
                hpx::is_invocable_v<BinOp,
                    typename hpx::util::invoke_result<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>::type,
                    typename hpx::util::invoke_result<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>::type
                >
            )>
        // clang-format on
        friend O tag_fallback_dispatch(hpx::ranges::transform_inclusive_scan_t,
            Rng&& rng, O dest, BinOp&& binary_op, UnOp&& unary_op)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(hpx::traits::is_input_iterator<iterator_type>::value,
                "Requires at least input iterator.");

            return hpx::parallel::v1::detail::transform_inclusive_scan<O>()
                .call(hpx::execution::seq, std::begin(rng), std::end(rng), dest,
                    std::forward<UnOp>(unary_op),
                    std::forward<BinOp>(binary_op));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,  typename O,
            typename BinOp, typename UnOp,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::is_invocable_v<UnOp,
                    typename hpx::traits::range_traits<Rng>::value_type> &&
                hpx::is_invocable_v<BinOp,
                    typename hpx::util::invoke_result<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>::type,
                    typename hpx::util::invoke_result<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>::type
                >
            )>
        // clang-format on
        friend
            typename parallel::util::detail::algorithm_result<ExPolicy, O>::type
            tag_fallback_dispatch(hpx::ranges::transform_inclusive_scan_t,
                ExPolicy&& policy, Rng&& rng, O dest, BinOp&& binary_op,
                UnOp&& unary_op)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::transform_inclusive_scan<O>()
                .call(std::forward<ExPolicy>(policy), std::begin(rng),
                    std::end(rng), dest, std::forward<UnOp>(unary_op),
                    std::forward<BinOp>(binary_op));
        }

        template <typename InIter, typename Sent, typename OutIter,
            typename BinOp, typename UnOp, typename T,
            HPX_CONCEPT_REQUIRES_(hpx::traits::is_iterator<InIter>::value&&
                    hpx::traits::is_sentinel_for<Sent, InIter>::value&&
                        hpx::traits::is_iterator<
                            OutIter>::value&& hpx::is_invocable_v<UnOp,
                            typename std::iterator_traits<InIter>::value_type>&&
                            hpx::is_invocable_v<BinOp,
                                typename hpx::util::invoke_result<UnOp,
                                    typename std::iterator_traits<
                                        InIter>::value_type>::type,
                                typename hpx::util::invoke_result<UnOp,
                                    typename std::iterator_traits<
                                        InIter>::value_type>::type>)>
        // clang-format on
        friend OutIter tag_fallback_dispatch(
            hpx::ranges::transform_inclusive_scan_t, InIter first, Sent last,
            OutIter dest, BinOp&& binary_op, UnOp&& unary_op, T init)
        {
            static_assert(hpx::traits::is_input_iterator<InIter>::value,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_output_iterator<OutIter>::value,
                "Requires at least output iterator.");

            return hpx::parallel::v1::detail::transform_inclusive_scan<
                OutIter>()
                .call(hpx::execution::seq, first, last, dest,
                    std::forward<UnOp>(unary_op), std::move(init),
                    std::forward<BinOp>(binary_op));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent,
            typename FwdIter2, typename BinOp, typename UnOp, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<UnOp,
                    typename std::iterator_traits<FwdIter1>::value_type> &&
                hpx::is_invocable_v<BinOp,
                    typename hpx::util::invoke_result<UnOp,
                        typename std::iterator_traits<FwdIter1>::value_type>::type,
                    typename hpx::util::invoke_result<UnOp,
                        typename std::iterator_traits<FwdIter1>::value_type>::type
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_dispatch(hpx::ranges::transform_inclusive_scan_t,
            ExPolicy&& policy, FwdIter1 first, Sent last, FwdIter2 dest,
            BinOp&& binary_op, UnOp&& unary_op, T init)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter1>::value,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<FwdIter2>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::transform_inclusive_scan<
                FwdIter2>()
                .call(std::forward<ExPolicy>(policy), first, last, dest,
                    std::forward<UnOp>(unary_op), std::move(init),
                    std::forward<BinOp>(binary_op));
        }

        // clang-format off
        template <typename Rng, typename O, typename BinOp,
            typename UnOp, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::is_invocable_v<UnOp,
                    typename hpx::traits::range_traits<Rng>::value_type> &&
                hpx::is_invocable_v<BinOp,
                    typename hpx::util::invoke_result<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>::type,
                    typename hpx::util::invoke_result<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>::type
                >
            )>
        // clang-format on
        friend O tag_fallback_dispatch(hpx::ranges::transform_inclusive_scan_t,
            Rng&& rng, O dest, BinOp&& binary_op, UnOp&& unary_op, T init)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(hpx::traits::is_input_iterator<iterator_type>::value,
                "Requires at least input iterator.");

            return hpx::parallel::v1::detail::transform_inclusive_scan<O>()
                .call(hpx::execution::seq, std::begin(rng), std::end(rng), dest,
                    std::forward<UnOp>(unary_op), std::move(init),
                    std::forward<BinOp>(binary_op));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,  typename O,
            typename BinOp, typename UnOp, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::is_invocable_v<UnOp,
                    typename hpx::traits::range_traits<Rng>::value_type> &&
                hpx::is_invocable_v<BinOp,
                    typename hpx::util::invoke_result<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>::type,
                    typename hpx::util::invoke_result<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>::type
                >
            )>
        // clang-format on
        friend
            typename parallel::util::detail::algorithm_result<ExPolicy, O>::type
            tag_fallback_dispatch(hpx::ranges::transform_inclusive_scan_t,
                ExPolicy&& policy, Rng&& rng, O dest, BinOp&& binary_op,
                UnOp&& unary_op, T init)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::transform_inclusive_scan<O>()
                .call(std::forward<ExPolicy>(policy), std::begin(rng),
                    std::end(rng), dest, std::forward<UnOp>(unary_op),
                    std::move(init), std::forward<BinOp>(binary_op));
        }
    } transform_inclusive_scan{};
}}    // namespace hpx::ranges

#endif
