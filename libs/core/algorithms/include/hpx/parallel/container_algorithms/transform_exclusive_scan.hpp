//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/transform_exclusive_scan.hpp

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
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam BinOp       The type of the binary function object used for
    ///                     the reduction operation.
    /// \tparam UnOp        The type of the unary function object used for
    ///                     the conversion operation.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    /// \param binary_op    Specifies the function (or function object) which
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
    /// \param unary_op     Specifies the function (or function object) which
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
    ///
    /// The reduce operations in the parallel \a transform_exclusive_scan
    /// algorithm invoked without an execution policy object execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a transform_exclusive_scan algorithm returns \a
    ///           transform_exclusive_scan_result<InIter, OutIter>.
    ///           The \a transform_exclusive_scan algorithm returns an input
    ///           iterator to the point denoted by the sentinel and an output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// Neither \a conv nor \a op shall invalidate iterators or sub-ranges, or
    /// modify elements in the ranges [first,last) or [result,result +
    /// (last - first)).
    ///
    /// The behavior of transform_exclusive_scan may be non-deterministic for
    /// a non-associative predicate.
    ///
    template <typename InIter, typename Sent, typename OutIter,
        typename BinOp, typename UnOp,
        typename T = typename std::iterator_traits<InIter>::value_type>
    transform_exclusive_scan_result<InIter, OutIter> transform_exclusive_scan(
        InIter first, Sent last, OutIter dest, T init, BinOp&& binary_op,
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
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam BinOp       The type of the binary function object used for
    ///                     the reduction operation.
    /// \tparam UnOp        The type of the unary function object used for
    ///                     the conversion operation.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    /// \param binary_op    Specifies the function (or function object) which
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
    /// \param unary_op     Specifies the function (or function object) which
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
    ///
    /// The reduce operations in the parallel \a transform_exclusive_scan
    /// algorithm invoked with an execution policy object of type \a
    /// sequenced_policy execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a transform_exclusive_scan
    /// algorithm invoked with an execution policy object of type \a
    /// parallel_policy or \a parallel_task_policy are permitted to execute
    /// in an unordered fashion in unspecified threads, and indeterminately
    /// sequenced within each thread.
    ///
    /// \returns  The \a transform_exclusive_scan algorithm returns a
    ///           \a hpx::future<transform_exclusive_result<FwdIter1,
    ///           FwdIter2>> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a transform_exclusive_result<FwdIter1,
    ///           FwdIter2> otherwise.
    ///           The \a transform_exclusive_scan algorithm returns an input
    ///           iterator to the point denoted by the sentinel and an output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// Neither \a conv nor \a op shall invalidate iterators or sub-ranges, or
    /// modify elements in the ranges [first,last) or [result,result +
    /// (last - first)).
    ///
    /// The behavior of transform_exclusive_scan may be non-deterministic for
    /// a non-associative predicate.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent,
        typename FwdIter2, typename BinOp, typename UnOp,
        typename T = typename std::iterator_traits<FwdIter1>::value_type>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        transform_exclusive_result<FwdIter1, FwdIter2>>::type
    transform_exclusive_scan(ExPolicy&& policy, FwdIter1 first, Sent last,
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
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam BinOp       The type of the binary function object used for
    ///                     the reduction operation.
    /// \tparam UnOp        The type of the unary function object used for
    ///                     the conversion operation.
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    /// \param binary_op    Specifies the function (or function object) which
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
    /// \param unary_op     Specifies the function (or function object) which
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
    ///
    /// The reduce operations in the parallel \a transform_exclusive_scan
    /// algorithm invoked without an execution policy object execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a transform_exclusive_scan algorithm returns a
    ///           returns \a transform_exclusive_scan_result<
    ///           traits::range_iterator_t<Rng>, O>.
    ///           The \a transform_exclusive_scan algorithm returns an input
    ///           iterator to one past the end of the range and an output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// Neither \a conv nor \a op shall invalidate iterators or sub-ranges, or
    /// modify elements in the ranges [first,last) or [result,result +
    /// (last - first)).
    ///
    /// The behavior of transform_exclusive_scan may be non-deterministic for
    /// a non-associative predicate.
    ///
    template <typename Rng, typename O, typename BinOp, typename UnOp,
        typename T = typename std::iterator_traits<
            hpx::traits::range_iterator_t<Rng>>::value_type>
    transform_exclusive_scan_result<hpx::traits::range_iterator_t<Rng>, O>
    transform_exclusive_scan(Rng&& rng, O dest, T init, BinOp&& binary_op,
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
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam BinOp       The type of the binary function object used for
    ///                     the reduction operation.
    /// \tparam UnOp        The type of the unary function object used for
    ///                     the conversion operation.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    /// \param binary_op    Specifies the function (or function object) which
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
    /// \param unary_op     Specifies the function (or function object) which
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
    ///
    /// The reduce operations in the parallel \a transform_exclusive_scan
    /// algorithm invoked with an execution policy object of type \a
    /// sequenced_policy execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a transform_exclusive_scan
    /// algorithm invoked with an execution policy object of type \a
    /// parallel_policy or \a parallel_task_policy are permitted to execute
    /// in an unordered fashion in unspecified threads, and indeterminately
    /// sequenced within each thread.
    ///
    /// \returns  The \a transform_exclusive_scan algorithm returns a
    ///           \a hpx::future<transform_exclusive_scan_result<
    ///           traits::range_iterator_t<Rng>, O>> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a transform_exclusive_scan_result<
    ///           traits::range_iterator_t<Rng>, O> otherwise.
    ///           The \a transform_exclusive_scan algorithm returns an input
    ///           iterator to one past the end of the range and an output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// Neither \a conv nor \a op shall invalidate iterators or sub-ranges, or
    /// modify elements in the ranges [first,last) or [result,result +
    /// (last - first)).
    ///
    /// The behavior of transform_exclusive_scan may be non-deterministic for
    /// a non-associative predicate.
    ///
    template <typename ExPolicy, typename Rng,  typename O,
        typename BinOp, typename UnOp,
        typename T = typename std::iterator_traits<
            hpx::traits::range_iterator_t<Rng>>::value_type>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        transform_exclusive_scan_result<hpx::traits::range_iterator_t<Rng>,
            O>>::type
    transform_exclusive_scan(ExPolicy&& policy, Rng&& rng, O dest, T init,
        BinOp&& binary_op, UnOp&& unary_op);

    // clang-format on
}}    // namespace hpx::ranges
#else

#include <hpx/config.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/transform_exclusive_scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::ranges {

    template <typename I, typename O>
    using transform_exclusive_scan_result = parallel::util::in_out_result<I, O>;

    inline constexpr struct transform_exclusive_scan_t final
      : hpx::detail::tag_parallel_algorithm<transform_exclusive_scan_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Sent, typename OutIter,
            typename BinOp, typename UnOp,
            typename T = typename std::iterator_traits<InIter>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_sentinel_for_v<Sent, InIter> &&
                hpx::traits::is_iterator_v<OutIter> &&
                hpx::is_invocable_v<UnOp,
                    typename std::iterator_traits<InIter>::value_type> &&
                hpx::is_invocable_v<BinOp,
                    hpx::util::invoke_result_t<UnOp,
                        typename std::iterator_traits<InIter>::value_type>,
                    hpx::util::invoke_result_t<UnOp,
                        typename std::iterator_traits<InIter>::value_type>
                >
            )>
        // clang-format on
        friend transform_exclusive_scan_result<InIter, OutIter>
        tag_fallback_invoke(hpx::ranges::transform_exclusive_scan_t,
            InIter first, Sent last, OutIter dest, T init, BinOp binary_op,
            UnOp unary_op)
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_output_iterator_v<OutIter>,
                "Requires at least output iterator.");

            using result_type =
                transform_exclusive_scan_result<InIter, OutIter>;

            return hpx::parallel::detail::transform_exclusive_scan<
                result_type>()
                .call(hpx::execution::seq, first, last, dest,
                    HPX_MOVE(unary_op), HPX_MOVE(init), HPX_MOVE(binary_op));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent,
            typename FwdIter2, typename BinOp, typename UnOp,
            typename T = typename std::iterator_traits<FwdIter1>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter1> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter1> &&
                hpx::traits::is_iterator_v<FwdIter2> &&
                hpx::is_invocable_v<UnOp,
                    typename std::iterator_traits<FwdIter1>::value_type> &&
                hpx::is_invocable_v<BinOp,
                    hpx::util::invoke_result_t<UnOp,
                        typename std::iterator_traits<FwdIter1>::value_type>,
                    hpx::util::invoke_result_t<UnOp,
                        typename std::iterator_traits<FwdIter1>::value_type>
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            transform_exclusive_scan_result<FwdIter1, FwdIter2>>
        tag_fallback_invoke(hpx::ranges::transform_exclusive_scan_t,
            ExPolicy&& policy, FwdIter1 first, Sent last, FwdIter2 dest, T init,
            BinOp binary_op, UnOp unary_op)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");

            using result_type =
                transform_exclusive_scan_result<FwdIter1, FwdIter2>;

            return hpx::parallel::detail::transform_exclusive_scan<
                result_type>()
                .call(HPX_FORWARD(ExPolicy, policy), first, last, dest,
                    HPX_MOVE(unary_op), HPX_MOVE(init), HPX_MOVE(binary_op));
        }

        // clang-format off
        template <typename Rng, typename O, typename BinOp, typename UnOp,
            typename T = typename std::iterator_traits<
                hpx::traits::range_iterator_t<Rng>>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                hpx::is_invocable_v<UnOp,
                    typename hpx::traits::range_traits<Rng>::value_type> &&
                hpx::is_invocable_v<BinOp,
                    hpx::util::invoke_result_t<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>,
                    hpx::util::invoke_result_t<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>
                >
            )>
        // clang-format on
        friend transform_exclusive_scan_result<
            hpx::traits::range_iterator_t<Rng>, O>
        tag_fallback_invoke(hpx::ranges::transform_exclusive_scan_t, Rng&& rng,
            O dest, T init, BinOp binary_op, UnOp unary_op)
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_input_iterator_v<iterator_type>,
                "Requires at least input iterator.");

            using result_type =
                transform_exclusive_scan_result<iterator_type, O>;

            return hpx::parallel::detail::transform_exclusive_scan<
                result_type>()
                .call(hpx::execution::seq, std::begin(rng), std::end(rng), dest,
                    HPX_MOVE(unary_op), HPX_MOVE(init), HPX_MOVE(binary_op));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,  typename O,
            typename BinOp, typename UnOp,
            typename T = typename std::iterator_traits<
                hpx::traits::range_iterator_t<Rng>>::value_type,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                hpx::is_invocable_v<UnOp,
                    typename hpx::traits::range_traits<Rng>::value_type> &&
                hpx::is_invocable_v<BinOp,
                    hpx::util::invoke_result_t<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>,
                    hpx::util::invoke_result_t<UnOp,
                        typename hpx::traits::range_traits<Rng>::value_type>
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            transform_exclusive_scan_result<hpx::traits::range_iterator_t<Rng>,
                O>>::type
        tag_fallback_invoke(hpx::ranges::transform_exclusive_scan_t,
            ExPolicy&& policy, Rng&& rng, O dest, T init, BinOp binary_op,
            UnOp unary_op)
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Requires at least forward iterator.");

            using result_type =
                transform_exclusive_scan_result<iterator_type, O>;

            return hpx::parallel::detail::transform_exclusive_scan<
                result_type>()
                .call(HPX_FORWARD(ExPolicy, policy), std::begin(rng),
                    std::end(rng), dest, HPX_MOVE(unary_op), HPX_MOVE(init),
                    HPX_MOVE(binary_op));
        }
    } transform_exclusive_scan{};
}    // namespace hpx::ranges

#endif
