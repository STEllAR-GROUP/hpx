//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/transform_reduce.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Returns GENERALIZED_SUM(red_op, init, conv_op(*first), ...,
    /// conv_op(*(first + (last - first) - 1))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicates \a red_op and \a conv_op.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Reduce      The type of the binary function object used for
    ///                     the reduction operation.
    /// \tparam Convert     The type of the unary function object used to
    ///                     transform the elements of the input sequence before
    ///                     invoking the reduce function.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param init         The initial value for the generalized sum.
    /// \param red_op       Specifies the function (or function object) which
    ///                     will be invoked for each of the values returned
    ///                     from the invocation of \a conv_op. This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The types \a Type1, \a Type2, and \a Ret must be
    ///                     such that an object of a type as returned from
    ///                     \a conv_op can be implicitly converted to any
    ///                     of those types.
    /// \param conv_op      Specifies the function (or function object) which
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
    ///                     type \a Iter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///                     The type \a R must be such that an object of this
    ///                     type can be implicitly converted to \a T.
    ///
    /// The reduce operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform_reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///           The \a transform_reduce algorithm returns the result of the
    ///           generalized sum over the values returned from \a conv_op when
    ///           applied to the elements given by the input range
    ///           [first, last).
    ///
    /// \note   GENERALIZED_SUM(op, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(op, b1, ..., bK), GENERALIZED_SUM(op, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a transform_reduce and \a accumulate is
    /// that the behavior of transform_reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename Rng, typename T, typename Reduce,
        typename Convert>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    transform_reduce(ExPolicy&& policy, Rng&& rng, T init, Reduce&& red_op,
        Convert&& conv_op);

    ///////////////////////////////////////////////////////////////////////////
    /// Returns the result of accumulating init with the inner products of the
    /// pairs formed by the elements of two ranges starting at first1 and
    /// first2.
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op2.
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam FwdIter2    The type of the second source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an forward iterator.
    /// \tparam T           The type of the value to be used as return)
    ///                     values (deduced).
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the result will be calculated with.
    /// \param init         The initial value for the sum.
    ///
    /// The operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform_reduce algorithm returns a \a hpx::future<T> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///
    template <typename ExPolicy, typename Rng1, typename FwdIter2, typename T>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    transform_reduce(ExPolicy&& policy, Rng1&& rng1, FwdIter2 first2, T init);

    ///////////////////////////////////////////////////////////////////////////
    /// Returns the result of accumulating init with the inner products of the
    /// pairs formed by the elements of two ranges starting at first1 and
    /// first2.
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op2.
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam FwdIter2    The type of the second source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an forward iterator.
    /// \tparam T           The type of the value to be used as return)
    ///                     values (deduced).
    /// \tparam Reduce      The type of the binary function object used for
    ///                     the multiplication operation.
    /// \tparam Convert     The type of the unary function object used to
    ///                     transform the elements of the input sequence before
    ///                     invoking the reduce function.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param first2       Refers to the beginning of the second sequence of
    ///                     elements the result will be calculated with.
    /// \param init         The initial value for the sum.
    /// \param red_op       Specifies the function (or function object) which
    ///                     will be invoked for the initial value and each
    ///                     of the return values of \a op2.
    ///                     This is a binary predicate. The
    ///                     signature of this predicate should be equivalent to
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The type \a Ret must be
    ///                     such that it can be implicitly converted to a type
    ///                     of \a T.
    /// \param conv_op      Specifies the function (or function object) which
    ///                     will be invoked for each of the input values
    ///                     of the sequence. This is a binary predicate. The
    ///                     signature of this predicate should be equivalent to
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it.
    ///                     The type \a Ret must be such that it can be
    ///                     implicitly converted to an object for the second
    ///                     argument type of \a op1.
    ///
    /// The operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform_reduce algorithm returns a \a hpx::future<T>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///
    template <typename ExPolicy, typename Rng1, typename FwdIter2,
        typename T, typename Reduce, typename Convert>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    transform_reduce(ExPolicy&& policy, Rng1&& rng1, FwdIter2 first2, T init,
        Reduce&& red_op, Convert&& conv_op);

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/segmented_iterator_traits.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>
#include <hpx/pack_traversal/unwrap.hpp>

#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/accumulate.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::transform_reduce
    HPX_INLINE_CONSTEXPR_VARIABLE struct transform_reduce_t final
      : hpx::functional::tag<transform_reduce_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent, typename T,
            typename Reduce, typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::is_invocable_v<Convert,
                    typename std::iterator_traits<Iter>::value_type
                > &&
                hpx::is_invocable_v<Reduce,
                   typename hpx::util::invoke_result<Convert,
                       typename std::iterator_traits<Iter>::value_type
                   >::type,
                   typename hpx::util::invoke_result<Convert,
                       typename std::iterator_traits<Iter>::value_type
                   >::type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(transform_reduce_t, ExPolicy&& policy, Iter first, Sent last,
            T init, Reduce&& red_op, Convert&& conv_op)
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Requires at least forward iterator.");

            using is_segmented = hpx::traits::is_segmented_iterator<Iter>;

            return hpx::parallel::v1::detail::transform_reduce_(
                std::forward<ExPolicy>(policy), first, last, std::move(init),
                std::forward<Reduce>(red_op), std::forward<Convert>(conv_op),
                is_segmented());
        }

        // clang-format off
        template <typename Iter, typename Sent, typename T, typename Reduce,
            typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::is_invocable_v<Convert,
                   typename std::iterator_traits<Iter>::value_type
                > &&
                hpx::is_invocable_v<Reduce,
                   typename hpx::util::invoke_result<Convert,
                       typename std::iterator_traits<Iter>::value_type
                   >::type,
                   typename hpx::util::invoke_result<Convert,
                       typename std::iterator_traits<Iter>::value_type
                   >::type
                >
            )>
        // clang-format on
        friend T tag_invoke(transform_reduce_t, Iter first, Sent last, T init,
            Reduce&& red_op, Convert&& conv_op)
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Requires at least forward iterator.");

            using is_segmented = hpx::traits::is_segmented_iterator<Iter>;

            return hpx::parallel::v1::detail::transform_reduce_(
                hpx::execution::seq, first, last, std::move(init),
                std::forward<Reduce>(red_op), std::forward<Convert>(conv_op),
                is_segmented());
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
            typename Iter2, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(transform_reduce_t, ExPolicy&& policy, Iter first, Sent last,
            Iter2 first2, T init)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<Iter>;

            return hpx::parallel::v1::detail::transform_reduce_(
                std::forward<ExPolicy>(policy), first, last, first2,
                std::move(init), hpx::parallel::v1::detail::plus(),
                hpx::parallel::v1::detail::multiplies(), is_segmented());
        }

        // clang-format off
        template <typename Iter, typename Sent, typename Iter2, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend T tag_invoke(
            transform_reduce_t, Iter first, Sent last, Iter2 first2, T init)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<Iter>;

            return hpx::parallel::v1::detail::transform_reduce_(
                hpx::execution::seq, first, last, first2, std::move(init),
                hpx::parallel::v1::detail::plus(),
                hpx::parallel::v1::detail::multiplies(), is_segmented());
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
            typename Iter2, typename T, typename Reduce, typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::traits::is_iterator<Iter2>::value &&
                hpx::is_invocable_v<Convert,
                    typename std::iterator_traits<Iter>::value_type,
                    typename std::iterator_traits<Iter2>::value_type
                > &&
                hpx::is_invocable_v<Reduce,
                    typename hpx::util::invoke_result<Convert,
                        typename std::iterator_traits<Iter>::value_type,
                        typename std::iterator_traits<Iter2>::value_type
                    >::type,
                    typename hpx::util::invoke_result<Convert,
                        typename std::iterator_traits<Iter>::value_type,
                        typename std::iterator_traits<Iter2>::value_type
                    >::type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(transform_reduce_t, ExPolicy&& policy, Iter first, Sent last,
            Iter2 first2, T init, Reduce&& red_op, Convert&& conv_op)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<Iter>;

            return hpx::parallel::v1::detail::transform_reduce_(
                std::forward<ExPolicy>(policy), first, last, first2,
                std::move(init), std::forward<Reduce>(red_op),
                std::forward<Convert>(conv_op), is_segmented());
        }

        // clang-format off
        template <typename Iter, typename Sent, typename Iter2, typename T,
            typename Reduce, typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::traits::is_iterator<Iter2>::value &&
                hpx::is_invocable_v<Convert,
                    typename std::iterator_traits<Iter>::value_type,
                    typename std::iterator_traits<Iter2>::value_type
                > &&
                hpx::is_invocable_v<Reduce,
                    typename hpx::util::invoke_result<Convert,
                        typename std::iterator_traits<Iter>::value_type,
                        typename std::iterator_traits<Iter2>::value_type
                    >::type,
                    typename hpx::util::invoke_result<Convert,
                        typename std::iterator_traits<Iter>::value_type,
                        typename std::iterator_traits<Iter2>::value_type
                    >::type
                >
            )>
        // clang-format on
        friend T tag_invoke(transform_reduce_t, Iter first, Sent last,
            Iter2 first2, T init, Reduce&& red_op, Convert&& conv_op)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<Iter>;

            return hpx::parallel::v1::detail::transform_reduce_(
                hpx::execution::seq, first, last, first2, std::move(init),
                std::forward<Reduce>(red_op), std::forward<Convert>(conv_op),
                is_segmented());
        }

        // range based versions
        // clang-format off
        template <typename ExPolicy, typename Rng, typename T, typename Reduce,
            typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::is_invocable_v<Convert,
                    typename hpx::traits::range_traits<Rng>::value_type
                > &&
                hpx::is_invocable_v<Reduce,
                    typename hpx::util::invoke_result<Convert,
                        typename hpx::traits::range_traits<Rng>::value_type
                    >::type,
                    typename hpx::util::invoke_result<Convert,
                        typename hpx::traits::range_traits<Rng>::value_type
                    >::type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(transform_reduce_t, ExPolicy&& policy, Rng&& rng, T init,
            Reduce&& red_op, Convert&& conv_op)
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                (hpx::traits::is_forward_iterator<iterator_type>::value),
                "Requires at least forward iterator.");

            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            return hpx::parallel::v1::detail::transform_reduce_(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), std::move(init),
                std::forward<Reduce>(red_op), std::forward<Convert>(conv_op),
                is_segmented());
        }

        // clang-format off
        template <typename Rng, typename T, typename Reduce, typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::is_invocable_v<Convert,
                    typename hpx::traits::range_traits<Rng>::value_type
                > &&
                hpx::is_invocable_v<Reduce,
                    typename hpx::util::invoke_result<Convert,
                        typename hpx::traits::range_traits<Rng>::value_type
                    >::type,
                    typename hpx::util::invoke_result<Convert,
                        typename hpx::traits::range_traits<Rng>::value_type
                    >::type
                >
            )>
        // clang-format on
        friend T tag_invoke(transform_reduce_t, Rng&& rng, T init,
            Reduce&& red_op, Convert&& conv_op)
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                (hpx::traits::is_forward_iterator<iterator_type>::value),
                "Requires at least forward iterator.");

            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            return hpx::parallel::v1::detail::transform_reduce_(
                hpx::execution::seq, hpx::util::begin(rng), hpx::util::end(rng),
                std::move(init), std::forward<Reduce>(red_op),
                std::forward<Convert>(conv_op), is_segmented());
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Iter2, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(transform_reduce_t, ExPolicy&& policy, Rng&& rng,
            Iter2 first2, T init)
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;
            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            return hpx::parallel::v1::detail::transform_reduce_(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), first2, std::move(init),
                hpx::parallel::v1::detail::plus(),
                hpx::parallel::v1::detail::multiplies(), is_segmented());
        }

        // clang-format off
        template <typename Rng, typename Iter2, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend T tag_invoke(transform_reduce_t, Rng&& rng, Iter2 first2, T init)
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;
            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            return hpx::parallel::v1::detail::transform_reduce_(
                hpx::execution::seq, hpx::util::begin(rng), hpx::util::end(rng),
                first2, std::move(init), hpx::parallel::v1::detail::plus(),
                hpx::parallel::v1::detail::multiplies(), is_segmented());
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Iter2,
            typename T, typename Reduce, typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator<Iter2>::value &&
                hpx::is_invocable_v<Convert,
                    typename hpx::traits::range_traits<Rng>::value_type,
                    typename std::iterator_traits<Iter2>::value_type
                > &&
                hpx::is_invocable_v<Reduce,
                    typename hpx::util::invoke_result<Convert,
                        typename hpx::traits::range_traits<Rng>::value_type,
                        typename std::iterator_traits<Iter2>::value_type
                    >::type,
                    typename hpx::util::invoke_result<Convert,
                        typename hpx::traits::range_traits<Rng>::value_type,
                        typename std::iterator_traits<Iter2>::value_type
                    >::type
                >
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(transform_reduce_t, ExPolicy&& policy, Rng&& rng,
            Iter2 first2, T init, Reduce&& red_op, Convert&& conv_op)
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;
            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            return hpx::parallel::v1::detail::transform_reduce_(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), first2, std::move(init),
                std::forward<Reduce>(red_op), std::forward<Convert>(conv_op),
                is_segmented());
        }

        // clang-format off
        template <typename Rng, typename Iter2, typename T, typename Reduce,
            typename Convert,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator<Iter2>::value &&
                hpx::is_invocable_v<Convert,
                    typename hpx::traits::range_traits<Rng>::value_type,
                    typename std::iterator_traits<Iter2>::value_type
                > &&
                hpx::is_invocable_v<Reduce,
                    typename hpx::util::invoke_result<Convert,
                        typename hpx::traits::range_traits<Rng>::value_type,
                        typename std::iterator_traits<Iter2>::value_type
                    >::type,
                    typename hpx::util::invoke_result<Convert,
                        typename hpx::traits::range_traits<Rng>::value_typee,
                        typename std::iterator_traits<Iter2>::value_type
                    >::type
                >
            )>
        // clang-format on
        friend T tag_invoke(transform_reduce_t, Rng&& rng, Iter2 first2, T init,
            Reduce&& red_op, Convert&& conv_op)
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;
            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            return hpx::parallel::v1::detail::transform_reduce_(
                hpx::execution::seq, hpx::util::begin(rng), hpx::util::end(rng),
                first2, std::move(init), std::forward<Reduce>(red_op),
                std::forward<Convert>(conv_op), is_segmented());
        }
    } transform_reduce{};
}}    // namespace hpx::ranges

#endif    // DOXYGEN
