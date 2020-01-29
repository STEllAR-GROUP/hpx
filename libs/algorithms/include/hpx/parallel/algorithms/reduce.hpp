//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/reduce.hpp

#if !defined(HPX_PARALLEL_DETAIL_REDUCE_JUN_01_2014_0903AM)
#define HPX_PARALLEL_DETAIL_REDUCE_JUN_01_2014_0903AM

#include <hpx/config.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/util/unwrap.hpp>

#include <hpx/execution/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/accumulate.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
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

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // reduce
    namespace detail {
        /// \cond NOINTERNAL
        template <typename T>
        struct reduce : public detail::algorithm<reduce<T>, T>
        {
            reduce()
              : reduce::algorithm("reduce")
            {
            }

            template <typename ExPolicy, typename InIterB, typename InIterE,
                typename T_, typename Reduce>
            static T sequential(
                ExPolicy, InIterB first, InIterE last, T_&& init, Reduce&& r)
            {
                return detail::accumulate(first, last, std::forward<T_>(init),
                    std::forward<Reduce>(r));
            }

            template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
                typename T_, typename Reduce>
            static typename util::detail::algorithm_result<ExPolicy, T>::type
            parallel(ExPolicy&& policy, FwdIterB first, FwdIterE last,
                T_&& init, Reduce&& r)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy, T>::get(
                        std::forward<T_>(init));
                }

                auto f1 = [r](FwdIterB part_begin, std::size_t part_size) -> T {
                    T val = *part_begin;
                    return util::accumulate_n(
                        ++part_begin, --part_size, std::move(val), r);
                };

                return util::partitioner<ExPolicy, T>::call(
                    std::forward<ExPolicy>(policy), first,
                    detail::distance(first, last), std::move(f1),
                    hpx::util::unwrapping([init = std::forward<T_>(init),
                                              r = std::forward<Reduce>(r)](
                                              std::vector<T>&& results) -> T {
                        return util::accumulate_n(hpx::util::begin(results),
                            hpx::util::size(results), init, r);
                    }));
            }
        };
        /// \endcond
        // Non Segmented Reduce
        template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
            typename T, typename F>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, T>::type>::type
        reduce_(ExPolicy&& policy, FwdIterB first, FwdIterE last, T init, F&& f,
            std::false_type)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIterB>::value),
                "Requires at least forward iterator.");

            typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

            return detail::reduce<T>().call(std::forward<ExPolicy>(policy),
                is_seq(), first, last, std::move(init), std::forward<F>(f));
        }

        // Forward Declaration of Segmented Reduce
        template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
            typename T, typename F>
        inline typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, T>::type>::type
        reduce_(ExPolicy&& policy, FwdIterB first, FwdIterE last, T init, F&& f,
            std::true_type);
    }    // namespace detail

    /// Returns GENERALIZED_SUM(f, init, *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIterB    The type of the source begin iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIterE    The type of the source end iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The types \a Type1 \a Ret must be
    ///                     such that an object of type \a FwdIterB can be
    ///                     dereferenced and then implicitly converted to any
    ///                     of those types.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum over the elements given by the input range
    ///           [first, last).
    ///
    /// \note   GENERALIZED_SUM(op, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(op, b1, ..., bK), GENERALIZED_SUM(op, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
        typename T, typename F>
    inline
        typename std::enable_if<execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, T>::type>::type
        reduce(ExPolicy&& policy, FwdIterB first, FwdIterE last, T init, F&& f)
    {
        typedef hpx::traits::is_segmented_iterator<FwdIterB> is_segmented;

        return detail::reduce_(std::forward<ExPolicy>(policy), first, last,
            std::move(init), std::forward<F>(f), is_segmented());
    }

    /// Returns GENERALIZED_SUM(+, init, *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         operator+().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIterB    The type of the source begin iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIterE    The type of the source end iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum (applying operator+()) over the elements given
    ///           by the input range [first, last).
    ///
    /// \note   GENERALIZED_SUM(+, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(+, b1, ..., bK), GENERALIZED_SUM(+, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
        typename T>
    inline
        typename std::enable_if<execution::is_execution_policy<ExPolicy>::value,
            typename util::detail::algorithm_result<ExPolicy, T>::type>::type
        reduce(ExPolicy&& policy, FwdIterB first, FwdIterE last, T init)
    {
        typedef hpx::traits::is_segmented_iterator<FwdIterB> is_segmented;

        return detail::reduce_(std::forward<ExPolicy>(policy), first, last,
            std::move(init), std::plus<T>(), is_segmented());
    }

    /// Returns GENERALIZED_SUM(+, T(), *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         operator+().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIterB    The type of the source begin iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIterE    The type of the source end iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns T otherwise (where T is the value_type of
    ///           \a FwdIterB).
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum (applying operator+()) over the elements given
    ///           by the input range [first, last).
    ///
    /// \note   The type of the initial value (and the result type) \a T is
    ///         determined from the value_type of the used \a FwdIterB.
    ///
    /// \note   GENERALIZED_SUM(+, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(+, b1, ..., bK), GENERALIZED_SUM(+, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename FwdIterB, typename FwdIterE>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<FwdIterB>::value_type>::type>::type
    reduce(ExPolicy&& policy, FwdIterB first, FwdIterE last)
    {
        typedef typename std::iterator_traits<FwdIterB>::value_type value_type;

        typedef hpx::traits::is_segmented_iterator<FwdIterB> is_segmented;

        return detail::reduce_(std::forward<ExPolicy>(policy), first, last,
            value_type(), std::plus<value_type>(), is_segmented());
    }
}}}    // namespace hpx::parallel::v1

#endif
