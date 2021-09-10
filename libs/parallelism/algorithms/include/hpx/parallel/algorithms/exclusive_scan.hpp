//  Copyright (c) 2014-2020 Hartmut Kaiser
//  Copyright (c) 2016 Minh-Khanh Do
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/exclusive_scan.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, init, *first, ...,
    /// *(first + (i - result) - 1))
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a std::plus<T>.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns \a OutIter.
    ///           The \a exclusive_scan algorithm returns the output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aK)
    ///           + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template <typename InIter, typename OutIter, typename T>
    OutIter exclusive_scan(InIter first, InIter last, OutIter dest, T init);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, init, *first, ...,
    /// *(first + (i - result) - 1))
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a std::plus<T>.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
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
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns a
    ///           \a hpx::future<FwdIter2> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a exclusive_scan algorithm returns the output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aK)
    ///           + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T>
    typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    exclusive_scan(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
        FwdIter2 dest, T init);

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, *first, ...,
    /// *(first + (i - result) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
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
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns \a OutIter.
    ///           The \a exclusive_scan algorithm returns the output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN))
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum. If
    /// \a op is not mathematically associative, the behavior of
    /// \a inclusive_scan may be non-deterministic.
    ///
    template <typename InIter, typename OutIter,typename T, typename Op>
    OutIter exclusive_scan(InIter first, InIter last, OutIter dest,
        T init, Op&& op);

        ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(binary_op, init, *first, ...,
    /// *(first + (i - result) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    /// \tparam Op          The type of the binary function object used for
    ///                     the reduction operation.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
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
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a exclusive_scan algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a exclusive_scan algorithm returns a
    ///           \a hpx::future<OutIter> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a exclusive_scan algorithm returns the output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK),
    ///           GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN))
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum. If
    /// \a op is not mathematically associative, the behavior of
    /// \a inclusive_scan may be non-deterministic.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T, typename Op>
    typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    exclusive_scan(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
        FwdIter2 dest, T init, Op&& op);
    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/zip_iterator.hpp>
#include <hpx/pack_traversal/unwrap.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/type_support/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // exclusive_scan
    namespace detail {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        // Our own version of the sequential exclusive_scan.
        template <typename InIter, typename Sent, typename OutIter, typename T,
            typename Op>
        OutIter sequential_exclusive_scan(
            InIter first, Sent last, OutIter dest, T init, Op&& op)
        {
            T temp = init;
            for (/* */; first != last; (void) ++first, ++dest)
            {
                init = hpx::util::invoke(op, init, *first);
                *dest = temp;
                temp = init;
            }
            return dest;
        }

        template <typename InIter, typename OutIter, typename T, typename Op>
        T sequential_exclusive_scan_n(
            InIter first, std::size_t count, OutIter dest, T init, Op&& op)
        {
            T temp = init;
            for (/* */; count-- != 0; (void) ++first, ++dest)
            {
                init = hpx::util::invoke(op, init, *first);
                *dest = temp;
                temp = init;
            }
            return init;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter2>
        struct exclusive_scan
          : public detail::algorithm<exclusive_scan<FwdIter2>, FwdIter2>
        {
            exclusive_scan()
              : exclusive_scan::algorithm("exclusive_scan")
            {
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename OutIter, typename T, typename Op>
            static OutIter sequential(ExPolicy, InIter first, Sent last,
                OutIter dest, T const& init, Op&& op)
            {
                return sequential_exclusive_scan(
                    first, last, dest, init, std::forward<Op>(op));
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent,
                typename T, typename Op>
            static typename util::detail::algorithm_result<ExPolicy,
                FwdIter2>::type
            parallel(ExPolicy&& policy, FwdIter1 first, Sent last,
                FwdIter2 dest, T init, Op&& op)
            {
                typedef util::detail::algorithm_result<ExPolicy, FwdIter2>
                    result;
                typedef hpx::util::zip_iterator<FwdIter1, FwdIter2>
                    zip_iterator;
                typedef typename std::iterator_traits<FwdIter1>::difference_type
                    difference_type;

                if (first == last)
                    return result::get(std::move(dest));

                difference_type count = detail::distance(first, last);

                FwdIter2 final_dest = dest;
                std::advance(final_dest, count);

                // The overall scan algorithm is performed by executing 3
                // steps. The first calculates the scan results for each
                // partition. The second accumulates the result from left to
                // right to be used by the third step--which operates on the
                // same partitions the first step operated on.

                using hpx::get;
                using hpx::util::make_zip_iterator;

                auto f3 = [op](zip_iterator part_begin, std::size_t part_size,
                              hpx::shared_future<T> curr,
                              hpx::shared_future<T> next) {
                    next.get();    // rethrow exceptions

                    T val = curr.get();
                    FwdIter2 dst = get<1>(part_begin.get_iterator_tuple());
                    *dst++ = val;

                    // MSVC 2015 fails if op is captured by reference
                    util::loop_n<std::decay_t<ExPolicy>>(
                        dst, part_size - 1, [=, &val](FwdIter2 it) {
                            *it = hpx::util::invoke(op, val, *it);
                        });
                };

                return util::scan_partitioner<ExPolicy, FwdIter2, T>::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first, dest), count, init,
                    // step 1 performs first part of scan algorithm
                    [op, last](
                        zip_iterator part_begin, std::size_t part_size) -> T {
                        T part_init = get<0>(*part_begin++);

                        auto iters = part_begin.get_iterator_tuple();
                        if (get<0>(iters) != last)
                        {
                            return sequential_exclusive_scan_n(get<0>(iters),
                                part_size - 1, get<1>(iters), part_init, op);
                        }
                        return part_init;
                    },
                    // step 2 propagates the partition results from left
                    // to right
                    hpx::unwrapping(op),
                    // step 3 runs final accumulation on each partition
                    std::move(f3),
                    // step 4 use this return value
                    [final_dest](std::vector<hpx::shared_future<T>>&&,
                        std::vector<hpx::future<void>>&&) {
                        return final_dest;
                    });
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T, typename Op,
        HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
    // clang-format on
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::exclusive_scan is deprecated, use hpx::exclusive_scan "
        "instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        exclusive_scan(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, T init, Op&& op)
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return hpx::parallel::v1::detail::exclusive_scan<FwdIter2>().call(
            std::forward<ExPolicy>(policy), first, last, dest, std::move(init),
            std::forward<Op>(op));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename T,
        HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
    // clang-format on
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::exclusive_scan is deprecated, use hpx::exclusive_scan "
        "instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        exclusive_scan(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, T init)
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return hpx::parallel::v1::detail::exclusive_scan<FwdIter2>().call(
            std::forward<ExPolicy>(policy), first, last, dest, std::move(init),
            std::plus<T>());
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::exclusive_scan
    HPX_INLINE_CONSTEXPR_VARIABLE struct exclusive_scan_t final
      : hpx::functional::tag_fallback<exclusive_scan_t>
    {
        // clang-format off
        template <typename InIter, typename OutIter, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value &&
                hpx::traits::is_iterator<OutIter>::value
            )>
        // clang-format on
        friend OutIter tag_fallback_dispatch(hpx::exclusive_scan_t,
            InIter first, InIter last, OutIter dest, T init)
        {
            static_assert((hpx::traits::is_input_iterator<InIter>::value),
                "Requires at least input iterator.");
            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Requires at least output iterator.");

            return hpx::parallel::v1::detail::exclusive_scan<OutIter>().call(
                hpx::execution::seq, first, last, dest, std::move(init),
                std::plus<T>());
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter1>::value &&
                hpx::traits::is_forward_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_dispatch(hpx::exclusive_scan_t, ExPolicy&& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest, T init)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter1>::value,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<FwdIter2>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::exclusive_scan<FwdIter2>().call(
                std::forward<ExPolicy>(policy), first, last, dest,
                std::move(init), std::plus<T>());
        }

        // clang-format off
        template <typename InIter, typename OutIter, typename T,
            typename Op,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value &&
                hpx::traits::is_iterator<OutIter>::value
            )>
        // clang-format on
        friend OutIter tag_fallback_dispatch(hpx::exclusive_scan_t,
            InIter first, InIter last, OutIter dest, T init, Op&& op)
        {
            static_assert((hpx::traits::is_input_iterator<InIter>::value),
                "Requires at least input iterator.");
            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Requires at least output iterator.");

            return hpx::parallel::v1::detail::exclusive_scan<OutIter>().call(
                hpx::execution::seq, first, last, dest, std::move(init),
                std::forward<Op>(op));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename T, typename Op,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter1>::value &&
                hpx::traits::is_forward_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_dispatch(hpx::exclusive_scan_t, ExPolicy&& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest, T init, Op&& op)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter1>::value,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<FwdIter2>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::exclusive_scan<FwdIter2>().call(
                std::forward<ExPolicy>(policy), first, last, dest,
                std::move(init), std::forward<Op>(op));
        }

    } exclusive_scan{};
}    // namespace hpx

#endif    // DOXYGEN
