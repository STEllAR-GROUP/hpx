//  Copyright (c) 2014-2016 Hartmut Kaiser
//  Copyright (c) 2016 Minh-Khanh Do
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/inclusive_scan.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_INCLUSIVE_SCAN_JAN_03_2015_0136PM)
#define HPX_PARALLEL_ALGORITHM_INCLUSIVE_SCAN_JAN_03_2015_0136PM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/unwrapped.hpp>
#include <hpx/util/zip_iterator.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // inclusive_scan
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        // Our own version of the sequential inclusive_scan.
        template <typename InIter, typename OutIter, typename T, typename Op>
        OutIter sequential_inclusive_scan(InIter first, InIter last,
            OutIter dest, T init, Op && op)
        {
            for (/* */; first != last; (void) ++first, ++dest)
            {
                init = op(init, *first);
                *dest = init;
            }
            return dest;
        }

        template <typename InIter, typename OutIter, typename T, typename Op>
        T sequential_inclusive_scan_n(InIter first, std::size_t count,
            OutIter dest, T init, Op && op)
        {
            for (/* */; count-- != 0; (void) ++first, ++dest)
            {
                init = op(init, *first);
                *dest = init;
            }
            return init;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename OutIter>
        struct inclusive_scan
          : public detail::algorithm<inclusive_scan<OutIter>, OutIter>
        {
            inclusive_scan()
              : inclusive_scan::algorithm("inclusive_scan")
            {}

            template <typename ExPolicy, typename InIter, typename T, typename Op>
            static OutIter
            sequential(ExPolicy, InIter first, InIter last,
                OutIter dest, T const& init, Op && op)
            {
                return sequential_inclusive_scan(first, last, dest,
                    init, std::forward<Op>(op));
            }

            template <typename ExPolicy, typename FwdIter, typename T, typename Op>
            static typename util::detail::algorithm_result<
                ExPolicy, OutIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                 OutIter dest, T const& init, Op && op)
            {
                typedef util::detail::algorithm_result<ExPolicy, OutIter>
                    result;
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                if (first == last)
                    return result::get(std::move(dest));

                difference_type count = std::distance(first, last);

                OutIter final_dest = dest;
                std::advance(final_dest, count);

                // The overall scan algorithm is performed by executing 3
                // steps. The first calculates the scan results for each
                // partition. The second accumulates the result from left to
                // right to be used by the third step--which operates on the
                // same partitions the first step operated on.

                using hpx::util::get;
                using hpx::util::make_zip_iterator;

                auto f3 =
                    [op, policy](
                        zip_iterator part_begin, std::size_t part_size,
                        hpx::shared_future<T> curr, hpx::shared_future<T> next
                    )
                    {
                        next.get();     // rethrow exceptions

                        T val = curr.get();
                        OutIter dst = get<1>(part_begin.get_iterator_tuple());

                        // MSVC 2015 fails if op is captured by reference
                        util::loop_n<ExPolicy>(
                            dst, part_size,
                            [=, &val](OutIter it)
                            {
                                *it = hpx::util::invoke(op, val, *it);
                            });
                    };

                return util::scan_partitioner<ExPolicy, OutIter, T>::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first, dest), count, init,
                    // step 1 performs first part of scan algorithm
                    [op](zip_iterator part_begin, std::size_t part_size) -> T
                    {
                        T part_init = get<0>(*part_begin);
                        get<1>(*part_begin++) = part_init;
                        auto iters = part_begin.get_iterator_tuple();
                        return sequential_inclusive_scan_n(
                            get<0>(iters),
                            part_size-1,
                            get<1>(iters),
                            part_init, op);
                    },
                    // step 2 propagates the partition results from left
                    // to right
                    hpx::util::unwrapped(op),
                    // step 3 runs final accumulation on each partition
                    std::move(f3),
                    // step 4 use this return value
                    [final_dest](std::vector<hpx::shared_future<T> > &&,
                        std::vector<hpx::future<void> > &&)
                    {
                        return final_dest;
                    });
            }
        };

        template <typename ExPolicy, typename InIter, typename OutIter, typename T,
            typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        inclusive_scan_(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
            T const& init, Op && op, std::false_type) {

            typedef std::integral_constant<bool,
                    parallel::is_sequential_execution_policy<ExPolicy>::value ||
                   !hpx::traits::is_forward_iterator<InIter>::value ||
                   !hpx::traits::is_forward_iterator<OutIter>::value
                > is_seq;

            return inclusive_scan<OutIter>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest, init, std::forward<Op>(op));
        }

        // forward declare the segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename OutIter, typename T,
            typename Op>
        static typename util::detail::algorithm_result<ExPolicy, OutIter>::type
        inclusive_scan_(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
            T const& init, Op && op, std::true_type);
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(op, init, *first, ..., *(first + (i - result))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
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
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<OutIter> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a inclusive_scan algorithm returns the output iterator
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
    template <typename ExPolicy, typename InIter, typename OutIter, typename T,
        typename Op>
    inline typename std::enable_if<
        is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    inclusive_scan(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
        T init, Op && op)
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_forward_iterator<OutIter>::value),
            "Requires at least output iterator.");

        typedef hpx::traits::is_segmented_iterator<InIter> is_segmented;

        return detail::inclusive_scan_(
            std::forward<ExPolicy>(policy), first, last, dest,
            init, std::forward<Op>(op),
            is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, init, *first, ..., *(first + (i - result))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
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
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<OutIter> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a inclusive_scan algorithm returns the output iterator
    ///           to the element in the destination range, one past the last
    ///           element copied.
    ///
    /// \note   GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aN) is defined as:
    ///         * a1 when N is 1
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK)
    ///           + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter, typename T>
    inline typename std::enable_if<
        is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    inclusive_scan(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
        T init)
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_forward_iterator<OutIter>::value),
            "Requires at least output iterator.");

        typedef hpx::traits::is_segmented_iterator<InIter> is_segmented;

        return detail::inclusive_scan_(
            std::forward<ExPolicy>(policy), first, last, dest,
            init, std::plus<T>(),
            is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// gENERALIZED_NONCOMMUTATIVE_SUM(+, *first, ..., *(first + (i - result))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a op.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<OutIter> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a inclusive_scan algorithm returns the output iterator
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
    template <typename ExPolicy, typename InIter, typename OutIter>
    inline typename std::enable_if<
        is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    inclusive_scan(ExPolicy&& policy, InIter first, InIter last, OutIter dest)
    {
        static_assert(
            (hpx::traits::is_input_iterator<InIter>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_forward_iterator<OutIter>::value),
            "Requires at least output iterator.");

        typedef typename std::iterator_traits<InIter>::value_type value_type;

        typedef hpx::traits::is_segmented_iterator<InIter> is_segmented;

        return detail::inclusive_scan_(
            std::forward<ExPolicy>(policy), first, last, dest,
            value_type(), std::plus<value_type>(),
            is_segmented());
    }
}}}

#endif
