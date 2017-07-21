//  Copyright (c) 2014-2017 Hartmut Kaiser
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
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/util/unused.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // inclusive_scan
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        // Our own version of the sequential inclusive_scan.
        template <typename InIter, typename OutIter, typename T, typename Op,
            typename Conv = util::projection_identity>
        OutIter sequential_inclusive_scan(InIter first, InIter last,
            OutIter dest, T init, Op && op, Conv && conv = Conv())
        {
            for (/* */; first != last; (void) ++first, ++dest)
            {
                init = hpx::util::invoke(op, init, hpx::util::invoke(conv, *first));
                *dest = init;
            }
            return dest;
        }

        template <typename InIter, typename OutIter, typename T, typename Op,
            typename Conv =util::projection_identity>
        T sequential_inclusive_scan_n(InIter first, std::size_t count,
            OutIter dest, T init, Op && op, Conv && conv = Conv())
        {
            for (/* */; count-- != 0; (void) ++first, ++dest)
            {
                init = hpx::util::invoke(op, init, hpx::util::invoke(conv, *first));
                *dest = init;
            }
            return init;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter2>
        struct inclusive_scan
          : public detail::algorithm<inclusive_scan<FwdIter2>, FwdIter2>
        {
            inclusive_scan()
              : inclusive_scan::algorithm("inclusive_scan")
            {}

            template <typename ExPolicy, typename InIter, typename OutIter,
                typename T, typename Op, typename Conv = util::projection_identity>
            static OutIter
            sequential(ExPolicy, InIter first, InIter last, OutIter dest,
                T const& init, Op && op, Conv && conv = Conv())
            {
                return sequential_inclusive_scan(first, last, dest,
                    init, std::forward<Op>(op), std::forward<Conv>(conv));
            }

            template <typename ExPolicy, typename FwdIter1, typename T, typename Op,
                typename Conv = util::projection_identity>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter2
            >::type
            parallel(ExPolicy && policy, FwdIter1 first, FwdIter1 last,
                 FwdIter2 dest, T const& init, Op && op, Conv && conv = Conv())
            {
                typedef util::detail::algorithm_result<ExPolicy, FwdIter2>
                    result;
                typedef hpx::util::zip_iterator<FwdIter1, FwdIter2> zip_iterator;
                typedef typename std::iterator_traits<FwdIter1>::difference_type
                    difference_type;

                if (first == last)
                    return result::get(std::move(dest));

                difference_type count = std::distance(first, last);

                FwdIter2 final_dest = dest;
                std::advance(final_dest, count);

                // The overall scan algorithm is performed by executing 3
                // steps. The first calculates the scan results for each
                // partition. The second accumulates the result from left to
                // right to be used by the third step--which operates on the
                // same partitions the first step operated on.

                using hpx::util::get;
                using hpx::util::make_zip_iterator;

                auto f3 =
                    [op, conv, policy](
                        zip_iterator part_begin, std::size_t part_size,
                        hpx::shared_future<T> curr, hpx::shared_future<T> next
                    )
                    {
                        HPX_UNUSED(policy);

                        next.get();     // rethrow exceptions

                        T val = curr.get();
                        FwdIter2 dst = get<1>(part_begin.get_iterator_tuple());

                        // MSVC 2015 fails if op is captured by reference
                        util::loop_n<ExPolicy>(
                            dst, part_size,
                            [=, &val](FwdIter2 it)
                            {
                                *it = hpx::util::invoke(op, val, *it);
                            });
                    };

                return util::scan_partitioner<ExPolicy, FwdIter2, T>::call(
                    std::forward<ExPolicy>(policy),
                    make_zip_iterator(first, dest), count, init,
                    // step 1 performs first part of scan algorithm
                    [op, conv](zip_iterator part_begin, std::size_t part_size) -> T
                    {
                        T part_init = hpx::util::invoke(conv, get<0>(*part_begin));
                        get<1>(*part_begin++) = part_init;
                        auto iters = part_begin.get_iterator_tuple();
                        return sequential_inclusive_scan_n(
                            get<0>(iters),
                            part_size-1,
                            get<1>(iters),
                            part_init, op, conv);
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

        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename T, typename Op, typename Conv = util::projection_identity>
        static typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        inclusive_scan_(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest,
            T const& init, Op && op, std::false_type, Conv && conv = Conv()) {

#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
            typedef std::integral_constant<bool,
                    parallel::execution::is_sequenced_execution_policy<
                        ExPolicy
                    >::value ||
                   !hpx::traits::is_forward_iterator<FwdIter1>::value ||
                   !hpx::traits::is_forward_iterator<FwdIter2>::value
                > is_seq;
#else
            typedef parallel::execution::is_sequenced_execution_policy<
                        ExPolicy
                    > is_seq;
#endif

            return inclusive_scan<FwdIter2>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, dest, init, std::forward<Op>(op));
        }

        // forward declare the segmented version of this algorithm
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename T, typename Op, typename Conv = util::projection_identity>
        static typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
        inclusive_scan_(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 dest, T const& init, Op && op, std::true_type,
            Conv && conv = Conv());
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
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<FwdIter2> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
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
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename Op,
        typename T,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter1>::value &&
        hpx::traits::is_iterator<FwdIter2>::value &&
        hpx::traits::is_invocable<Op,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter1>::value_type
            >::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    inclusive_scan(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest,
        Op && op, T init)
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least output iterator.");
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");
#endif

        typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;

        return detail::inclusive_scan_(
            std::forward<ExPolicy>(policy), first, last, dest,
            init, std::forward<Op>(op),
            is_segmented());
    }

#if defined(HPX_HAVE_INCLUSIVE_SCAN_COMPATIBILITY)
    /// \cond NOINTERNAL
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename T,
        typename Op,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter1>::value &&
        hpx::traits::is_iterator<FwdIter2>::value &&
        hpx::traits::is_invocable<Op,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter1>::value_type
            >::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    inclusive_scan(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest,
        T init, Op && op)
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least output iterator.");
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");
#endif

        typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;

        return detail::inclusive_scan_(
            std::forward<ExPolicy>(policy), first, last, dest,
            init, std::forward<Op>(op),
            is_segmented());
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, init, *first, ..., *(first + (i - result))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a +.
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
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<FwdIter2> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
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
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename T,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter1>::value &&
        hpx::traits::is_iterator<FwdIter2>::value &&
       !hpx::traits::is_invocable<T,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter1>::value_type
            >::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    inclusive_scan(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest,
        T init)
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least output iterator.");
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");
#endif

        typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;

        return detail::inclusive_scan_(
            std::forward<ExPolicy>(policy), first, last, dest,
            init, std::plus<T>(),
            is_segmented());
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(op, *first, ..., *(first + (i - result))).
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
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<FwdIter2> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
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
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2, typename Op,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter1>::value &&
        hpx::traits::is_iterator<FwdIter2>::value &&
        hpx::traits::is_invocable<Op,
                typename std::iterator_traits<FwdIter1>::value_type,
                typename std::iterator_traits<FwdIter1>::value_type
            >::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    inclusive_scan(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest,
        Op && op)
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least output iterator.");
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");
#endif

        typedef typename std::iterator_traits<FwdIter1>::value_type value_type;
        typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;

        return detail::inclusive_scan_(
            std::forward<ExPolicy>(policy), first, last, dest,
            value_type(), std::forward<Op>(op),
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
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a inclusive_scan algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<FwdIter2> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
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
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter2>::type
    >::type
    inclusive_scan(ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest)
    {
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
        static_assert(
            (hpx::traits::is_input_iterator<FwdIter1>::value),
            "Requires at least input iterator.");
        static_assert(
            (hpx::traits::is_output_iterator<FwdIter2>::value ||
                hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least output iterator.");
#else
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");
#endif

        typedef typename std::iterator_traits<FwdIter1>::value_type value_type;

        typedef hpx::traits::is_segmented_iterator<FwdIter1> is_segmented;

        return detail::inclusive_scan_(
            std::forward<ExPolicy>(policy), first, last, dest,
            value_type(), std::plus<value_type>(),
            is_segmented());
    }
}}}

#endif
