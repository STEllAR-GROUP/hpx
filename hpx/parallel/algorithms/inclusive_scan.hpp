//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/inclusive_scan.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_INCLUSIVE_SCAN_JAN_03_2015_0136PM)
#define HPX_PARALLEL_ALGORITHM_INCLUSIVE_SCAN_JAN_03_2015_0136PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/algorithm_result.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/scan_partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <numeric>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

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
            for (/**/; first != last; (void) ++first, ++dest)
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
            for (/**/; count-- != 0; (void) ++first, ++dest)
            {
                init = op(init, *first);
                *dest = init;
            }
            return init;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename T, typename OutIter, typename Op>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        inclusive_scan_helper(ExPolicy const& policy,
            std::vector<hpx::shared_future<T> >&& r,
            boost::shared_array<T> data, std::size_t count,
            OutIter dest, Op && op, std::vector<std::size_t> const& chunk_sizes)
        {
            typedef hpx::util::zip_iterator<T*, OutIter> zip_iterator;
            typedef typename zip_iterator::reference reference;

            using hpx::util::make_zip_iterator;
            return
                util::partitioner<ExPolicy, OutIter, void>::call_with_data(
                    policy, make_zip_iterator(data.get(), dest), count,
                    [=](hpx::shared_future<T>&& val,
                        zip_iterator part_begin, std::size_t part_size)
                    {
                        T const& v = val.get();
                        parallel::util::loop_n(part_begin, part_size,
                            [&](zip_iterator d)
                            {
                                using hpx::util::get;
                                *get<1>(d.get_iterator_tuple()) =
                                    op(*get<0>(d.get_iterator_tuple()), v);
                            });
                    },
                    [dest, count, data](
                        std::vector<future<void> > && r) mutable -> OutIter
                    {
                        std::advance(dest, count);
                        return dest;
                    },
                    chunk_sizes, std::move(r)
                );
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
            sequential(ExPolicy const&, InIter first, InIter last,
                OutIter dest, T && init, Op && op)
            {
                return sequential_inclusive_scan(first, last, dest,
                    std::forward<T>(init), std::forward<Op>(op));
            }

            template <typename ExPolicy, typename FwdIter, typename T, typename Op>
            static typename detail::algorithm_result<ExPolicy, OutIter>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last,
                 OutIter dest, T && init, Op && op)
            {
                typedef detail::algorithm_result<ExPolicy, OutIter> result;
                typedef hpx::util::zip_iterator<FwdIter, T*> zip_iterator;

                if (first == last)
                    return result::get(std::move(dest));

                std::size_t count = std::distance(first, last);
                boost::shared_array<T> data(new T[count]);

                // The overall scan algorithm is performed by executing 2
                // subsequent parallel steps. The first calculates the scan
                // results for each partition and the second produces the
                // overall result

                using hpx::util::make_zip_iterator;
                return
                    util::scan_partitioner<ExPolicy, OutIter, T>::call(
                        policy, make_zip_iterator(first, data.get()), count, init,
                        // step 1 performs first part of scan algorithm
                        [=](zip_iterator part_begin, std::size_t part_size) -> T
                        {
                            using hpx::util::get;
                            return sequential_inclusive_scan_n(
                                get<0>(part_begin.get_iterator_tuple()), part_size,
                                get<1>(part_begin.get_iterator_tuple()), init, op);
                        },
                        // step 2 propagates the partition results from left
                        // to right
                        hpx::util::unwrapped(
                            [=](T const& prev, T const& curr) -> T
                            {
                                return op(prev, curr);
                            }),
                        // step 3 runs the remaining operation
                        [=](std::vector<hpx::shared_future<T> >&& r,
                            std::vector<std::size_t> const& chunk_sizes)
                        {
                            // run the final copy step and produce the required
                            // result
                            return inclusive_scan_helper(policy, std::move(r),
                                data, count, dest, op, chunk_sizes);
                        }
                    );
            }
        };
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
    ///         * op(GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK), GENERALIZED_NONCOMMUTATIVE_SUM(op, aM, ..., aN))
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum. If
    /// \a op is not mathematically associative, the behavior of
    /// \a inclusive_scan may be non-deterministic.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter, typename T,
        typename Op>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    inclusive_scan(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
        T init, Op && op)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::input_iterator_tag, iterator_category>::value),
            "Requires at least input iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

        return detail::inclusive_scan<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest, std::move(init), std::forward<Op>(op));
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
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(op, a1, ..., aK) + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    inclusive_scan(ExPolicy&& policy, InIter first, InIter last, OutIter dest,
        T init)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::input_iterator_tag, iterator_category>::value),
            "Requires at least input iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

        return detail::inclusive_scan<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest, std::move(init), std::plus<T>());
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Assigns through each iterator \a i in [result, result + (last - first))
    /// the value of
    /// GENERALIZED_NONCOMMUTATIVE_SUM(+, *first, ..., *(first + (i - result))).
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
    ///         * GENERALIZED_NONCOMMUTATIVE_SUM(+, a1, ..., aK) + GENERALIZED_NONCOMMUTATIVE_SUM(+, aM, ..., aN)
    ///           where 1 < K+1 = M <= N.
    ///
    /// The difference between \a exclusive_scan and \a inclusive_scan is that
    /// \a inclusive_scan includes the ith input element in the ith sum.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    inclusive_scan(ExPolicy&& policy, InIter first, InIter last, OutIter dest)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;
        typedef typename std::iterator_traits<OutIter>::iterator_category
            output_iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::input_iterator_tag, iterator_category>::value),
            "Requires at least input iterator.");

        BOOST_STATIC_ASSERT_MSG(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, output_iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, output_iterator_category>
            >::value),
            "Requires at least output iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>,
            boost::is_same<std::output_iterator_tag, output_iterator_category>
        >::type is_seq;

        typedef typename std::iterator_traits<InIter>::value_type value_type;

        return detail::inclusive_scan<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest, value_type(), std::plus<value_type>());
    }
}}}

#endif
