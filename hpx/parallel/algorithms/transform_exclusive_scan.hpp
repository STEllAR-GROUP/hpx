//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform_exclusive_scan.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_TRANSFORM_EXCLUSIVE_SCAN_JAN_04_2015_0354PM)
#define HPX_PARALLEL_ALGORITHM_TRANSFORM_EXCLUSIVE_SCAN_JAN_04_2015_0354PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/transform_inclusive_scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
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
    // transform_exclusive_scan
    namespace detail
    {
        /// \cond NOINTERNAL

        // Our own version of the sequential transform_exclusive_scan.
        template <typename InIter, typename OutIter, typename Conv, typename T,
            typename Op>
        OutIter sequential_transform_exclusive_scan(InIter first, InIter last,
            OutIter dest, Conv && conv, T init, Op && op)
        {
            T temp = init;
            for (/* */; first != last; (void) ++first, ++dest)
            {
                init = op(init, conv(*first));
                *dest = temp;
                temp = init;
            }
            return dest;
        }

        template <typename InIter, typename OutIter, typename Conv, typename T,
            typename Op>
        T sequential_transform_exclusive_scan_n(InIter first, std::size_t count,
            OutIter dest, Conv && conv, T init, Op && op)
        {
            T temp = init;
            for (/* */; count-- != 0; (void) ++first, ++dest)
            {
                init  = op(init, conv(*first));
                *dest = temp;
                temp  = init;
            }
            return init;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename OutIter>
        struct transform_exclusive_scan
          : public detail::algorithm<transform_exclusive_scan<OutIter>, OutIter>
        {
            transform_exclusive_scan()
              : transform_exclusive_scan::algorithm("transform_exclusive_scan")
            {}

            template <typename ExPolicy, typename InIter, typename Conv,
                typename T, typename Op>
            static OutIter
            sequential(ExPolicy, InIter first, InIter last,
                OutIter dest, Conv && conv, T && init, Op && op)
            {
                return sequential_transform_exclusive_scan(first, last, dest,
                    std::forward<Conv>(conv), std::forward<T>(init),
                    std::forward<Op>(op));
            }

            template <typename ExPolicy, typename FwdIter, typename Conv,
                typename T, typename Op>
            static typename util::detail::algorithm_result<
                ExPolicy, OutIter
            >::type
            parallel(ExPolicy policy, FwdIter first, FwdIter last,
                 OutIter dest, Conv && conv, T && init, Op && op)
            {
                typedef util::detail::algorithm_result<ExPolicy, OutIter> result;
                typedef hpx::util::zip_iterator<FwdIter, OutIter> zip_iterator;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;

                if (first == last)
                    return result::get(std::move(dest));

                difference_type count = std::distance(first, last);

                OutIter final_dest = dest;
                std::advance(final_dest, count);

                // The overall scan algorithm is performed by executing 2
                // subsequent parallel steps. The first calculates the scan
                // results for each partition and the second produces the
                // overall result

                using hpx::util::get;
                using hpx::util::make_zip_iterator;
                return util::scan_partitioner<ExPolicy, OutIter, T>::call(
                    policy, make_zip_iterator(first, dest), count, init,
                    // step 1 performs first part of scan algorithm
                    [op, conv](zip_iterator part_begin, std::size_t part_size) -> T
                    {
                        T part_init = conv(get<0>(*part_begin++));
                        return sequential_transform_exclusive_scan_n(
                            get<0>(part_begin.get_iterator_tuple()),
                            part_size - 1,
                            get<1>(part_begin.get_iterator_tuple()),
                            conv, part_init, op);
                    },
                    // step 2 propagates the partition results from left
                    // to right
                    hpx::util::unwrapped(op),
                    // step 3 runs final_accumulation on each partition
                    [op](zip_iterator part_begin, std::size_t part_size,
                        hpx::shared_future<T> f_accu)
                    {
                        T val = f_accu.get();
                        OutIter dst = get<1>(part_begin.get_iterator_tuple());
                        *dst++ = val;
                        util::loop_n(dst, part_size - 1,
                            [&op, &val](OutIter it)
                            {
                                *it = op(*it, val);
                            });
                    },
                    // use this return value
                    [final_dest](std::vector<hpx::shared_future<T> > &&,
                        std::vector<hpx::future<void> > &&)
                    {
                        return final_dest;
                    });
            }
        };
        /// \endcond
    }

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
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
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
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
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
    ///                     type \a InIter can be dereferenced and then
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
    /// The reduce operations in the parallel \a transform_exclusive_scan algorithm
    /// invoked with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a transform_exclusive_scan algorithm
    /// invoked with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a copy_n algorithm returns a \a hpx::future<OutIter> if
    ///           the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a OutIter otherwise.
    ///           The \a transform_exclusive_scan algorithm returns the output
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
    /// modify elements in the ranges [first,last) or [result,result + (last - first)).
    ///
    /// The behavior of transform_exclusive_scan may be non-deterministic for
    /// a non-associative predicate.
    ///
    template <typename ExPolicy, typename InIter, typename OutIter,
        typename Conv, typename T, typename Op>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    transform_exclusive_scan(ExPolicy&& policy, InIter first, InIter last,
        OutIter dest, Conv && conv, T init, Op && op)
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

        return detail::transform_exclusive_scan<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, dest, std::forward<Conv>(conv), std::move(init),
            std::forward<Op>(op));
    }
}}}

#endif
