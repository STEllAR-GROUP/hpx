//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/detail/transform_reduce.hpp

#if !defined(HPX_PARALLEL_DETAIL_TRANSFORM_REDUCE_JUL_11_2014_0428PM)
#define HPX_PARALLEL_DETAIL_TRANSFORM_REDUCE_JUL_11_2014_0428PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/unwrapped.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/dispatch.hpp>
#include <hpx/parallel/detail/predicates.hpp>
#include <hpx/parallel/util/partitioner.hpp>
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
    // transform_reduce
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct transform_reduce
          : public detail::algorithm<transform_reduce<T>, T>
        {
            transform_reduce()
              : transform_reduce::algorithm("transform_reduce")
            {}

            template <typename ExPolicy, typename InIter, typename Reduce,
                typename Convert>
            static T
            sequential(ExPolicy const&, InIter first, InIter last,
                T && init, Reduce && r, Convert && conv)
            {
                typedef typename std::iterator_traits<InIter>::value_type
                    value_type;

                return std::accumulate(first, last, std::forward<T>(init),
                    [&r, &conv](T const& res, value_type const& next)
                    {
                        return r(res, conv(next));
                    });
            }

            template <typename ExPolicy, typename FwdIter, typename Reduce,
                typename Convert>
            static typename detail::algorithm_result<ExPolicy, T>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last,
                T && init, Reduce && r, Convert && conv)
            {
                if (first == last)
                {
                    return detail::algorithm_result<ExPolicy, T>::get(
                        std::forward<T>(init));
                }

                typedef typename std::iterator_traits<FwdIter>::reference
                    reference;

                return util::partitioner<ExPolicy, T>::call(
                    policy, first, std::distance(first, last),
                    [r, conv](FwdIter part_begin, std::size_t part_size)
                    {
                        T val = conv(*part_begin);
                        return util::accumulate_n(++part_begin, --part_size,
                            std::move(val),
                            [&r, &conv](T const& res, reference next)
                            {
                                return r(res, conv(next));
                            });
                    },
                    hpx::util::unwrapped([init, r](std::vector<T> && results)
                    {
                        return util::accumulate_n(boost::begin(results),
                            boost::size(results), init, r);
                    }));
            }
        };
        /// \endcond
    }

    /// Returns GENERALIZED_SUM(red_op, init, conv_op(*first), ..., conv_op(*(first + (last - first) - 1))).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicates \a red_op and \a conv_op.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
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
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///                     The type \a R must be such that an object of this
    ///                     type can be implicitly converted to \a T.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a transform_reduce algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a transform_reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type \a task_execution_policy and
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
    template <typename ExPolicy, typename InIter, typename T, typename Reduce,
        typename Convert>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, T>::type
    >::type
    transform_reduce(ExPolicy&& policy, InIter first, InIter last, T init,
        Reduce && red_op, Convert && conv_op)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::input_iterator_tag, iterator_category>::value),
            "Requires at least input iterator.");

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::transform_reduce<T>().call(
            std::forward<ExPolicy>(policy), first, last, std::move(init),
            std::forward<Reduce>(red_op), std::forward<Convert>(conv_op),
            is_seq());
    }
}}}

#endif
