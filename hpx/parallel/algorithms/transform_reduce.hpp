//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/transform_reduce.hpp

#if !defined(HPX_PARALLEL_DETAIL_TRANSFORM_REDUCE_JUL_11_2014_0428PM)
#define HPX_PARALLEL_DETAIL_TRANSFORM_REDUCE_JUL_11_2014_0428PM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/unwrapped.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <boost/range/functions.hpp>

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

            template <typename ExPolicy, typename InIter, typename T_,
                typename Reduce, typename Convert>
            static T
            sequential(ExPolicy, InIter first, InIter last,
                T_ && init, Reduce && r, Convert && conv)
            {
                typedef typename std::iterator_traits<InIter>::value_type
                    value_type;

                return std::accumulate(first, last, std::forward<T_>(init),
                    [&r, &conv](T const& res, value_type const& next)
                    {
                        return r(res, conv(next));
                    });
            }

            template <typename ExPolicy, typename FwdIter, typename T_,
                typename Reduce, typename Convert>
            static typename util::detail::algorithm_result<ExPolicy, T>::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                T_ && init, Reduce && r, Convert && conv)
            {
                if (first == last)
                {
                    T init_ = init;
                    return util::detail::algorithm_result<ExPolicy, T>::get(
                        std::move(init_));
                }

                typedef typename std::iterator_traits<FwdIter>::reference
                    reference;

                return util::partitioner<ExPolicy, T>::call(
                    std::forward<ExPolicy>(policy),
                    first, std::distance(first, last),
                    [r, conv](FwdIter part_begin, std::size_t part_size) -> T
                    {
                        T val = conv(*part_begin);
                        return util::accumulate_n(++part_begin, --part_size,
                            std::move(val),
                            // MSVC14 bails out if r and conv are captured by
                            // reference
                            [=](T const& res, reference next)
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

        template <typename ExPolicy, typename InIter, typename T,
            typename Reduce, typename Convert>
        inline typename util::detail::algorithm_result<
            ExPolicy, typename hpx::util::decay<T>::type
        >::type
        transform_reduce_(ExPolicy && policy, InIter first, InIter last,
            T && init, Reduce && red_op, Convert && conv_op, std::false_type)
        {
            static_assert(
                (hpx::traits::is_input_iterator<InIter>::value),
                "Requires at least input iterator.");

            typedef std::integral_constant<bool,
                    parallel::execution::is_sequential_execution_policy<
                        ExPolicy
                    >::value ||
                   !hpx::traits::is_forward_iterator<InIter>::value
                > is_seq;

            typedef typename hpx::util::decay<T>::type init_type;

            return transform_reduce<init_type>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, std::forward<T>(init),
                std::forward<Reduce>(red_op), std::forward<Convert>(conv_op));
        }

        // forward declare the segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename T,
            typename Reduce, typename Convert>
        typename util::detail::algorithm_result<
            ExPolicy, typename hpx::util::decay<T>::type
        >::type
        transform_reduce_(ExPolicy && policy, InIter first, InIter last,
            T && init, Reduce && red_op, Convert && conv_op, std::true_type);

        /// \endcond
    }

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
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
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
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
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
    template <typename ExPolicy, typename InIter, typename T, typename Reduce,
        typename Convert>
    inline typename std::enable_if<
        execution::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, T>::type
    >::type
    transform_reduce(ExPolicy&& policy, InIter first, InIter last,
        T init, Convert && conv_op, Reduce && red_op)
    {
        typedef hpx::traits::is_segmented_iterator<InIter> is_segmented;

        return detail::transform_reduce_(
            std::forward<ExPolicy>(policy), first, last, std::move(init),
            std::forward<Reduce>(red_op), std::forward<Convert>(conv_op),
            is_segmented());
    }
}}}

#endif
