//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file reduce.hpp

#if !defined(HPX_PARALLEL_DETAILREUCE_JUN_01_2014_0903AM)
#define HPX_PARALLEL_DETAILREUCE_JUN_01_2014_0903AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/unwrapped.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <numeric>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel
{
    ///////////////////////////////////////////////////////////////////////////
    // reduce
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename InIter, typename T, typename Pred>
        typename detail::algorithm_result<ExPolicy, T>::type
        reduce(ExPolicy const&, InIter first, InIter last, T && init,
            Pred && op, boost::mpl::true_)
        {
            try {
                detail::synchronize(first, last);
                return detail::algorithm_result<ExPolicy, T>::get(
                    std::accumulate(first, last, std::forward<T>(init),
                        std::forward<Pred>(op)));
            }
            catch (...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename FwdIter, typename T, typename Pred>
        typename detail::algorithm_result<ExPolicy, T>::type
        reduce(ExPolicy const& policy, FwdIter first, FwdIter last, T && init,
            Pred && op, boost::mpl::false_)
        {
            if (first == last)
            {
                return detail::algorithm_result<ExPolicy, T>::get(
                    std::forward<T>(init));
            }

            typedef typename std::iterator_traits<FwdIter>::iterator_category
                category;

            return util::partitioner<ExPolicy, T>::call(
                policy, first, std::distance(first, last),
                [op](FwdIter part_begin, std::size_t part_count)
                {
                    T val = *part_begin;
                    return util::accumulate_n(++part_begin, --part_count,
                        std::move(val), op);
                },
                hpx::util::unwrapped([init, op](std::vector<T>&& results)
                {
                    return util::accumulate_n(boost::begin(results),
                        boost::size(results), init, op);
                }));
        }

        template <typename InIter, typename T, typename Pred>
        T reduce(execution_policy const& policy, InIter first, InIter last,
            T && init, Pred && op, boost::mpl::false_)
        {
            HPX_PARALLEL_DISPATCH(policy, detail::reduce, first, last,
                std::forward<T>(init), std::forward<Pred>(op));
        }

        template<typename InIter, typename T, typename Pred>
        T reduce(execution_policy const& policy, InIter first, InIter last,
            T init, Pred && op, boost::mpl::true_ t)
        {
            return detail::reduce(sequential_execution_policy(),
                first, last, std::forward<T>(init), std::forward<Pred>(op), t);
        }
        /// \endcond
    }

    /// Returns GENERALIZED_SUM(f, init, *first, ..., *(first + last - first - 1)).
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
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
    ///                     requirements of \a CopyConstructible, but not
    ///                     \a MoveConstructible.
    /// \tparam T           The type of the value to be assigned (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).This is an
    ///                     unary predicate which returns \a true for the
    ///                     required elements. The signature of this predicate
    ///                     should be equivalent to:
    /// \param init         The value to be assigned.
    ///
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode
    ///
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// \note The assignments in the parallel \a copy_if algorithm invoked with
    ///       an execution policy object of type \a sequential_execution_policy
    ///       execute in sequential order in the calling thread.
    /// \note The assignments in the parallel \a copy_if algorithm invoked with
    ///       an execution policy object of type \a parallel_execution_policy or
    ///       \a task_execution_policy are permitted to execute in an unordered
    ///       fashion in unspecified threads, and indeterminately sequenced
    ///       within each thread.
    ///
    /// \returns  The \a copy_if algorithm returns a \a hpx::future<OutIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a OutIter otherwise.
    /// \returns  The \a copy_if algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    /// \note   GENERALIZED_SUM(op, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(op, b1, ..., bM), GENERALIZED_SUM(op, bM, ..., bN))
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 0 < M < N.
    ///
    template <typename ExPolicy, typename InIter, typename T, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, T>::type
    >::type
    reduce(ExPolicy&& policy, InIter first, InIter last, T init, F && f)
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

        return detail::reduce(std::forward<ExPolicy>(policy), first, last,
            std::move(init), std::forward<F>(f), is_seq());
    }

    template <typename ExPolicy, typename InIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, T>::type
    >::type
    reduce(ExPolicy&& policy, InIter first, InIter last, T init)
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

        return detail::reduce(std::forward<ExPolicy>(policy), first, last,
            std::move(init), std::plus<T>(), is_seq());
    }

#if !defined(BOOST_NO_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS)
    template <typename ExPolicy, typename InIter,
        typename T = typename std::iterator_traits<InIter>::value_type>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, T>::type
    >::type
    reduce(ExPolicy&& policy, InIter first, InIter last)
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

        return detail::reduce(std::forward<ExPolicy>(policy), first, last,
            T(), std::plus<T>(), is_seq());
    }
#endif
}}

#endif
