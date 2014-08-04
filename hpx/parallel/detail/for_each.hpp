//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/detail/for_each.hpp

#if !defined(HPX_PARALLEL_DETAIL_FOR_EACH_MAY_29_2014_0932PM)
#define HPX_PARALLEL_DETAIL_FOR_EACH_MAY_29_2014_0932PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/util/move.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/dispatch.hpp>
#include <hpx/parallel/detail/is_negative.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // for_each_n
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Iter>
        struct for_each_n : public detail::algorithm<for_each_n<Iter>, Iter>
        {
            for_each_n()
              : for_each_n::algorithm("for_each_n")
            {}

            template <typename ExPolicy, typename F>
            static Iter
            sequential(ExPolicy const&, Iter first, std::size_t count, F && f)
            {
                return util::loop_n(first, count, std::forward<F>(f));
            }

            template <typename ExPolicy, typename F>
            static typename detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy const& policy, Iter first, std::size_t count,
                F && f)
            {
                if (count != 0)
                {
                    return util::foreach_n_partitioner<ExPolicy>::call(
                        policy, first, count,
                        [f](Iter part_begin, std::size_t part_size)
                        {
                            util::loop_n(part_begin, part_size, f);
                        });
                }

                return detail::algorithm_result<ExPolicy, Iter>::get(
                    std::move(first));
            }
        };
        /// \endcond
    }

    /// Applies \a f to the result of dereferencing every iterator in the range
    /// [first, first + count), starting from first and proceeding to
    /// first + count - 1.
    ///
    /// \note   Complexity: Applies \a f exactly \a count times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a for_each_n algorithm returns a
    ///           \a hpx::future<InIter> if the execution policy is of
    ///           type \a task_execution_policy and returns \a InIter
    ///           otherwise.
    ///           It returns \a first + \a count for non-negative values of
    ///           \a count and \a first for negative values.
    ///
    template <typename ExPolicy, typename InIter, typename Size, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, InIter>::type
    >::type
    for_each_n(ExPolicy && policy, InIter first, Size count, F && f)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<std::input_iterator_tag, iterator_category>::value),
            "Requires at least input iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative<Size>::call(count))
        {
            return detail::algorithm_result<ExPolicy, InIter>::get(
                std::move(first));
        }

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::input_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::for_each_n<InIter>().call(
            std::forward<ExPolicy>(policy),
            first, std::size_t(count), std::forward<F>(f), is_seq());
    }

    ///////////////////////////////////////////////////////////////////////////
    // for_each
    namespace detail
    {
        /// \cond NOINTERNAL
        struct for_each : public detail::algorithm<for_each>
        {
            for_each()
              : for_each::algorithm("for_each")
            {}

            template <typename ExPolicy, typename InIter, typename F>
            static hpx::util::unused_type
            sequential(ExPolicy const&, InIter first, InIter last, F && f)
            {
                std::for_each(first, last, std::forward<F>(f));
                return hpx::util::unused;
            }

            template <typename ExPolicy, typename FwdIter, typename F>
            static typename detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy const& policy, FwdIter first, FwdIter last, F && f)
            {
                typedef
                    typename detail::algorithm_result<ExPolicy>::type
                result_type;

                return hpx::util::void_guard<result_type>(),
                    detail::for_each_n<FwdIter>().call(
                        policy, first, std::distance(first, last),
                        std::forward<F>(f), boost::mpl::false_());
            }
        };
        /// \endcond
    }

    /// Applies \a f to the result of dereferencing every iterator in the
    /// range [first, last).
    ///
    /// \note   Complexity: Applies \a f exactly \a last - \a first times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a for_each algorithm returns a \a hpx::future<void> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a void otherwise.
    ///
    template <typename ExPolicy, typename InIter, typename F>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, void>::type
    >::type
    for_each(ExPolicy && policy, InIter first, InIter last, F && f)
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

        return detail::for_each().call(
            std::forward<ExPolicy>(policy),
            first, last, std::forward<F>(f), is_seq());
    }
}}}

#endif
