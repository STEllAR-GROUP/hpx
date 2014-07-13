//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file fill.hpp

#if !defined(HPX_PARALLEL_DETAIL_FILL_JUNE_12_2014_0405PM)
#define HPX_PARALLEL_DETAIL_FILL_JUNE_12_2014_0405PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/detail/algorithm_result.hpp>
#include <hpx/parallel/detail/for_each.hpp>
#include <hpx/parallel/detail/is_negative.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/util/move.hpp>

#include <algorithm>
#include <iterator>

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // fill
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename InIter, typename T>
        typename detail::algorithm_result<ExPolicy, void>::type
        fill(ExPolicy const&, InIter first, InIter last, T val,
            boost::mpl::true_)
        {
            try {
                std::fill(first, last, val);
                return detail::algorithm_result<ExPolicy, void>::get();
            }
            catch(...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename FwdIter, typename T>
        typename detail::algorithm_result<ExPolicy, void>::type
        fill(ExPolicy const& policy, FwdIter first, FwdIter last, T val,
            boost::mpl::false_ f)
        {
            typedef typename detail::algorithm_result<ExPolicy, void>::type
                result_type;
            typedef typename std::iterator_traits<FwdIter>::value_type type;

            return hpx::util::void_guard<result_type>(),
                for_each_n(policy, first,
                    std::distance(first, last),
                    [val](type& v){
                        v = val;
                    }, f);
        }

        template <typename InIter, typename T>
        void fill(execution_policy const& policy,
            InIter first, InIter last, T val, boost::mpl::false_)
        {
            HPX_PARALLEL_DISPATCH(policy, detail::fill, first, last, val);
        }

       template <typename InIter, typename T>
        void fill(execution_policy const& policy,
            InIter first, InIter last, T val, boost::mpl::true_ t)
        {
            detail::fill(sequential_execution_policy(),
                first, last, val, t);
        }
        /// \endcond
    }

    /// Assigns the given value to the elements in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam T           The type of the value to be assigned (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param value        The value to be assigned.
    ///
    /// The comparisons in the parallel \a fill algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a fill algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a fill algorithm returns a \a hpx::future<void> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a void.
    ///
    template <typename ExPolicy, typename InIter, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, void>::type
    >::type
    fill(ExPolicy && policy, InIter first, InIter last, T value)
    {
        typedef typename std::iterator_traits<InIter>::iterator_category
             iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::is_base_of<
                std::forward_iterator_tag, iterator_category>::value),
            "Required at least forward iterator.");

        typedef typename is_sequential_execution_policy<ExPolicy>::type is_seq;

        return detail::fill( std::forward<ExPolicy>(policy),
            first, last, value, is_seq());
    }
    ///////////////////////////////////////////////////////////////////////////
    // fill_n
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename ExPolicy, typename OutIter, typename T>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        fill_n(ExPolicy const&, OutIter first, std::size_t count, T val,
        boost::mpl::true_)
        {
            try {
                return detail::algorithm_result<ExPolicy, OutIter>::get(
                    std::fill_n(first, count, val));
            }
            catch (...) {
                detail::handle_exception<ExPolicy>::call();
            }
        }

        template <typename ExPolicy, typename OutIter, typename T>
        typename detail::algorithm_result<ExPolicy, OutIter>::type
        fill_n(ExPolicy const& policy, OutIter first, std::size_t count, T val,
            boost::mpl::false_ f)
        {
            typedef typename std::iterator_traits<OutIter>::iterator_category
                category;
            typedef typename std::iterator_traits<OutIter>::value_type type;

            return for_each_n(policy, first, count,
                        [val](type& v) {
                            v = val;
                        }, f);

        }

        template <typename OutIter, typename T>
        OutIter fill_n(execution_policy const& policy,
            OutIter first, std::size_t count, T val, boost::mpl::false_)
        {
            HPX_PARALLEL_DISPATCH(policy, detail::fill_n, first, count, val);
        }

        template <typename OutIter, typename T>
        OutIter fill_n(execution_policy const& policy,
            OutIter first, std::size_t count, T val, boost::mpl::true_ t)
        {
            return detail::fill_n(sequential_execution_policy(),
                first, count, val, t);
        }
        /// \endcond
    }

    /// Assigns the given value value to the first count elements in the range
    /// beginning at first if count > 0. Does nothing otherwise.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, for
    ///         count > 0.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam OutIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam T           The type of the value to be assigned (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param value        The value to be assigned.
    ///
    /// The comparisons in the parallel \a fill_n algorithm invoked with
    /// an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a fill_n algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a fill_n algorithm returns a \a hpx::future<void> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a void.
    ///
    template <typename ExPolicy, typename OutIter, typename Size, typename T>
    inline typename boost::enable_if<
        is_execution_policy<ExPolicy>,
        typename detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    fill_n(ExPolicy && policy, OutIter first, Size count, T value)
    {
        typedef typename std::iterator_traits<OutIter>::iterator_category
            iterator_category;

        BOOST_STATIC_ASSERT_MSG(
            (boost::mpl::or_<
                boost::is_base_of<
                    std::forward_iterator_tag, iterator_category>,
                boost::is_same<
                    std::output_iterator_tag, iterator_category>
            >::value),
            "Requires at least output iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative<Size>::call(count))
        {
            return detail::algorithm_result<ExPolicy, OutIter>::get(
                std::move(first));
        }

        typedef typename boost::mpl::or_<
            is_sequential_execution_policy<ExPolicy>,
            boost::is_same<std::output_iterator_tag, iterator_category>
        >::type is_seq;

        return detail::fill_n(
            std::forward<ExPolicy>(policy),
            first, std::size_t(count), value, is_seq());
    }
}}}

#endif
