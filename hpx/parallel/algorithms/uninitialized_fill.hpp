//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/uninitialized_fill.hpp

#if !defined(HPX_PARALLEL_DETAIL_UNINITIALIZED_FILL_OCT_06_2014_1019AM)
#define HPX_PARALLEL_DETAIL_UNINITIALIZED_FILL_OCT_06_2014_1019AM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner_with_cleanup.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_fill
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename FwdIter, typename T>
        FwdIter sequential_uninitialized_fill_n(FwdIter first, std::size_t count,
            T const& value, util::cancellation_token<util::detail::no_data>& tok)
        {
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;

            return
                util::loop_with_cleanup_n_with_token(
                    first, count, tok,
                    [&value](FwdIter it) {
                        ::new (&*it) value_type(value);
                    },
                    [](FwdIter it) {
                        (*it).~value_type();
                    });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Iter, typename T>
        typename util::detail::algorithm_result<ExPolicy>::type
        parallel_sequential_uninitialized_fill_n(ExPolicy && policy,
            Iter first, std::size_t count, T const& value)
        {
            if (count == 0)
                return util::detail::algorithm_result<ExPolicy>::get();

            typedef std::pair<Iter, Iter> partition_result_type;
            typedef typename std::iterator_traits<Iter>::value_type
                value_type;

            util::cancellation_token<util::detail::no_data> tok;
            return util::partitioner_with_cleanup<
                    ExPolicy, void, partition_result_type
                >::call(
                    std::forward<ExPolicy>(policy), first, count,
                    [value, tok](Iter it, std::size_t part_size)
                        mutable -> partition_result_type
                    {
                        return std::make_pair(it,
                            sequential_uninitialized_fill_n(
                                it, part_size, value, tok));
                    },
                    // finalize, called once if no error occurred
                    [](std::vector<hpx::future<partition_result_type> > &&)
                        -> void
                    {
                    },
                    // cleanup function, called for each partition which
                    // didn't fail, but only if at least one failed
                    [](partition_result_type && r) -> void
                    {
                        while (r.first != r.second)
                        {
                            (*r.first).~value_type();
                            ++r.first;
                        }
                    });
        }

        ///////////////////////////////////////////////////////////////////////
        struct uninitialized_fill
          : public detail::algorithm<uninitialized_fill>
        {
            uninitialized_fill()
              : uninitialized_fill::algorithm("uninitialized_fill")
            {}

            template <typename ExPolicy, typename Iter, typename T>
            static hpx::util::unused_type
            sequential(ExPolicy, Iter first, Iter last, T const& value)
            {
                std::uninitialized_fill(first, last, value);
                return hpx::util::unused;
            }

            template <typename ExPolicy, typename Iter, typename T>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy && policy, Iter first, Iter last,
                T const& value)
            {
                if (first == last)
                    return util::detail::algorithm_result<ExPolicy>::get();

                return parallel_sequential_uninitialized_fill_n(
                    std::forward<ExPolicy>(policy), first,
                    std::distance(first, last), value);
            }
        };
        /// \endcond
    }

    /// Copies the given \a value to an uninitialized memory area, defined by
    /// the range [first, last). If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last
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
    /// The initializations in the parallel \a uninitialized_fill algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The initializations in the parallel \a uninitialized_fill algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified threads,
    /// and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a uninitialized_fill algorithm returns a
    ///           \a hpx::future<void>, if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns nothing
    ///           otherwise.
    ///
    template <typename ExPolicy, typename InIter, typename T>
    inline typename std::enable_if<
        is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy>::type
    >::type
    uninitialized_fill(ExPolicy && policy, InIter first, InIter last,
        T const& value)
    {
        static_assert(
            (hpx::traits::is_at_least_input_iterator<InIter>::value),
            "Required at least input iterator.");

        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
                hpx::traits::is_input_iterator<InIter>::value
            > is_seq;

        return detail::uninitialized_fill().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, last, value);
    }

    /////////////////////////////////////////////////////////////////////////////
    // uninitialized_fill_n
    namespace detail
    {
        /// \cond NOINTERNAL
        struct uninitialized_fill_n
          : public detail::algorithm<uninitialized_fill_n>
        {
            uninitialized_fill_n()
              : uninitialized_fill_n::algorithm("uninitialized_fill_n")
            {}

            template <typename ExPolicy, typename FwdIter, typename T>
            static hpx::util::unused_type
            sequential(ExPolicy, FwdIter first, std::size_t count,
                T const& value)
            {
                std::uninitialized_fill_n(first, count, value);
                return hpx::util::unused;
            }

            template <typename ExPolicy, typename FwdIter, typename T>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy && policy, FwdIter first, std::size_t count,
                T const& value)
            {
                return parallel_sequential_uninitialized_fill_n(
                    std::forward<ExPolicy>(policy), first, count, value);
            }
        };
        /// \endcond
    }

    /// Copies the given \a value value to the first count elements in an
    /// uninitialized memory area beginning at first. If an exception is thrown
    /// during the initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
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
    /// The initializations in the parallel \a uninitialized_fill_n algorithm
    /// invoked with an execution policy object of type
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The initializations in the parallel \a uninitialized_fill_n algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified threads,
    /// and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a uninitialized_fill_n algorithm returns a
    ///           \a hpx::future<void>, if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns nothing
    ///           otherwise.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size, typename T>
    inline typename std::enable_if<
        is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy>::type
    >::type
    uninitialized_fill_n(ExPolicy && policy, FwdIter first, Size count,
        T const& value)
    {
        static_assert(
            (hpx::traits::is_at_least_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative(count))
        {
            return util::detail::algorithm_result<ExPolicy>::get();
        }

        typedef is_sequential_execution_policy<ExPolicy> is_seq;

        return detail::uninitialized_fill_n().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, std::size_t(count), value);
    }
}}}

#endif
