//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/uninitialized_value_construct.hpp

#if !defined(HPX_PARALLEL_DETAIL_UNINITIALIZED_VALUE_CONSTRUCT_JUN_01_2017_1049AM)
#define HPX_PARALLEL_DETAIL_UNINITIALIZED_VALUE_CONSTRUCT_JUN_01_2017_1049AM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/void_guard.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner_with_cleanup.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_value_construct
    namespace detail
    {
        /// \cond NOINTERNAL

        // provide our own implementation of std::uninitialized_value_construct
        // as some versions of MSVC horribly fail at compiling it for some types
        // T
        template <typename InIter>
        void std_uninitialized_value_construct(InIter first, InIter last)
        {
            typedef typename std::iterator_traits<InIter>::value_type
                value_type;

            InIter s_first = first;
            try {
                for (/* */; first != last; ++first)
                {
                    ::new (std::addressof(*first)) value_type();
                }
            }
            catch (...) {
                for (/* */; s_first != first; ++s_first)
                {
                    (*s_first).~value_type();
                }
                throw;
            }
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename InIter>
        InIter sequential_uninitialized_value_construct_n(
            InIter first, std::size_t count,
            util::cancellation_token<util::detail::no_data>& tok)
        {
            typedef typename std::iterator_traits<InIter>::value_type
                value_type;

            return
                util::loop_with_cleanup_n_with_token(
                    first, count, tok,
                    [](InIter it) -> void
                    {
                        ::new (std::addressof(*it)) value_type();
                    },
                    [](InIter it) -> void
                    {
                        (*it).~value_type();
                    });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename FwdIter>
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        parallel_sequential_uninitialized_value_construct_n(
            ExPolicy && policy, FwdIter first, std::size_t count)
        {
            if (count == 0)
            {
                return util::detail::algorithm_result<ExPolicy, FwdIter>::get(
                    std::move(first));
            }

            typedef std::pair<FwdIter, FwdIter> partition_result_type;
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;
            typedef typename util::detail::algorithm_result<ExPolicy>::type
                result_type;

            util::cancellation_token<util::detail::no_data> tok;
            return util::partitioner_with_cleanup<
                    ExPolicy, FwdIter, partition_result_type
                >::call(
                    std::forward<ExPolicy>(policy), first, count,
                    [tok](FwdIter it, std::size_t part_size)
                        mutable -> partition_result_type
                    {
                        return std::make_pair(it,
                            sequential_uninitialized_value_construct_n(
                                it, part_size, tok));
                    },
                    // finalize, called once if no error occurred
                    [first, count](
                        std::vector<hpx::future<partition_result_type> > &&)
                            mutable -> FwdIter
                    {
                        std::advance(first, count);
                        return first;
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
        template <typename FwdIter>
        struct uninitialized_value_construct
          : public detail::algorithm<uninitialized_value_construct<FwdIter> >
        {
            uninitialized_value_construct()
              : uninitialized_value_construct::algorithm(
                    "uninitialized_value_construct")
            {}

            template <typename ExPolicy, typename InIter>
            static hpx::util::unused_type
            sequential(ExPolicy, InIter first, InIter last)
            {
                std_uninitialized_value_construct(first, last);
                return hpx::util::unused;
            }

            template <typename ExPolicy>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last)
            {
                return util::detail::algorithm_result<ExPolicy>::get(
                    parallel_sequential_uninitialized_value_construct_n(
                        std::forward<ExPolicy>(policy), first,
                        std::distance(first, last)));
            }
        };
        /// \endcond
    }

    /// Constructs objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the uninitialized storage designated by the range [first, last) by
    /// default-initialization. If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct
    /// algorithm invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct
    /// algorithm invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_value_construct algorithm returns a
    ///           \a hpx::future<void>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a void otherwise.
    ///
    template <typename ExPolicy, typename FwdIter,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value)>
    typename util::detail::algorithm_result<ExPolicy>::type
    uninitialized_value_construct(ExPolicy && policy, FwdIter first,
        FwdIter last)
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::uninitialized_value_construct<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last);
    }

    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_value_construct_n
    namespace detail
    {
        /// \cond NOINTERNAL

        // provide our own implementation of std::uninitialized_value_construct
        // as some versions of MSVC horribly fail at compiling it for some
        // types T
        template <typename InIter>
        InIter std_uninitialized_value_construct_n(InIter first, std::size_t count)
        {
            typedef typename std::iterator_traits<InIter>::value_type
                value_type;

            InIter s_first = first;
            try {
                for (/* */; count != 0; (void) ++first, --count)
                {
                    ::new (std::addressof(*first)) value_type();
                }
                return first;
            }
            catch (...) {
                for (/* */; s_first != first; ++s_first)
                {
                    (*s_first).~value_type();
                }
                throw;
            }
        }

        template <typename FwdIter>
        struct uninitialized_value_construct_n
          : public detail::algorithm<
                uninitialized_value_construct_n<FwdIter>, FwdIter>
        {
            uninitialized_value_construct_n()
              : uninitialized_value_construct_n::algorithm(
                    "uninitialized_value_construct_n")
            {}

            template <typename ExPolicy, typename InIter>
            static InIter sequential(ExPolicy, InIter first, std::size_t count)
            {
                return std_uninitialized_value_construct_n(first, count);
            }

            template <typename ExPolicy>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, std::size_t count)
            {
                return parallel_sequential_uninitialized_value_construct_n(
                    std::forward<ExPolicy>(policy), first, count);
            }
        };
        /// \endcond
    }

    /// Constructs objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the uninitialized storage designated by the range [first, first + count) by
    /// default-initialization. If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct_n
    /// algorithm invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_value_construct_n
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_value_construct_n algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a uninitialized_value_construct_n algorithm returns the
    ///           iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    uninitialized_value_construct_n(ExPolicy && policy,
        FwdIter first, Size count)
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative(count))
        {
            return util::detail::algorithm_result<ExPolicy, FwdIter>::get(
                std::move(first));
        }

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::uninitialized_value_construct_n<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, std::size_t(count));
    }
}}}

#endif
