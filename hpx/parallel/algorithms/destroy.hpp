//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/destroy.hpp

#if !defined(HPX_PARALLEL_DETAIL_destroy_JUN_01_2017_1049AM)
#define HPX_PARALLEL_DETAIL_destroy_JUN_01_2017_1049AM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/void_guard.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

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
    // destroy
    namespace detail
    {
        /// \cond NOINTERNAL

        // provide our own implementation of std::destroy
        // as some versions of MSVC horribly fail at compiling it for some types
        // T
        template <typename FwdIter>
        void std_destroy(FwdIter first, FwdIter last)
        {
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;

            for (/* */; first != last; ++first)
            {
                std::addressof(*first)->~value_type();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename FwdIter>
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        parallel_sequential_destroy_n(
            ExPolicy && policy, FwdIter first, std::size_t count)
        {
            if (count == 0)
            {
                return util::detail::algorithm_result<ExPolicy, FwdIter>::get(
                    std::move(first));
            }

            return util::foreach_partitioner<ExPolicy>::call(
                std::forward<ExPolicy>(policy), first, count,
                [](FwdIter first, std::size_t count, std::size_t)
                {
                    typedef typename std::iterator_traits<FwdIter>::value_type
                        value_type;

                    return util::loop_n<ExPolicy>(first, count,
                        [](FwdIter it)
                        {
                            std::addressof(*it)->~value_type();
                        });
                },
                util::projection_identity());
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter>
        struct destroy
          : public detail::algorithm<destroy<FwdIter> >
        {
            destroy()
              : destroy::algorithm("destroy")
            {}

            template <typename ExPolicy>
            static hpx::util::unused_type
            sequential(ExPolicy, FwdIter first, FwdIter last)
            {
                std_destroy(first, last);
                return hpx::util::unused;
            }

            template <typename ExPolicy>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last)
            {
                return util::detail::algorithm_result<ExPolicy>::get(
                    parallel_sequential_destroy_n(
                        std::forward<ExPolicy>(policy), first,
                        std::distance(first, last)));
            }
        };
        /// \endcond
    }

    /// Destroys objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first operations.
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
    /// The operations in the parallel \a destroy
    /// algorithm invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The operations in the parallel \a destroy
    /// algorithm invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a destroy algorithm returns a
    ///           \a hpx::future<void>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a void otherwise.
    ///
    template <typename ExPolicy, typename FwdIter,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value)>
    typename util::detail::algorithm_result<ExPolicy>::type
    destroy(ExPolicy && policy, FwdIter first, FwdIter last)
    {
        static_assert(
            (hpx::traits::is_forward_iterator<FwdIter>::value),
            "Required at least forward iterator.");

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value
            > is_seq;

        return detail::destroy<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last);
    }

    ///////////////////////////////////////////////////////////////////////////
    // destroy_n
    namespace detail
    {
        /// \cond NOINTERNAL

        // provide our own implementation of std::destroy
        // as some versions of MSVC horribly fail at compiling it for some
        // types T
        template <typename FwdIter>
        FwdIter std_destroy_n(FwdIter first, std::size_t count)
        {
            typedef typename std::iterator_traits<FwdIter>::value_type
                value_type;

            for (/* */; count != 0; (void) ++first, --count)
            {
                std::addressof(*first)->~value_type();
            }

            return first;
        }

        template <typename FwdIter>
        struct destroy_n
          : public detail::algorithm<
                destroy_n<FwdIter>, FwdIter>
        {
            destroy_n()
              : destroy_n::algorithm("destroy_n")
            {}

            template <typename ExPolicy>
            static FwdIter sequential(ExPolicy, FwdIter first, std::size_t count)
            {
                return std_destroy_n(first, count);
            }

            template <typename ExPolicy>
            static typename util::detail::algorithm_result<
                ExPolicy, FwdIter
            >::type
            parallel(ExPolicy && policy, FwdIter first, std::size_t count)
            {
                return parallel_sequential_destroy_n(
                    std::forward<ExPolicy>(policy), first, count);
            }
        };
        /// \endcond
    }

    /// Destroys objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the range [first, first + count).
    ///
    /// \note   Complexity: Performs exactly \a count operations, if
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
    ///                     elements to apply this algorithm to.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    ///
    /// The operations in the parallel \a destroy_n
    /// algorithm invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The operations in the parallel \a destroy_n
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a destroy_n algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a destroy_n algorithm returns the
    ///           iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_iterator<FwdIter>::value)>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    destroy_n(ExPolicy && policy, FwdIter first, Size count)
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

        typedef std::integral_constant<bool,
                execution::is_sequenced_execution_policy<ExPolicy>::value
            > is_seq;

        return detail::destroy_n<FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, std::size_t(count));
    }
}}}

#endif
