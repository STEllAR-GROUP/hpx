//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/fill.hpp

#if !defined(HPX_PARALLEL_DETAIL_FILL_JUNE_12_2014_0405PM)
#define HPX_PARALLEL_DETAIL_FILL_JUNE_12_2014_0405PM

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/is_value_proxy.hpp>
#include <hpx/util/void_guard.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/is_negative.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

// extra
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // fill
    namespace detail
    {
        template <typename T>
        struct fill_iteration
        {
            typename hpx::util::decay<T>::type val_;

            template <typename U>
            HPX_HOST_DEVICE
            typename std::enable_if<
                !hpx::traits::is_value_proxy<U>::value
            >::type
            operator()(U &u)
            {
                u = val_;
            }

            template <typename U>
            HPX_HOST_DEVICE
            typename std::enable_if<
                hpx::traits::is_value_proxy<U>::value
            >::type
            operator()(U u)
            {
                u = val_;
            }
        };

        /// \cond NOINTERNAL
        struct fill : public detail::algorithm<fill>
        {
            fill()
              : fill::algorithm("fill")
            {}

            template <typename ExPolicy, typename InIter, typename T>
            HPX_HOST_DEVICE
            static hpx::util::unused_type
            sequential(ExPolicy, InIter first, InIter last,
                T const& val)
            {
                std::fill(first, last, val);
                return hpx::util::unused;
            }

            template <typename ExPolicy, typename FwdIter, typename T>
            static typename util::detail::algorithm_result<ExPolicy>::type
            parallel(ExPolicy && policy, FwdIter first, FwdIter last,
                 T const& val)
            {
                typedef typename util::detail::algorithm_result<ExPolicy>::type
                    result_type;
                typedef typename std::iterator_traits<FwdIter>::value_type type;

                if(first == last)
                    return util::detail::algorithm_result<ExPolicy>::get();

                return hpx::util::void_guard<result_type>(),
                    for_each_n<FwdIter>().call(
                        std::forward<ExPolicy>(policy), std::false_type(),
                        first, std::distance(first, last),
                        fill_iteration<T>{val},
                        util::projection_identity());
            }
        };

        template <typename ExPolicy, typename InIter, typename T>
        inline typename util::detail::algorithm_result<
            ExPolicy, void
        >::type
        fill_(ExPolicy && policy, InIter first, InIter last, T value,
            std::false_type)
        {
            typedef is_sequential_execution_policy<ExPolicy> is_seq;

            return detail::fill().call(
                std::forward<ExPolicy>(policy), is_seq(),
                first, last, value);
        }

        // forward declare the segmented version of this algorithm
        template <typename ExPolicy, typename InIter, typename T>
        inline typename util::detail::algorithm_result<
            ExPolicy, void
        >::type
        fill_(ExPolicy && policy, InIter first, InIter last, T value,
            std::true_type);

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
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a fill algorithm returns a \a hpx::future<void> if the
    ///           execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a void.
    ///
    template <typename ExPolicy, typename InIter, typename T>
    inline typename std::enable_if<
        is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, void>::type
    >::type
    fill(ExPolicy && policy, InIter first, InIter last, T value)
    {
        static_assert(
            (hpx::traits::is_forward_iterator<InIter>::value),
            "Requires at least forward iterator.");

        typedef hpx::traits::is_segmented_iterator<InIter> is_segmented;

        return detail::fill_(
            std::forward<ExPolicy>(policy), first, last, value,
                is_segmented()
            );
    }


    ///////////////////////////////////////////////////////////////////////////
    // fill_n
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename OutIter>
        struct fill_n : public detail::algorithm<fill_n<OutIter>, OutIter>
        {
            fill_n()
              : fill_n::algorithm("fill_n")
            {}

            template <typename ExPolicy, typename T>
            static OutIter
            sequential(ExPolicy, OutIter first, std::size_t count,
                T const& val)
            {
                return std::fill_n(first, count, val);
            }

            template <typename ExPolicy, typename T>
            static typename util::detail::algorithm_result<
                ExPolicy, OutIter
            >::type
            parallel(ExPolicy && policy, OutIter first, std::size_t count,
                T const& val)
            {
                typedef typename std::iterator_traits<OutIter>::value_type type;

                return
                    for_each_n<OutIter>().call(
                        std::forward<ExPolicy>(policy),
                        std::false_type(), first, count,
                        [val](type& v) -> void
                        {
                            v = val;
                        },
                        util::projection_identity());
            }
        };
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
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a fill_n algorithm returns a \a hpx::future<void> if the
    ///           execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a void.
    ///
    template <typename ExPolicy, typename OutIter, typename Size, typename T>
    inline typename std::enable_if<
        is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, OutIter>::type
    >::type
    fill_n(ExPolicy && policy, OutIter first, Size count, T value)
    {
        static_assert(
            (hpx::traits::is_output_iterator<OutIter>::value ||
                hpx::traits::is_forward_iterator<OutIter>::value),
            "Requires at least output iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative(count))
        {
            return util::detail::algorithm_result<ExPolicy, OutIter>::get(
                std::move(first));
        }

        typedef std::integral_constant<bool,
                is_sequential_execution_policy<ExPolicy>::value ||
               !hpx::traits::is_forward_iterator<OutIter>::value
            > is_seq;

        return detail::fill_n<OutIter>().call(
            std::forward<ExPolicy>(policy), is_seq(),
            first, std::size_t(count), value);
    }
}}}

#endif
