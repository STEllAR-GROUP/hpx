//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2017-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/fill.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Assigns the given value to the elements in the range [first, last).
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
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a fill algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a fill algorithm returns a \a hpx::future<void> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a void.
    ///
    template <typename ExPolicy, typename FwdIter, typename T>
    typename util::detail::algorithm_result<ExPolicy>::type
    fill(ExPolicy&& policy, FwdIter first, FwdIter last, T value);

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
    /// \tparam FwdIter     The type of the source iterators used (deduced).
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
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a fill_n algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a fill_n algorithm returns a \a hpx::future<void> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a void.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size, typename T>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    fill_n(ExPolicy&& policy, FwdIter first, Size count, T value);

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/local/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/algorithms/traits/is_value_proxy.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/type_support/void_guard.hpp>

#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/detail/fill.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // fill
    namespace detail {
        /// \cond NOINTERNAL
        template <typename T>
        struct fill_iteration
        {
            typename std::decay<T>::type val_;

            template <typename U>
            HPX_HOST_DEVICE typename std::enable_if<
                !hpx::traits::is_value_proxy<U>::value>::type
            operator()(U& u) const
            {
                u = val_;
            }

            template <typename U>
            HPX_HOST_DEVICE typename std::enable_if<
                hpx::traits::is_value_proxy<U>::value>::type
            operator()(U u) const
            {
                u = val_;
            }
        };

        template <typename Iter>
        struct fill : public detail::algorithm<fill<Iter>, Iter>
        {
            fill()
              : fill::algorithm("fill")
            {
            }

            template <typename ExPolicy, typename InIter, typename Sent,
                typename T>
            HPX_HOST_DEVICE static InIter sequential(
                ExPolicy&& policy, InIter first, Sent last, T const& val)
            {
                return detail::sequential_fill(
                    std::forward<ExPolicy>(policy), first, last, val);
            }

            template <typename ExPolicy, typename FwdIter, typename Sent,
                typename T>
            static
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(
                    ExPolicy&& policy, FwdIter first, Sent last, T const& val)
            {
                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        FwdIter>::get(std::move(first));
                }

                return for_each_n<FwdIter>().call(
                    std::forward<ExPolicy>(policy), first,
                    detail::distance(first, last), fill_iteration<T>{val},
                    util::projection_identity());
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename T,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_forward_iterator<FwdIter>::value
        )>
    // clang-format on
    HPX_LOCAL_DEPRECATED_V(
        1, 6, "hpx::parallel::fill is deprecated, use hpx::fill instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        fill(ExPolicy&& policy, FwdIter first, FwdIter last, T const& value)
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::fill<FwdIter>().call(
            std::forward<ExPolicy>(policy), first, last, value);
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    // fill_n
    namespace detail {
        /// \cond NOINTERNAL
        template <typename FwdIter>
        struct fill_n : public detail::algorithm<fill_n<FwdIter>, FwdIter>
        {
            fill_n()
              : fill_n::algorithm("fill_n")
            {
            }

            template <typename ExPolicy, typename InIter, typename T>
            static InIter sequential(ExPolicy&& policy, InIter first,
                std::size_t count, T const& val)
            {
                return detail::sequential_fill_n(
                    std::forward<ExPolicy>(policy), first, count, val);
            }

            template <typename ExPolicy, typename T>
            static
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(ExPolicy&& policy, FwdIter first, std::size_t count,
                    T const& val)
            {
                return for_each_n<FwdIter>().call(
                    std::forward<ExPolicy>(policy), first, count,
                    [val](auto& v) -> void { v = val; },
                    util::projection_identity());
            }
        };
        /// \endcond
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename Size, typename T,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_forward_iterator<FwdIter>::value
        )>
    // clang-format on
    HPX_LOCAL_DEPRECATED_V(
        1, 6, "hpx::parallel::fill_n is deprecated, use hpx::fill_n instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        fill_n(ExPolicy&& policy, FwdIter first, Size count, T const& value)
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");

        // if count is representing a negative value, we do nothing
        if (detail::is_negative(count))
        {
            return util::detail::algorithm_result<ExPolicy, FwdIter>::get(
                std::move(first));
        }

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return detail::fill_n<FwdIter>().call(
            std::forward<ExPolicy>(policy), first, std::size_t(count), value);
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::fill
    HPX_INLINE_CONSTEXPR_VARIABLE struct fill_t final
      : hpx::detail::tag_parallel_algorithm<fill_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<
            ExPolicy>::type
        tag_fallback_dispatch(fill_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, T const& value)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            using result_type =
                typename hpx::parallel::util::detail::algorithm_result<
                    ExPolicy>::type;

            return hpx::util::void_guard<result_type>(),
                   hpx::parallel::v1::detail::fill<FwdIter>().call(
                       std::forward<ExPolicy>(policy), first, last, value);
        }

        // clang-format off
        template <typename FwdIter, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend void tag_fallback_dispatch(
            fill_t, FwdIter first, FwdIter last, T const& value)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            hpx::parallel::v1::detail::fill<FwdIter>().call(
                hpx::execution::seq, first, last, value);
        }
    } fill{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::fill_n
    HPX_INLINE_CONSTEXPR_VARIABLE struct fill_n_t final
      : hpx::detail::tag_parallel_algorithm<fill_n_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_dispatch(fill_n_t, ExPolicy&& policy, FwdIter first,
            Size count, T const& value)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(std::move(first));
            }

            return hpx::parallel::v1::detail::fill_n<FwdIter>().call(
                std::forward<ExPolicy>(policy), first, std::size_t(count),
                value);
        }

        // clang-format off
        template <typename FwdIter, typename Size, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_dispatch(
            fill_n_t, FwdIter first, Size count, T const& value)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return first;
            }

            return hpx::parallel::v1::detail::fill_n<FwdIter>().call(
                hpx::execution::seq, first, std::size_t(count), value);
        }
    } fill_n{};

}    // namespace hpx

#endif    // DOXYGEN
