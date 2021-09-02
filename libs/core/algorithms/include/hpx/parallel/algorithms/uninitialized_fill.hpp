//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/uninitialized_fill.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {

    /// Copies the given \a value to an uninitialized memory area, defined by
    /// the range [first, last). If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to be assigned (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param value        The value to be assigned.
    ///
    /// The assignments in the parallel \a uninitialized_fill algorithm invoked
    /// without an execution policy object will execute in sequential order in
    /// the calling thread.
    ///
    /// \returns  The \a uninitialized_fill algorithm  returns nothing
    ///
    template <typename FwdIter, typename T>
    void uninitialized_fill(FwdIter first, FwdIter last, T const& value);

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
    /// The initializations in the parallel \a uninitialized_fill algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The initializations in the parallel \a uninitialized_fill algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified threads,
    /// and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a uninitialized_fill algorithm returns a
    ///           \a hpx::future<void>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns nothing
    ///           otherwise.
    ///
    template <typename ExPolicy, typename FwdIter, typename T>
    typename parallel::util::detail::algorithm_result<ExPolicy>::type
    uninitialized_fill(
        ExPolicy&& policy, FwdIter first, FwdIter last, T const& value);

    /// Copies the given \a value value to the first count elements in an
    /// uninitialized memory area beginning at first. If an exception is thrown
    /// during the initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam T           The type of the value to be assigned (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param value        The value to be assigned.
    ///
    /// The assignments in the parallel \a uninitialized_fill_n algorithm
    /// invoked without an execution policy object execute in sequential order
    /// in the calling thread.
    ///
    /// \returns  The \a uninitialized_fill_n algorithm returns a
    ///           returns \a FwdIter.
    ///           The \a uninitialized_fill_n algorithm returns the output
    ///           iterator to the element in the range, one past
    ///           the last element copied.
    ///
    template <typename FwdIter, typename Size, typename T>
    FwdIter uninitialized_fill_n(FwdIter first, Size count, T const& value);

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
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The initializations in the parallel \a uninitialized_fill_n algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified threads,
    /// and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a uninitialized_fill_n algorithm returns a
    ///           \a hpx::future<FwdIter>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns FwdIter
    ///           otherwise.
    ///           The \a uninitialized_fill_n algorithm returns the output
    ///           iterator to the element in the range, one past
    ///           the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size, typename T>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter>::type
    uninitialized_fill_n(
        ExPolicy&& policy, FwdIter first, Size count, T const& value);
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/local/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/type_support/void_guard.hpp>

#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
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

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_fill
    namespace detail {
        /// \cond NOINTERNAL

        // provide our own implementation of std::uninitialized_fill as some
        // versions of MSVC horribly fail at compiling it for some types T
        template <typename InIter, typename Sent, typename T>
        InIter std_uninitialized_fill(InIter first, Sent last, T const& value)
        {
            using value_type =
                typename std::iterator_traits<InIter>::value_type;

            InIter current = first;
            try
            {
                for (/* */; current != last; ++current)
                {
                    ::new (std::addressof(*current)) value_type(value);
                }
                return current;
            }
            catch (...)
            {
                for (/* */; first != current; ++first)
                {
                    (*first).~value_type();
                }
                throw;
            }
        }

        template <typename InIter, typename T>
        InIter sequential_uninitialized_fill_n(InIter first, std::size_t count,
            T const& value,
            util::cancellation_token<util::detail::no_data>& tok)
        {
            using value_type =
                typename std::iterator_traits<InIter>::value_type;

            return util::loop_with_cleanup_n_with_token(
                first, count, tok,
                [&value](InIter it) -> void {
                    ::new (std::addressof(*it)) value_type(value);
                },
                [](InIter it) -> void { (*it).~value_type(); });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Iter, typename T>
        typename util::detail::algorithm_result<ExPolicy, Iter>::type
        parallel_sequential_uninitialized_fill_n(
            ExPolicy&& policy, Iter first, std::size_t count, T const& value)
        {
            if (count == 0)
                return util::detail::algorithm_result<ExPolicy, Iter>::get(
                    std::move(first));

            typedef std::pair<Iter, Iter> partition_result_type;
            typedef typename std::iterator_traits<Iter>::value_type value_type;

            util::cancellation_token<util::detail::no_data> tok;
            return util::partitioner_with_cleanup<ExPolicy, Iter,
                partition_result_type>::
                call(
                    std::forward<ExPolicy>(policy), first, count,
                    [value, tok](Iter it, std::size_t part_size) mutable
                    -> partition_result_type {
                        return std::make_pair(it,
                            sequential_uninitialized_fill_n(
                                it, part_size, value, tok));
                    },
                    // finalize, called once if no error occurred
                    [first, count](
                        std::vector<hpx::future<partition_result_type>>&&
                            data) mutable -> Iter {
                        // make sure iterators embedded in function object that is
                        // attached to futures are invalidated
                        data.clear();

                        std::advance(first, count);
                        return first;
                    },
                    // cleanup function, called for each partition which
                    // didn't fail, but only if at least one failed
                    [](partition_result_type&& r) -> void {
                        while (r.first != r.second)
                        {
                            (*r.first).~value_type();
                            ++r.first;
                        }
                    });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter>
        struct uninitialized_fill
          : public detail::algorithm<uninitialized_fill<Iter>, Iter>
        {
            uninitialized_fill()
              : uninitialized_fill::algorithm("uninitialized_fill")
            {
            }

            template <typename ExPolicy, typename Sent, typename T>
            static Iter sequential(
                ExPolicy, Iter first, Sent last, T const& value)
            {
                return std_uninitialized_fill(first, last, value);
            }

            template <typename ExPolicy, typename Sent, typename T>
            static typename util::detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy&& policy, Iter first, Sent last, T const& value)
            {
                if (first == last)
                    return util::detail::algorithm_result<ExPolicy, Iter>::get(
                        std::move(first));

                return parallel_sequential_uninitialized_fill_n(
                    std::forward<ExPolicy>(policy), first,
                    detail::distance(first, last), value);
            }
        };
        /// \endcond
    }    // namespace detail

    template <typename ExPolicy, typename FwdIter, typename T>
    HPX_LOCAL_DEPRECATED_V(1, 7,
        "hpx::parallel::uninitialized_fill is deprecated, use "
        "hpx::uninitialized_fill "
        "instead")
    inline typename std::enable_if<hpx::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy>::type>::type
        uninitialized_fill(
            ExPolicy&& policy, FwdIter first, FwdIter last, T const& value)
    {
        static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
            "Required at least forward iterator.");

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        using result_type =
            typename hpx::parallel::util::detail::algorithm_result<
                ExPolicy>::type;

        return hpx::util::void_guard<result_type>(),
               hpx::parallel::v1::detail::uninitialized_fill<FwdIter>().call(
                   std::forward<ExPolicy>(policy), first, last, value);
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }

    /////////////////////////////////////////////////////////////////////////////
    // uninitialized_fill_n
    namespace detail {
        /// \cond NOINTERNAL

        // provide our own implementation of std::uninitialized_fill_n as some
        // versions of MSVC horribly fail at compiling it for some types T
        template <typename InIter, typename Size, typename T>
        InIter std_uninitialized_fill_n(
            InIter first, Size count, T const& value)
        {
            using value_type =
                typename std::iterator_traits<InIter>::value_type;

            InIter current = first;
            try
            {
                for (/* */; count > 0; ++current, (void) --count)
                {
                    ::new (static_cast<void*>(std::addressof(*current)))
                        value_type(value);
                }
                return current;
            }
            catch (...)
            {
                for (/* */; first != current; ++first)
                {
                    (*first).~value_type();
                }
                throw;
            }
        }

        template <typename Iter>
        struct uninitialized_fill_n
          : public detail::algorithm<uninitialized_fill_n<Iter>, Iter>
        {
            uninitialized_fill_n()
              : uninitialized_fill_n::algorithm("uninitialized_fill_n")
            {
            }

            template <typename ExPolicy, typename T>
            static Iter sequential(
                ExPolicy, Iter first, std::size_t count, T const& value)
            {
                return std_uninitialized_fill_n(first, count, value);
            }

            template <typename ExPolicy, typename T>
            static typename util::detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy&& policy, Iter first, std::size_t count,
                T const& value)
            {
                return parallel_sequential_uninitialized_fill_n(
                    std::forward<ExPolicy>(policy), first, count, value);
            }
        };
        /// \endcond
    }    // namespace detail

    template <typename ExPolicy, typename FwdIter, typename Size, typename T>
    HPX_LOCAL_DEPRECATED_V(1, 7,
        "hpx::parallel::uninitialized_fill_n is deprecated, use "
        "hpx::uninitialized_fill_n "
        "instead")
    inline typename std::enable_if<hpx::is_execution_policy<ExPolicy>::value,
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type>::type
        uninitialized_fill_n(
            ExPolicy&& policy, FwdIter first, Size count, T const& value)
    {
        static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
            "Required at least forward iterator.");

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
        return detail::uninitialized_fill_n<FwdIter>().call(
            std::forward<ExPolicy>(policy), first, std::size_t(count), value);
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::uninitialized_fill
    HPX_INLINE_CONSTEXPR_VARIABLE struct uninitialized_fill_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_fill_t>
    {
        // clang-format off
        template <typename FwdIter, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend void tag_fallback_dispatch(hpx::uninitialized_fill_t,
            FwdIter first, FwdIter last, T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            hpx::parallel::v1::detail::uninitialized_fill<FwdIter>().call(
                hpx::execution::seq, first, last, value);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy>::type
        tag_fallback_dispatch(hpx::uninitialized_fill_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            using result_type =
                typename hpx::parallel::util::detail::algorithm_result<
                    ExPolicy>::type;

            return hpx::util::void_guard<result_type>(),
                   hpx::parallel::v1::detail::uninitialized_fill<FwdIter>()
                       .call(
                           std::forward<ExPolicy>(policy), first, last, value);
        }

    } uninitialized_fill{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::uninitialized_fill_n
    HPX_INLINE_CONSTEXPR_VARIABLE struct uninitialized_fill_n_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_fill_n_t>
    {
        // clang-format off
        template <typename FwdIter, typename Size, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                std::is_integral<Size>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_dispatch(hpx::uninitialized_fill_n_t,
            FwdIter first, Size count, T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return first;
            }

            return hpx::parallel::v1::detail::uninitialized_fill_n<FwdIter>()
                .call(hpx::execution::seq, first, std::size_t(count), value);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                std::is_integral<Size>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_dispatch(hpx::uninitialized_fill_n_t, ExPolicy&& policy,
            FwdIter first, Size count, T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(std::move(first));
            }

            return hpx::parallel::v1::detail::uninitialized_fill_n<FwdIter>()
                .call(std::forward<ExPolicy>(policy), first, std::size_t(count),
                    value);
        }

    } uninitialized_fill_n{};
}    // namespace hpx

#endif    // DOXYGEN
