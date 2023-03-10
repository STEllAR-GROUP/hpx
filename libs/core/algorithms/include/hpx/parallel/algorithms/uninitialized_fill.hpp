//  Copyright (c) 2014-2023 Hartmut Kaiser
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
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam T           The type of the value to be assigned (deduced).
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param value        The value to be assigned.
    ///
    /// \returns  The \a uninitialized_fill algorithm returns nothing
    ///
    template <typename FwdIter, typename T>
    void uninitialized_fill(FwdIter first, FwdIter last, T const& value);

    /// Copies the given \a value to an uninitialized memory area, defined by
    /// the range [first, last). If an exception is thrown during the
    /// initialization, the function has no effects. Executed according to the
    /// policy.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
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
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy>
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
    /// during the initialization, the function has no effects. Executed
    /// according to the policy.
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
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, FwdIter>
    uninitialized_fill_n(
        ExPolicy&& policy, FwdIter first, Size count, T const& value);
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/clear_container.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner_with_cleanup.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/type_support/construct_at.hpp>
#include <hpx/type_support/void_guard.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::parallel {

    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_fill
    namespace detail {
        /// \cond NOINTERNAL

        // provide our own implementation of std::uninitialized_fill as some
        // versions of MSVC horribly fail at compiling it for some types T
        template <typename InIter, typename Sent, typename T>
        InIter std_uninitialized_fill(InIter first, Sent last, T const& value)
        {
            InIter current = first;
            try
            {
                for (/* */; current != last; ++current)
                {
                    hpx::construct_at(std::addressof(*current), value);
                }
                return current;
            }
            catch (...)
            {
                for (/* */; first != current; ++first)
                {
                    std::destroy_at(std::addressof(*first));
                }
                throw;
            }
        }

        template <typename InIter, typename T>
        InIter sequential_uninitialized_fill_n(InIter first, std::size_t count,
            T const& value,
            util::cancellation_token<util::detail::no_data>& tok)
        {
            return util::loop_with_cleanup_n_with_token(
                first, count, tok,
                [&value](InIter it) -> void {
                    hpx::construct_at(std::addressof(*it), value);
                },
                [](InIter it) -> void {
                    std::destroy_at(std::addressof(*it));
                });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Iter, typename T>
        util::detail::algorithm_result_t<ExPolicy, Iter>
        parallel_sequential_uninitialized_fill_n(
            ExPolicy&& policy, Iter first, std::size_t count, T const& value)
        {
            if (count == 0)
            {
                return util::detail::algorithm_result<ExPolicy, Iter>::get(
                    HPX_MOVE(first));
            }

            using partition_result_type = std::pair<Iter, Iter>;

            util::cancellation_token<util::detail::no_data> tok;
            return util::partitioner_with_cleanup<ExPolicy, Iter,
                partition_result_type>::
                call(
                    HPX_FORWARD(ExPolicy, policy), first, count,
                    [value, tok](Iter it, std::size_t part_size) mutable
                    -> partition_result_type {
                        return std::make_pair(it,
                            sequential_uninitialized_fill_n(
                                it, part_size, value, tok));
                    },
                    // finalize, called once if no error occurred
                    [first, count](auto&& data) mutable -> Iter {
                        // make sure iterators embedded in function object that
                        // is attached to futures are invalidated
                        util::detail::clear_container(data);

                        std::advance(first, count);
                        return first;
                    },
                    // cleanup function, called for each partition which didn't
                    // fail, but only if at least one failed
                    [](partition_result_type&& r) -> void {
                        while (r.first != r.second)
                        {
                            std::destroy_at(std::addressof(*r.first));
                            ++r.first;
                        }
                    });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter>
        struct uninitialized_fill
          : public algorithm<uninitialized_fill<Iter>, Iter>
        {
            constexpr uninitialized_fill() noexcept
              : algorithm<uninitialized_fill, Iter>("uninitialized_fill")
            {
            }

            template <typename ExPolicy, typename Sent, typename T>
            static Iter sequential(
                ExPolicy, Iter first, Sent last, T const& value)
            {
                return std_uninitialized_fill(first, last, value);
            }

            template <typename ExPolicy, typename Sent, typename T>
            static util::detail::algorithm_result_t<ExPolicy, Iter> parallel(
                ExPolicy&& policy, Iter first, Sent last, T const& value)
            {
                if (first == last)
                    return util::detail::algorithm_result<ExPolicy, Iter>::get(
                        HPX_MOVE(first));

                return parallel_sequential_uninitialized_fill_n(
                    HPX_FORWARD(ExPolicy, policy), first,
                    detail::distance(first, last), value);
            }
        };
        /// \endcond
    }    // namespace detail

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
            InIter current = first;
            try
            {
                for (/* */; count > 0; ++current, (void) --count)
                {
                    hpx::construct_at(std::addressof(*current), value);
                }
                return current;
            }
            catch (...)
            {
                for (/* */; first != current; ++first)
                {
                    std::destroy_at(std::addressof(*first));
                }
                throw;
            }
        }

        template <typename Iter>
        struct uninitialized_fill_n
          : public algorithm<uninitialized_fill_n<Iter>, Iter>
        {
            constexpr uninitialized_fill_n() noexcept
              : algorithm<uninitialized_fill_n, Iter>("uninitialized_fill_n")
            {
            }

            template <typename ExPolicy, typename T>
            static Iter sequential(
                ExPolicy, Iter first, std::size_t count, T const& value)
            {
                return std_uninitialized_fill_n(first, count, value);
            }

            template <typename ExPolicy, typename T>
            static util::detail::algorithm_result_t<ExPolicy, Iter> parallel(
                ExPolicy&& policy, Iter first, std::size_t count,
                T const& value)
            {
                return parallel_sequential_uninitialized_fill_n(
                    HPX_FORWARD(ExPolicy, policy), first, count, value);
            }
        };
        /// \endcond
    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::uninitialized_fill
    inline constexpr struct uninitialized_fill_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_fill_t>
    {
        // clang-format off
        template <typename FwdIter, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator_v<FwdIter>
            )>
        // clang-format on
        friend void tag_fallback_invoke(hpx::uninitialized_fill_t,
            FwdIter first, FwdIter last, T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            hpx::parallel::detail::uninitialized_fill<FwdIter>().call(
                hpx::execution::seq, first, last, value);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter>
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy>
        tag_fallback_invoke(hpx::uninitialized_fill_t, ExPolicy&& policy,
            FwdIter first, FwdIter last, T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            using result_type =
                typename hpx::parallel::util::detail::algorithm_result<
                    ExPolicy>::type;

            return hpx::util::void_guard<result_type>(),
                   hpx::parallel::detail::uninitialized_fill<FwdIter>().call(
                       HPX_FORWARD(ExPolicy, policy), first, last, value);
        }

    } uninitialized_fill{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::uninitialized_fill_n
    inline constexpr struct uninitialized_fill_n_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_fill_n_t>
    {
        // clang-format off
        template <typename FwdIter, typename Size, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(hpx::uninitialized_fill_n_t,
            FwdIter first, Size count, T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return first;
            }

            return hpx::parallel::detail::uninitialized_fill_n<FwdIter>().call(
                hpx::execution::seq, first, static_cast<std::size_t>(count),
                value);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(hpx::uninitialized_fill_n_t, ExPolicy&& policy,
            FwdIter first, Size count, T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(HPX_MOVE(first));
            }

            return hpx::parallel::detail::uninitialized_fill_n<FwdIter>().call(
                HPX_FORWARD(ExPolicy, policy), first,
                static_cast<std::size_t>(count), value);
        }
    } uninitialized_fill_n{};
}    // namespace hpx

#endif    // DOXYGEN
