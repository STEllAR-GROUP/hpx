//  Copyright (c) 2014-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/uninitialized_default_construct.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// by default-initialization. If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct
    /// algorithm invoked without an execution policy object will execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a uninitialized_default_construct algorithm
    ///           returns nothing
    ///
    template <typename FwdIter>
    void uninitialized_default_construct(FwdIter first, FwdIter last);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// by default-initialization. If an exception is thrown during the
    /// initialization, the function has no effects. Executed according
    /// to the policy.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct
    /// algorithm invoked with an execution policy object of type \a
    /// sequenced_policy execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct
    /// algorithm invoked with an execution policy object of type \a
    /// parallel_policy or \a parallel_task_policy are permitted to execute
    /// in an unordered fashion in unspecified threads, and indeterminately
    /// sequenced within each thread.
    ///
    /// \returns  The \a uninitialized_default_construct algorithm returns a
    ///           \a hpx::future<void>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns nothing
    ///           otherwise.
    ///
    template <typename ExPolicy, typename FwdIter>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy>
    uninitialized_default_construct(
        ExPolicy&& policy, FwdIter first, FwdIter last);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// [first, first + count) by default-initialization. If an exception
    /// is thrown during the initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    ///
    /// \returns  The \a uninitialized_default_construct_n algorithm returns a
    ///           returns \a FwdIter.
    ///           The \a uninitialized_default_construct_n algorithm returns
    ///           the iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename FwdIter, typename Size>
    FwdIter uninitialized_default_construct_n(FwdIter first, Size count);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// [first, first + count) by default-initialization. If an exception
    /// is thrown during the initialization, the function has no effects.
    /// Executed according to the policy.
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
    /// The assignments in the parallel \a uninitialized_default_construct_n
    /// algorithm invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct_n
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_default_construct_n algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a uninitialized_default_construct_n algorithm returns
    ///           the iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, FwdIter>
    uninitialized_default_construct_n(
        ExPolicy&& policy, FwdIter first, Size count);
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
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
#include <hpx/type_support/void_guard.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::parallel {

    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_default_construct
    namespace detail {
        /// \cond NOINTERNAL

        // provide our own implementation of
        // std::uninitialized_default_construct as some versions of MSVC
        // horribly fail at compiling it for some types T
        template <typename Iter, typename Sent>
        Iter std_uninitialized_default_construct(Iter first, Sent last)
        {
            using value_type = typename std::iterator_traits<Iter>::value_type;

            Iter s_first = first;
            try
            {
                for (/* */; first != last; ++first)
                {
                    ::new (std::addressof(*first)) value_type;
                }
                return first;
            }
            catch (...)
            {
                for (/* */; s_first != first; ++s_first)
                {
                    std::destroy_at(std::addressof(*s_first));
                }
                throw;
            }
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename InIter>
        InIter sequential_uninitialized_default_construct_n(InIter first,
            std::size_t count,
            util::cancellation_token<util::detail::no_data>& tok)
        {
            using value_type =
                typename std::iterator_traits<InIter>::value_type;

            return util::loop_with_cleanup_n_with_token(
                first, count, tok,
                [](InIter it) -> void {
                    ::new (std::addressof(*it)) value_type;
                },
                [](InIter it) -> void {
                    std::destroy_at(std::addressof(*it));
                });
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename FwdIter>
        util::detail::algorithm_result_t<ExPolicy, FwdIter>
        parallel_sequential_uninitialized_default_construct_n(
            ExPolicy&& policy, FwdIter first, std::size_t count)
        {
            if (count == 0)
            {
                return util::detail::algorithm_result<ExPolicy, FwdIter>::get(
                    HPX_MOVE(first));
            }

            using partition_result_type = std::pair<FwdIter, FwdIter>;

            util::cancellation_token<util::detail::no_data> tok;
            return util::partitioner_with_cleanup<ExPolicy, FwdIter,
                partition_result_type>::
                call(
                    HPX_FORWARD(ExPolicy, policy), first, count,
                    [tok](FwdIter it, std::size_t part_size) mutable
                    -> partition_result_type {
                        return std::make_pair(it,
                            sequential_uninitialized_default_construct_n(
                                it, part_size, tok));
                    },
                    // finalize, called once if no error occurred
                    [first, count](auto&& data) mutable -> FwdIter {
                        // make sure iterators embedded in function object that is
                        // attached to futures are invalidated
                        util::detail::clear_container(data);

                        std::advance(first, count);
                        return first;
                    },
                    // cleanup function, called for each partition which
                    // didn't fail, but only if at least one failed
                    [](partition_result_type&& r) -> void {
                        while (r.first != r.second)
                        {
                            std::destroy_at(std::addressof(*r.first));
                            ++r.first;
                        }
                    });
        }
        ///////////////////////////////////////////////////////////////////////
        template <typename FwdIter>
        struct uninitialized_default_construct
          : public algorithm<uninitialized_default_construct<FwdIter>, FwdIter>
        {
            constexpr uninitialized_default_construct() noexcept
              : algorithm<uninitialized_default_construct, FwdIter>(
                    "uninitialized_default_construct")
            {
            }

            template <typename ExPolicy, typename Sent>
            static FwdIter sequential(ExPolicy, FwdIter first, Sent last)
            {
                return std_uninitialized_default_construct(first, last);
            }

            template <typename ExPolicy, typename Sent>
            static util::detail::algorithm_result_t<ExPolicy, FwdIter> parallel(
                ExPolicy&& policy, FwdIter first, Sent last)
            {
                return parallel_sequential_uninitialized_default_construct_n(
                    HPX_FORWARD(ExPolicy, policy), first,
                    detail::distance(first, last));
            }
        };
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_default_construct_n
    namespace detail {
        /// \cond NOINTERNAL

        // provide our own implementation of std::uninitialized_default_construct as some
        // versions of MSVC horribly fail at compiling it for some types T
        template <typename InIter>
        InIter std_uninitialized_default_construct_n(
            InIter first, std::size_t count)
        {
            using value_type =
                typename std::iterator_traits<InIter>::value_type;

            InIter s_first = first;
            try
            {
                for (/* */; count != 0; (void) ++first, --count)
                {
                    ::new (std::addressof(*first)) value_type;
                }
                return first;
            }
            catch (...)
            {
                for (/* */; s_first != first; ++s_first)
                {
                    std::destroy_at(std::addressof(*s_first));
                }
                throw;
            }
        }

        template <typename FwdIter>
        struct uninitialized_default_construct_n
          : public algorithm<uninitialized_default_construct_n<FwdIter>,
                FwdIter>
        {
            constexpr uninitialized_default_construct_n() noexcept
              : algorithm<uninitialized_default_construct_n, FwdIter>(
                    "uninitialized_default_construct_n")
            {
            }

            template <typename ExPolicy>
            static FwdIter sequential(
                ExPolicy, FwdIter first, std::size_t count)
            {
                return std_uninitialized_default_construct_n(first, count);
            }

            template <typename ExPolicy>
            static util::detail::algorithm_result_t<ExPolicy, FwdIter> parallel(
                ExPolicy&& policy, FwdIter first, std::size_t count)
            {
                return parallel_sequential_uninitialized_default_construct_n(
                    HPX_FORWARD(ExPolicy, policy), first, count);
            }
        };
        /// \endcond
    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::uninitialized_default_construct
    inline constexpr struct uninitialized_default_construct_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_default_construct_t>
    {
        // clang-format off
        template <typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator_v<FwdIter>
            )>
        // clang-format on
        friend void tag_fallback_invoke(
            hpx::uninitialized_default_construct_t, FwdIter first, FwdIter last)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            hpx::parallel::detail::uninitialized_default_construct<FwdIter>()
                .call(hpx::execution::seq, first, last);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter>
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy>
        tag_fallback_invoke(hpx::uninitialized_default_construct_t,
            ExPolicy&& policy, FwdIter first, FwdIter last)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            using result_type =
                typename hpx::parallel::util::detail::algorithm_result<
                    ExPolicy>::type;

            return hpx::util::void_guard<result_type>(),
                   hpx::parallel::detail::uninitialized_default_construct<
                       FwdIter>()
                       .call(HPX_FORWARD(ExPolicy, policy), first, last);
        }

    } uninitialized_default_construct{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::uninitialized_default_construct_n
    inline constexpr struct uninitialized_default_construct_n_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_default_construct_n_t>
    {
        // clang-format off
        template <typename FwdIter, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            hpx::uninitialized_default_construct_n_t, FwdIter first, Size count)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return first;
            }

            return hpx::parallel::detail::uninitialized_default_construct_n<
                FwdIter>()
                .call(hpx::execution::seq, first,
                    static_cast<std::size_t>(count));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(hpx::uninitialized_default_construct_n_t,
            ExPolicy&& policy, FwdIter first, Size count)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(HPX_MOVE(first));
            }

            return hpx::parallel::detail::uninitialized_default_construct_n<
                FwdIter>()
                .call(HPX_FORWARD(ExPolicy, policy), first,
                    static_cast<std::size_t>(count));
        }
    } uninitialized_default_construct_n{};
}    // namespace hpx

#endif    // DOXYGEN
