//  Copyright (c) 2014-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/uninitialized_copy.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the copy operation, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// without an execution policy object will execute in sequential order in
    /// the calling thread.
    ///
    /// \returns  The \a uninitialized_copy algorithm returns \a FwdIter.
    ///           The \a uninitialized_copy algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename InIter, typename FwdIter>
    FwdIter uninitialized_copy(InIter first, InIter last, FwdIter dest);

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the copy operation, the function has no effects.
    /// Executed according to the policy.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_copy algorithm returns a
    ///           \a hpx::future<FwdIter2>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a uninitialized_copy algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, FwdIter2>
    uninitialized_copy(
        ExPolicy&& policy, FwdIter1 first, FwdIter1 last, FwdIter2 dest);

    /// Copies the elements in the range [first, first + count), starting from
    /// first and proceeding to first + count - 1., to another range beginning
    /// at dest. If an exception is thrown during the copy operation, the
    /// function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a uninitialized_copy_n algorithm
    /// invoked without an execution policy object execute in sequential order
    /// in the calling thread.
    ///
    /// \returns  The \a uninitialized_copy_n algorithm returns a
    ///           returns \a FwdIter.
    ///           The \a uninitialized_copy_n algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename InIter, typename Size, typename FwdIter>
    FwdIter uninitialized_copy_n(InIter first, Size count, FwdIter dest);

    /// Copies the elements in the range [first, first + count), starting from
    /// first and proceeding to first + count - 1., to another range beginning
    /// at dest. If an exception is thrown during the copy operation, the
    /// function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a uninitialized_copy_n algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_copy_n algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_copy_n algorithm returns a
    ///           \a hpx::future<FwdIter2> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a uninitialized_copy_n algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Size,
        typename FwdIter2>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, FwdIter2>
    uninitialized_copy_n(
        ExPolicy&& policy, FwdIter1 first, Size count, FwdIter2 dest);
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
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/type_support/construct_at.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::parallel {

    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_copy
    namespace detail {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename InIter1, typename FwdIter2, typename Cond>
        util::in_out_result<InIter1, FwdIter2> sequential_uninitialized_copy(
            InIter1 first, FwdIter2 dest, Cond cond)
        {
            FwdIter2 current = dest;
            try
            {
                for (/* */; cond(first, current); (void) ++first, ++current)
                {
                    hpx::construct_at(std::addressof(*current), *first);
                }
                return util::in_out_result<InIter1, FwdIter2>{first, current};
            }
            catch (...)
            {
                for (/* */; dest != current; ++dest)
                {
                    std::destroy_at(std::addressof(*dest));
                }
                throw;
            }
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename InIter1, typename InIter2>
        util::in_out_result<InIter1, InIter2> sequential_uninitialized_copy_n(
            InIter1 first, std::size_t count, InIter2 dest,
            util::cancellation_token<util::detail::no_data>& tok)
        {
            return {std::next(first, count),
                util::loop_with_cleanup_n_with_token(
                    first, count, dest, tok,
                    [](InIter1 it, InIter2 dest) -> void {
                        hpx::construct_at(std::addressof(*dest), *it);
                    },
                    [](InIter2 dest) -> void {
                        std::destroy_at(std::addressof(*dest));
                    })};
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Iter, typename FwdIter2>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<Iter, FwdIter2>>::type
        parallel_sequential_uninitialized_copy_n(
            ExPolicy&& policy, Iter first, std::size_t count, FwdIter2 dest)
        {
            if (count == 0)
            {
                return util::detail::algorithm_result<ExPolicy,
                    util::in_out_result<Iter, FwdIter2>>::
                    get(util::in_out_result<Iter, FwdIter2>{first, dest});
            }

            using zip_iterator = hpx::util::zip_iterator<Iter, FwdIter2>;
            using partition_result_type = std::pair<FwdIter2, FwdIter2>;

            util::cancellation_token<util::detail::no_data> tok;

            return util::partitioner_with_cleanup<ExPolicy,
                util::in_out_result<Iter, FwdIter2>, partition_result_type>::
                call(
                    HPX_FORWARD(ExPolicy, policy), zip_iterator(first, dest),
                    count,
                    [tok](zip_iterator t, std::size_t part_size) mutable
                    -> partition_result_type {
                        using hpx::get;
                        auto iters = t.get_iterator_tuple();
                        FwdIter2 dest = get<1>(iters);
                        return std::make_pair(dest,
                            util::get_second_element(
                                sequential_uninitialized_copy_n(
                                    get<0>(iters), part_size, dest, tok)));
                    },
                    // finalize, called once if no error occurred
                    [dest, first, count](auto&& data) mutable
                    -> util::in_out_result<Iter, FwdIter2> {
                        // make sure iterators embedded in function object that is
                        // attached to futures are invalidated
                        util::detail::clear_container(data);

                        std::advance(first, count);
                        std::advance(dest, count);
                        return util::in_out_result<Iter, FwdIter2>{first, dest};
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
        template <typename IterPair>
        struct uninitialized_copy
          : public algorithm<uninitialized_copy<IterPair>, IterPair>
        {
            constexpr uninitialized_copy() noexcept
              : algorithm<uninitialized_copy, IterPair>("uninitialized_copy")
            {
            }

            template <typename ExPolicy, typename InIter1, typename Sent,
                typename FwdIter2>
            static util::in_out_result<InIter1, FwdIter2> sequential(
                ExPolicy, InIter1 first, Sent last, FwdIter2 dest)
            {
                return sequential_uninitialized_copy(
                    first, dest, [last](InIter1 first, FwdIter2) -> bool {
                        return first != last;
                    });
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename FwdIter2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<Iter, FwdIter2>>::type
            parallel(ExPolicy&& policy, Iter first, Sent last, FwdIter2 dest)
            {
                return parallel_sequential_uninitialized_copy_n(
                    HPX_FORWARD(ExPolicy, policy), first,
                    detail::distance(first, last), dest);
            }
        };
        /// \endcond
    }    // namespace detail

    /////////////////////////////////////////////////////////////////////////////
    // uninitialized_copy_sent
    namespace detail {

        /// \cond NOINTERNAL
        template <typename IterPair>
        struct uninitialized_copy_sent
          : public algorithm<uninitialized_copy_sent<IterPair>, IterPair>
        {
            constexpr uninitialized_copy_sent() noexcept
              : algorithm<uninitialized_copy_sent, IterPair>(
                    "uninitialized_copy_sent")
            {
            }

            template <typename ExPolicy, typename InIter1, typename Sent1,
                typename FwdIter2, typename Sent2>
            static util::in_out_result<InIter1, FwdIter2> sequential(ExPolicy,
                InIter1 first, Sent1 last, FwdIter2 dest, Sent2 last_d)
            {
                return sequential_uninitialized_copy(first, dest,
                    [last, last_d](InIter1 first, FwdIter2 current) -> bool {
                        return !(first == last || current == last_d);
                    });
            }

            template <typename ExPolicy, typename Iter, typename Sent1,
                typename FwdIter2, typename Sent2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_out_result<Iter, FwdIter2>>::type
            parallel(ExPolicy&& policy, Iter first, Sent1 last, FwdIter2 dest,
                Sent2 last_d)
            {
                std::size_t const dist1 = detail::distance(first, last);
                std::size_t const dist2 = detail::distance(dest, last_d);
                std::size_t dist = dist1 <= dist2 ? dist1 : dist2;

                return parallel_sequential_uninitialized_copy_n(
                    HPX_FORWARD(ExPolicy, policy), first, dist, dest);
            }
        };
        /// \endcond
    }    // namespace detail

    /////////////////////////////////////////////////////////////////////////////
    // uninitialized_copy_n
    namespace detail {

        /// \cond NOINTERNAL
        template <typename IterPair>
        struct uninitialized_copy_n
          : public algorithm<uninitialized_copy_n<IterPair>, IterPair>
        {
            constexpr uninitialized_copy_n() noexcept
              : algorithm<uninitialized_copy_n, IterPair>(
                    "uninitialized_copy_n")
            {
            }

            template <typename ExPolicy, typename InIter, typename FwdIter2>
            static util::in_out_result<InIter, FwdIter2> sequential(
                ExPolicy, InIter first, std::size_t count, FwdIter2 dest)
            {
                FwdIter2 current = dest;
                try
                {
                    for (/* */; count > 0; ++first, (void) ++current, --count)
                    {
                        hpx::construct_at(std::addressof(*current), *first);
                    }
                    return util::in_out_result<InIter, FwdIter2>{
                        first, current};
                }
                catch (...)
                {
                    for (/* */; dest != current; ++dest)
                    {
                        std::destroy_at(std::addressof(*dest));
                    }
                    throw;
                }
            }

            template <typename ExPolicy, typename Iter, typename FwdIter2>
            static util::detail::algorithm_result_t<ExPolicy,
                util::in_out_result<Iter, FwdIter2>>
            parallel(
                ExPolicy&& policy, Iter first, std::size_t count, FwdIter2 dest)
            {
                return parallel_sequential_uninitialized_copy_n(
                    HPX_FORWARD(ExPolicy, policy), first, count, dest);
            }
        };
        /// \endcond
    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::uninitialized_copy
    inline constexpr struct uninitialized_copy_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_copy_t>
    {
        // clang-format off
        template <typename InIter, typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_forward_iterator_v<FwdIter>
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            hpx::uninitialized_copy_t, InIter first, InIter last, FwdIter dest)
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "Required at least input iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_copy<
                    parallel::util::in_out_result<InIter, FwdIter>>()
                    .call(hpx::execution::seq, first, last, dest));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter1> &&
                hpx::traits::is_forward_iterator_v<FwdIter2>
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy, FwdIter2>
        tag_fallback_invoke(hpx::uninitialized_copy_t, ExPolicy&& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");

            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_copy<
                    parallel::util::in_out_result<FwdIter1, FwdIter2>>()
                    .call(HPX_FORWARD(ExPolicy, policy), first, last, dest));
        }
    } uninitialized_copy{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::uninitialized_copy_n
    inline constexpr struct uninitialized_copy_n_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_copy_n_t>
    {
        // clang-format off
        template <typename InIter, typename Size,
            typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            hpx::uninitialized_copy_n_t, InIter first, Size count, FwdIter dest)
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "Required at least input iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return dest;
            }

            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_copy_n<
                    parallel::util::in_out_result<InIter, FwdIter>>()
                    .call(hpx::execution::seq, first,
                        static_cast<std::size_t>(count), dest));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Size,
            typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter1> &&
                hpx::traits::is_forward_iterator_v<FwdIter2> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_invoke(hpx::uninitialized_copy_n_t, ExPolicy&& policy,
            FwdIter1 first, Size count, FwdIter2 dest)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter2>::get(HPX_MOVE(dest));
            }

            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_copy_n<
                    parallel::util::in_out_result<FwdIter1, FwdIter2>>()
                    .call(HPX_FORWARD(ExPolicy, policy), first,
                        static_cast<std::size_t>(count), dest));
        }
    } uninitialized_copy_n{};
}    // namespace hpx

#endif    // DOXYGEN
