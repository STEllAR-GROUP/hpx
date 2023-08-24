//  Copyright (c) 2014-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/uninitialized_relocate.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {

    /// Relocates the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the move-construction of an element, all elements left
    /// in the input range, as well as all objects already constructed in the
    /// destination range are destroyed. After this algorithm completes, the
    /// source range should be freed or reused without destroying the objects.
    ///
    /// \note   Complexity: time: O(n), space: O(1)
    ///         1)  For "trivially relocatable" underlying types (T) and
    ///             a contiguous iterator range [first, last):
    ///             std::distance(first, last)*sizeof(T) bytes are copied.
    ///         2)  For "trivially relocatable" underlying types (T) and
    ///             a non-contiguous iterator range [first, last):
    ///             std::distance(first, last) memory copies of sizeof(T)
    ///             bytes each are performed.
    ///         3)  For "non-trivially relocatable" underlying types (T):
    ///             std::distance(first, last) move assignments and
    ///             destructions are performed.
    ///
    /// \note   Declare a type as "trivially relocatable" using the
    ///         `HPX_DECLARE_TRIVIALLY_RELOCATABLE` macros found in
    ///         <hpx/type_support/is_trivially_relocatable.hpp>.
    ///
    /// \tparam InIter1     The type of the source iterator first (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam InIter2     The type of the source iterator last (deduced).
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
    /// The assignments in the parallel \a uninitialized_relocate algorithm invoked
    /// without an execution policy object will execute in sequential order in
    /// the calling thread.
    ///
    /// \returns  The \a uninitialized_relocate algorithm returns \a FwdIter.
    ///           The \a uninitialized_relocate algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element relocated.
    ///
    template <typename InIter1, typename InIter2, typename FwdIter>
    FwdIter uninitialized_relocate(InIter1 first, InIter2 last, FwdIter dest);

    /// Relocates the elements in the range defined by [first, first + count), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the move-construction of an element, all elements left
    /// in the input range, as well as all objects already constructed in the
    /// destination range are destroyed. After this algorithm completes, the
    /// source range should be freed or reused without destroying the objects.
    ///
    /// \note   Complexity: time: O(n), space: O(1)
    ///         1)  For "trivially relocatable" underlying types (T) and
    ///             a contiguous iterator range [first, last):
    ///             std::distance(first, last)*sizeof(T) bytes are copied.
    ///         2)  For "trivially relocatable" underlying types (T) and
    ///             a non-contiguous iterator range [first, last):
    ///             std::distance(first, last) memory copies of sizeof(T)
    ///             bytes each are performed.
    ///         3)  For "non-trivially relocatable" underlying types (T):
    ///             std::distance(first, last) move assignments and
    ///             destructions are performed.
    ///
    /// \note   Declare a type as "trivially relocatable" using the
    ///         `HPX_DECLARE_TRIVIALLY_RELOCATABLE` macros found in
    ///         <hpx/type_support/is_trivially_relocatable.hpp>.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter1     The type of the source iterator first (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam InIter2     The type of the source iterator last (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam FwdIter     The type of the iterator representing the
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
    //// The assignments in the parallel \a uninitialized_relocate_n algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_relocate algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_relocate algorithm returns a
    ///           \a hpx::future<FwdIter>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a uninitialized_relocate algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element relocated.
    ///
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename FwdIter>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, FwdIter>
    uninitialized_relocate(
        ExPolicy&& policy, InIter1 first, InIter2 last, FwdIter dest);

    /// Relocates the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the move-construction of an element, all elements left
    /// in the input range, as well as all objects already constructed in the
    /// destination range are destroyed. After this algorithm completes, the
    /// source range should be freed or reused without destroying the objects.
    ///
    /// \note   Complexity: time: O(n), space: O(1)
    ///         1)  For "trivially relocatable" underlying types (T) and
    ///             a contiguous iterator range [first, first+count):
    ///             `count*sizeof(T)` bytes are copied.
    ///         2)  For "trivially relocatable" underlying types (T) and
    ///             a non-contiguous iterator range [first, first+count):
    ///             `count` memory copies of sizeof(T) bytes each are performed.
    ///         3)  For "non-trivially relocatable" underlying types (T):
    ///             `count` move assignments and destructions are performed.
    ///
    /// \note   Declare a type as "trivially relocatable" using the
    ///         `HPX_DECLARE_TRIVIALLY_RELOCATABLE` macros found in
    ///         <hpx/type_support/is_trivially_relocatable.hpp>.
    ///
    /// \tparam InIter      The type of the source iterator first (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to relocate.
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
    /// The assignments in the parallel \a uninitialized_relocate_n algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a uninitialized_relocate_n algorithm returns \a FwdIter.
    ///           The \a uninitialized_relocate_n algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element relocated.
    ///
    template <typename InIter, typename Size, typename FwdIter>
    FwdIter uninitialized_relocate_n(InIter first, Size count, FwdIter dest);

    /// Relocates the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the move-construction of an element, all elements left
    /// in the input range, as well as all objects already constructed in the
    /// destination range are destroyed. After this algorithm completes, the
    /// source range should be freed or reused without destroying the objects.
    ///
    /// \note   Complexity: time: O(n), space: O(1)
    ///         1)  For "trivially relocatable" underlying types (T) and
    ///             a contiguous iterator range [first, first+count):
    ///             `count*sizeof(T)` bytes are copied.
    ///         2)  For "trivially relocatable" underlying types (T) and
    ///             a non-contiguous iterator range [first, first+count):
    ///             `count` memory copies of sizeof(T) bytes each are performed.
    ///         3)  For "non-trivially relocatable" underlying types (T):
    ///             `count` move assignments and destructions are performed.
    ///
    /// \note   Declare a type as "trivially relocatable" using the
    ///         `HPX_DECLARE_TRIVIALLY_RELOCATABLE` macros found in
    ///         <hpx/type_support/is_trivially_relocatable.hpp>.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter      The type of the source iterator first (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to relocate.
    /// \tparam FwdIter     The type of the iterator representing the
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
    /// The assignments in the parallel \a uninitialized_relocate_n algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_relocate_n algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_relocate_n algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a uninitialized_relocate_n algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element relocated.
    ///
    template <typename ExPolicy, typename InIter, typename Size,
        typename FwdIter>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, FwdIter>
    uninitialized_relocate_n(
        ExPolicy&& policy, InIter first, Size count, FwdIter dest);
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/pointer_category.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/unseq/loop.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/clear_container.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner_with_cleanup.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/transfer.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/type_support/construct_at.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::parallel {

    ///////////////////////////////////////////////////////////////////////////
    // uninitialized_relocate
    namespace detail {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////

        // clang-format off
        template <typename ExPolicy, typename InIter, typename FwdIter,
        typename Size, HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_input_iterator_v<InIter> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<InIter, FwdIter>>::type
        parallel_uninitialized_relocate_n(
            ExPolicy&& policy, InIter first, Size count, FwdIter dest)
        {
            if (count == 0)
            {
                return util::detail::algorithm_result<ExPolicy,
                    util::in_out_result<InIter, FwdIter>>::
                    get(util::in_out_result<InIter, FwdIter>{first, dest});
            }

            using zip_iterator = hpx::util::zip_iterator<InIter, FwdIter>;
            using partition_result_type = std::pair<FwdIter, FwdIter>;

            return util::partitioner_with_cleanup<ExPolicy,
                util::in_out_result<InIter, FwdIter>, partition_result_type>::
                call(
                    HPX_FORWARD(ExPolicy, policy), zip_iterator(first, dest),
                    count,
                    [policy](zip_iterator t, std::size_t part_size) mutable
                    -> partition_result_type {
                        using hpx::get;

                        auto iters = t.get_iterator_tuple();

                        InIter part_source = get<0>(iters);
                        FwdIter part_dest = get<1>(iters);

                        // returns (dest begin, dest end)
                        return std::make_pair(part_dest,
                            util::get_second_element(
                                hpx::parallel::util::uninit_relocate_n(
                                    HPX_FORWARD(ExPolicy, policy), part_source,
                                    part_size, part_dest)));
                    },
                    // finalize, called once if no error occurred
                    [dest, first, count](auto&& data) mutable
                    -> util::in_out_result<InIter, FwdIter> {
                        // make sure iterators embedded in function object that is
                        // attached to futures are invalidated
                        util::detail::clear_container(data);

                        std::advance(first, count);
                        std::advance(dest, count);
                        return util::in_out_result<InIter, FwdIter>{
                            first, dest};
                    },
                    // cleanup function, called for each partition which
                    // didn't fail, but only if at least one failed
                    [](partition_result_type&& r) -> void {
                        std::destroy(r.first, r.second);
                    });
        }

        /////////////////////////////////////////////////////////////////////////////
        // uninitialized_relocate_n

        /// \cond NOINTERNAL
        template <typename IterPair>
        struct uninitialized_relocate_n
          : public algorithm<uninitialized_relocate_n<IterPair>, IterPair>
        {
            constexpr uninitialized_relocate_n() noexcept
              : algorithm<uninitialized_relocate_n, IterPair>(
                    "uninitialized_relocate_n")
            {
            }

            // non vectorized overload
            // clang-format off
            template <typename ExPolicy, typename InIter, typename FwdIter,
                HPX_CONCEPT_REQUIRES_(
                    hpx::is_sequenced_execution_policy_v<ExPolicy> &&
                    hpx::traits::is_input_iterator_v<InIter> &&
                    hpx::traits::is_forward_iterator_v<FwdIter>
                )>
            static util::in_out_result<InIter, FwdIter> sequential(
                ExPolicy&& policy, InIter first, std::size_t count, FwdIter dest)
                noexcept(hpx::traits::pointer_relocate_category<InIter,
                        FwdIter>::is_noexcept_relocatable_v)
            // clang-format on
            {
                return hpx::parallel::util::uninit_relocate_n(
                    HPX_FORWARD(ExPolicy, policy), first, count, dest);
            }

            // clang-format off
            template <typename ExPolicy, typename InIter, typename FwdIter,
                HPX_CONCEPT_REQUIRES_(
                    hpx::is_execution_policy_v<ExPolicy> &&
                    hpx::traits::is_input_iterator_v<InIter> &&
                    hpx::traits::is_forward_iterator_v<FwdIter>
                )>
            static util::detail::algorithm_result_t<ExPolicy,
                util::in_out_result<InIter, FwdIter>>
            parallel(ExPolicy&& policy, InIter first, std::size_t count,
                FwdIter dest) noexcept(
                hpx::traits::pointer_relocate_category<InIter,
                FwdIter>::is_noexcept_relocatable_v)
            // clang-format on
            {
                return parallel_uninitialized_relocate_n(
                    HPX_FORWARD(ExPolicy, policy), first, count, dest);
            }
        };
        /// \endcond

        /////////////////////////////////////////////////////////////////////////////
        // uninitialized_relocate
        /// \cond NOINTERNAL
        template <typename IterPair>
        struct uninitialized_relocate
          : public algorithm<uninitialized_relocate<IterPair>, IterPair>
        {
            constexpr uninitialized_relocate() noexcept
              : algorithm<uninitialized_relocate, IterPair>(
                    "uninitialized_relocate")
            {
            }

            // non vectorized overload
            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename FwdIter,
                HPX_CONCEPT_REQUIRES_(hpx::is_sequenced_execution_policy_v<
                    ExPolicy>&& hpx::traits::is_input_iterator_v<InIter1>&&
                        hpx::traits::is_input_iterator_v<InIter2>&&
                            hpx::traits::is_forward_iterator_v<FwdIter>)>
            static util::in_out_result<InIter1, FwdIter> sequential(
                ExPolicy&& policy, InIter1 first, InIter2 last,
                FwdIter dest) noexcept(hpx::traits::
                    pointer_relocate_category<InIter1,
                        FwdIter>::is_noexcept_relocatable_v)
            {
                auto count = std::distance(first, last);

                return hpx::parallel::util::uninit_relocate_n(
                    HPX_FORWARD(ExPolicy, policy), first, count, dest);
            }

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename FwdIter,
                HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy_v<ExPolicy>&&
                        hpx::traits::is_input_iterator_v<InIter1>&&
                            hpx::traits::is_input_iterator_v<InIter2>&&
                                hpx::traits::is_forward_iterator_v<FwdIter>)>
            static util::detail::algorithm_result_t<ExPolicy,
                util::in_out_result<InIter1, FwdIter>>
            parallel(ExPolicy&& policy, InIter1 first, InIter2 last,
                FwdIter dest) noexcept(hpx::traits::
                    pointer_relocate_category<InIter1,
                        FwdIter>::is_noexcept_relocatable_v)
            {
                auto count = std::distance(first, last);

                return parallel_uninitialized_relocate_n(
                    HPX_FORWARD(ExPolicy, policy), first, count, dest);
            }
        };
        /// \endcond

    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::uninitialized_relocate_n
    inline constexpr struct uninitialized_relocate_n_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_relocate_n_t>
    {
        // clang-format off
        template <typename InIter, typename Size,
            typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_input_iterator_v<InIter> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )>
        friend FwdIter tag_fallback_invoke(hpx::uninitialized_relocate_n_t,
            InIter first, Size count, FwdIter dest) noexcept(
            hpx::traits::pointer_relocate_category<InIter,
            FwdIter>::is_noexcept_relocatable_v)
        // clang-format on
        {
            static_assert(hpx::traits::pointer_relocate_category<InIter,
                              FwdIter>::valid_relocation,
                "Relocating from this source type to this destination type is "
                "ill-formed");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return dest;
            }

            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_relocate_n<
                    parallel::util::in_out_result<InIter, FwdIter>>()
                    .call(hpx::execution::seq, first,
                        static_cast<std::size_t>(count), dest));
        }

        // clang-format off
        template <typename ExPolicy, typename InIter, typename Size,
            typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )>
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(hpx::uninitialized_relocate_n_t, ExPolicy&& policy,
            InIter first, Size count, FwdIter dest) noexcept(
            hpx::traits::pointer_relocate_category<InIter,
                FwdIter>::is_noexcept_relocatable_v)
        // clang-format on
        {
            static_assert(hpx::traits::pointer_relocate_category<InIter,
                              FwdIter>::valid_relocation,
                "Relocating from this source type to this destination type is "
                "ill-formed");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(HPX_MOVE(dest));
            }

            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_relocate_n<
                    parallel::util::in_out_result<InIter, FwdIter>>()
                    .call(HPX_FORWARD(ExPolicy, policy), first,
                        static_cast<std::size_t>(count), dest));
        }
    } uninitialized_relocate_n{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::uninitialized_relocate
    inline constexpr struct uninitialized_relocate_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_relocate_t>
    {
        // clang-format off
        template <typename InIter1, typename InIter2, typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_input_iterator_v<InIter1> &&
                hpx::traits::is_input_iterator_v<InIter2> &&
                hpx::traits::is_forward_iterator_v<FwdIter>
            )>
        friend FwdIter tag_fallback_invoke(hpx::uninitialized_relocate_t,
            InIter1 first, InIter2 last, FwdIter dest) noexcept(
            hpx::traits::pointer_relocate_category<InIter1,
            FwdIter>::is_noexcept_relocatable_v
        )
        // clang-format on
        {
            static_assert(hpx::traits::pointer_relocate_category<InIter1,
                              FwdIter>::valid_relocation,
                "Relocating from this source type to this destination type is "
                "ill-formed");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(std::distance(first, last)))
            {
                return dest;
            }

            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_relocate<
                    parallel::util::in_out_result<InIter1, FwdIter>>()
                    .call(hpx::execution::seq, first, last, dest));
        }

        // clang-format off
        template <typename ExPolicy, typename InIter1, typename InIter2, typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_input_iterator_v<InIter1> &&
                hpx::traits::is_input_iterator_v<InIter2> &&
                hpx::traits::is_forward_iterator_v<FwdIter>
            )>
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(hpx::uninitialized_relocate_t, ExPolicy&& policy,
            InIter1 first, InIter2 last, FwdIter dest) noexcept(
            hpx::traits::pointer_relocate_category<InIter1,
            FwdIter>::is_noexcept_relocatable_v)
        // clang-format on
        {
            static_assert(hpx::traits::pointer_relocate_category<InIter1,
                              FwdIter>::valid_relocation,
                "Relocating from this source type to this destination type is "
                "ill-formed");

            auto count = std::distance(first, last);

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(HPX_MOVE(dest));
            }

            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_relocate_n<
                    parallel::util::in_out_result<InIter1, FwdIter>>()
                    .call(HPX_FORWARD(ExPolicy, policy), first, count, dest));
        }
    } uninitialized_relocate{};
}    // namespace hpx
#endif    // DOXYGEN