//  Copyright (c) 2014-2023 Hartmut Kaiser
//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/uninitialized_relocate.hpp
/// \page hpx::uninitialized_relocate, hpx::uninitialized_relocate_n
/// \headerfile hpx/algorithm.hpp

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

    /// Relocates the elements in the range defined by [first, last), to an
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
    /// uninitialized memory area ending at \a dest_last. The objects are
    /// processed in reverse order. If an exception is thrown during the
    /// the move-construction of an element, all elements left in the
    /// input range, as well as all objects already constructed in the
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
    /// \tparam BiIter1     The type of the source range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     Bidirectional iterator.
    /// \tparam BiIter2     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     Bidirectional iterator.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest_last    Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a uninitialized_relocate algorithm invoked
    /// without an execution policy object will execute in sequential order in
    /// the calling thread.
    ///
    /// \returns  The \a uninitialized_relocate_backward algorithm returns \a BiIter2.
    ///           The \a uninitialized_relocate_backward algorithm returns the
    ///           bidirectional iterator to the first element in the destination range.
    ///
    template <typename BiIter1, typename BiIter2>
    BiIter2 uninitialized_relocate_backward(
        BiIter1 first, BiIter1 last, BiIter2 dest_last);

    /// Relocates the elements in the range, defined by [first, last), to an
    /// uninitialized memory area ending at \a dest_last. The order of the
    /// relocation of the objects depends on the execution policy. If an
    /// exception is thrown during the  the move-construction of an element,
    /// all elements left in the input range, as well as all objects already
    /// constructed in the destination range are destroyed. After this algorithm
    /// completes, the source range should be freed or reused without destroying
    /// the objects.
    ///
    /// \note   Using the \a uninitialized_relocate_backward algorithm with the
    ///         with a non-sequenced execution policy, will not guarantee the
    ///         order of the relocation of the objects.
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
    /// \tparam BiIter1     The type of the source range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     Bidirectional iterator.
    /// \tparam BiIter2     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     Bidirectional iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest_last    Refers to the end of the destination range.
    ///
    /// The assignments in the parallel \a uninitialized_relocate_backward algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_relocate_backward algorithm returns a
    ///           \a hpx::future<FwdIter>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a BiIter2 otherwise.
    ///           The \a uninitialized_relocate_backward algorithm returns the
    ///           bidirectional iterator to the first element in the destination
    ///           range.
    ///
    template <typename ExPolicy, typename BiIter1, typename BiIter2>
    hpx::parallel::util::detail::algorithm_result<ExPolicy, BiIter2>
    uninitialized_relocate_backward(
        ExPolicy&& policy, BiIter1 first, BiIter1 last, BiIter2 dest_last);

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
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/type_support/is_contiguous_iterator.hpp>
#include <hpx/type_support/uninitialized_relocation_primitives.hpp>

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

        template <typename ExPolicy, typename InIter, typename FwdIter,
            typename Size>
        // clang-format off
            requires(hpx::traits::is_input_iterator_v<InIter> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                std::is_integral_v<Size>)
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

            using zip_iter = ::hpx::util::zip_iterator<InIter, FwdIter>;
            using partition_result_type = std::pair<FwdIter, FwdIter>;

            return util::partitioner_with_cleanup<ExPolicy,
                util::in_out_result<InIter, FwdIter>, partition_result_type>::
                call(
                    HPX_FORWARD(ExPolicy, policy), zip_iter(first, dest), count,
                    [](zip_iter t, std::size_t part_size) mutable
                        -> partition_result_type {
                        using hpx::get;

                        auto iters = t.get_iterator_tuple();

                        InIter part_source = get<0>(iters);
                        FwdIter part_dest = get<1>(iters);

                        auto [part_source_advanced, part_dest_advanced] =
                            hpx::experimental::util::
                                uninitialized_relocate_n_primitive(
                                    part_source, part_size, part_dest);

                        // returns (dest begin, dest end)
                        return std::make_pair(part_dest, part_dest_advanced);
                    },
                    // finalize, called once if no error occurred
                    [first, dest, count](auto&& data) mutable
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

            template <typename ExPolicy, typename InIter, typename FwdIter>
            // clang-format off
                requires (
                    hpx::is_sequenced_execution_policy_v<ExPolicy> &&
                    hpx::traits::is_input_iterator_v<InIter> &&
                    hpx::traits::is_forward_iterator_v<FwdIter>
                )
            // clang-format on
            static util::in_out_result<InIter, FwdIter> sequential(ExPolicy&&,
                InIter first, std::size_t count, FwdIter dest) noexcept(hpx::
                    experimental::util::detail::relocation_traits<InIter,
                        FwdIter>::is_noexcept_relocatable_v)
            {
                auto [first_advanced, dest_advanced] =
                    hpx::experimental::util::uninitialized_relocate_n_primitive(
                        first, count, dest);

                return util::in_out_result<InIter, FwdIter>{
                    first_advanced, dest_advanced};
            }

            template <typename ExPolicy, typename InIter, typename FwdIter>
            // clang-format off
                requires (
                    hpx::is_execution_policy_v<ExPolicy> &&
                    hpx::traits::is_input_iterator_v<InIter> &&
                    hpx::traits::is_forward_iterator_v<FwdIter>
                )
            // clang-format on
            static util::detail::algorithm_result_t<ExPolicy,
                util::in_out_result<InIter, FwdIter>>
            parallel(ExPolicy&& policy, InIter first, std::size_t count,
                FwdIter dest) noexcept(hpx::experimental::util::detail::
                    relocation_traits<InIter,
                        FwdIter>::is_noexcept_relocatable_v)
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
                typename FwdIter>
            // clang-format off
                requires (
                    hpx::is_sequenced_execution_policy_v<ExPolicy>&&
                    hpx::traits::is_input_iterator_v<InIter1>&&
                    hpx::traits::is_input_iterator_v<InIter2>&&
                    hpx::traits::is_forward_iterator_v<FwdIter>
                )
                //  clang-format on
            static util::in_out_result<InIter1, FwdIter> sequential(
                ExPolicy&&, InIter1 first, InIter2 last,
                FwdIter dest) noexcept(hpx::experimental::util::detail::relocation_traits<
                    InIter1, FwdIter>::is_noexcept_relocatable_v)
            {
                auto [first_advanced, dest_advanced] =
                    hpx::experimental::util::uninitialized_relocate_primitive(
                        first, last, dest);

                return util::in_out_result<InIter1, FwdIter>{first_advanced,
                                dest_advanced};
            }

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename FwdIter>
            // clang-format off
                requires (
                    hpx::is_execution_policy_v<ExPolicy>&&
                    hpx::traits::is_input_iterator_v<InIter1>&&
                    hpx::traits::is_input_iterator_v<InIter2>&&
                    hpx::traits::is_forward_iterator_v<FwdIter>
                )
            // clang-format on
            static util::detail::algorithm_result_t<ExPolicy,
                util::in_out_result<InIter1, FwdIter>>
            parallel(ExPolicy&& policy, InIter1 first, InIter2 last,
                FwdIter dest) noexcept(hpx::experimental::util::detail::
                    relocation_traits<InIter1,
                        FwdIter>::is_noexcept_relocatable_v)
            {
                auto count = std::distance(first, last);

                return parallel_uninitialized_relocate_n(
                    HPX_FORWARD(ExPolicy, policy), first, count, dest);
            }
        };
        /// \endcond

        /////////////////////////////////////////////////////////////////////////////
        // uninitialized_relocate_backward
        /// \cond NOINTERNAL
        template <typename IterPair>
        struct uninitialized_relocate_backward
          : public algorithm<uninitialized_relocate_backward<IterPair>,
                IterPair>
        {
            constexpr uninitialized_relocate_backward() noexcept
              : algorithm<uninitialized_relocate_backward, IterPair>(
                    "uninitialized_relocate_backward")
            {
            }

            // non vectorized overload
            template <typename ExPolicy, typename BiIter1, typename BiIter2>
            // clang-format off
                requires (
                    hpx::is_sequenced_execution_policy_v<ExPolicy> &&
                    hpx::traits::is_bidirectional_iterator_v<BiIter1> &&
                    hpx::traits::is_bidirectional_iterator_v<BiIter2>
                )
            //  clang-format on
            static util::in_out_result<BiIter1, BiIter2> sequential(
                ExPolicy&&, BiIter1 first, BiIter1 last,
                BiIter2 dest_last) noexcept(hpx::experimental::util::detail::
                relocation_traits<BiIter1, BiIter2>::is_noexcept_relocatable_v)
            {
                auto [last_advanced, dest_last_advanced] =
                hpx::experimental::util::uninitialized_relocate_backward_primitive(
                        first, last, dest_last);

                return util::in_out_result<BiIter1, BiIter2>{last_advanced,
                            dest_last_advanced};
            }

            template <typename ExPolicy, typename BiIter1, typename BiIter2>
            // clang-format off
                requires (
                    hpx::is_execution_policy_v<ExPolicy>&&
                    hpx::traits::is_bidirectional_iterator_v<BiIter1>&&
                    hpx::traits::is_bidirectional_iterator_v<BiIter2>
                )
            // clang-format on
            static util::detail::algorithm_result_t<ExPolicy,
                util::in_out_result<BiIter1, BiIter2>>
            parallel(ExPolicy&& policy, BiIter1 first, BiIter1 last,
                BiIter2 dest_last) noexcept(hpx::experimental::util::detail::
                    relocation_traits<BiIter1,
                        BiIter2>::is_noexcept_relocatable_v)
            {
                auto count = std::distance(first, last);

                auto dest_first = std::prev(dest_last, count);

                return parallel_uninitialized_relocate_n(
                    HPX_FORWARD(ExPolicy, policy), first, count, dest_first);
            }
        };
        /// \endcond

    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx::experimental {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::uninitialized_relocate_n
    inline constexpr struct uninitialized_relocate_n_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_relocate_n_t>
    {
        template <typename InIter, typename Size, typename FwdIter>
        // clang-format off
            requires (
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )
        // clang-format on
        friend FwdIter tag_fallback_invoke(uninitialized_relocate_n_t,
            InIter first, Size count,
            FwdIter dest) noexcept(util::detail::relocation_traits<InIter,
            FwdIter>::is_noexcept_relocatable_v)
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "The 'first' argument must meet the requirements "
                "of an input iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "The 'dest' argument must meet the requirements of a forward "
                "iterator.");
            static_assert(util::detail::relocation_traits<InIter,
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

        template <typename ExPolicy, typename InIter, typename Size,
            typename FwdIter>
        // clang-format off
            requires (
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(uninitialized_relocate_n_t, ExPolicy&& policy,
            InIter first, Size count,
            FwdIter dest) noexcept(util::detail::relocation_traits<InIter,
            FwdIter>::is_noexcept_relocatable_v)
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "The 'first' argument must meet the requirements "
                "of an input iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "The 'dest' argument must meet the requirements of a forward "
                "iterator.");
            static_assert(util::detail::relocation_traits<InIter,
                              FwdIter>::valid_relocation,
                "Relocating from this source type to this destination type is "
                "ill-formed");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(HPX_MOVE(dest));
            }

            // If running in non-sequenced execution policy, we must check
            // that the ranges are not overlapping in the left
            if constexpr (!hpx::is_sequenced_execution_policy_v<ExPolicy>)
            {
                // if we can check for overlapping ranges
                if constexpr (hpx::traits::is_contiguous_iterator_v<InIter> &&
                    hpx::traits::is_contiguous_iterator_v<FwdIter>)
                {
                    auto dest_last = std::next(dest, count);
                    auto last = std::next(first, count);
                    // if it is not overlapping in the left direction
                    if (!((first < dest_last) && (dest_last < last)))
                    {
                        // use parallel version
                        return parallel::util::get_second_element(
                            hpx::parallel::detail::uninitialized_relocate_n<
                                parallel::util::in_out_result<InIter,
                                    FwdIter>>()
                                .call(HPX_FORWARD(ExPolicy, policy), first,
                                    count, dest));
                    }
                    // if it is we continue to use the sequential version
                }
                // else we assume that the ranges are overlapping, and continue
                // to use the sequential version
            }

            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_relocate_n<
                    parallel::util::in_out_result<InIter, FwdIter>>()
                    .call(hpx::execution::seq, first,
                        static_cast<std::size_t>(count), dest));
        }
    } uninitialized_relocate_n{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::uninitialized_relocate
    inline constexpr struct uninitialized_relocate_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_relocate_t>
    {
        template <typename InIter1, typename InIter2, typename FwdIter>
        // clang-format off
            requires (
                hpx::traits::is_iterator_v<InIter1> &&
                hpx::traits::is_iterator_v<InIter2> &&
                hpx::traits::is_iterator_v<FwdIter>
            )
        // clang-format on
        friend FwdIter tag_fallback_invoke(uninitialized_relocate_t,
            InIter1 first, InIter2 last,
            FwdIter dest) noexcept(util::detail::relocation_traits<InIter1,
            FwdIter>::is_noexcept_relocatable_v)
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter1> &&
                    hpx::traits::is_input_iterator_v<InIter2>,
                "The 'first' and 'last' arguments must meet the requirements "
                "of input iterators.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "The 'dest' argument must meet the requirements of a forward "
                "iterator.");
            static_assert(util::detail::relocation_traits<InIter1,
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

        template <typename ExPolicy, typename InIter1, typename InIter2,
            typename FwdIter>
        // clang-format off
            requires (
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<InIter1> &&
                hpx::traits::is_iterator_v<InIter2> &&
                hpx::traits::is_iterator_v<FwdIter>
            )
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(uninitialized_relocate_t, ExPolicy&& policy,
            InIter1 first, InIter2 last,
            FwdIter dest) noexcept(util::detail::relocation_traits<InIter1,
            FwdIter>::is_noexcept_relocatable_v)
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter1> &&
                    hpx::traits::is_input_iterator_v<InIter2>,
                "The 'first' and 'last' arguments must meet the requirements "
                "of input iterators.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "The 'dest' argument must meet the requirements of a forward "
                "iterator.");
            static_assert(util::detail::relocation_traits<InIter1,
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

            // If running in non-sequenced execution policy, we must check
            // that the ranges are not overlapping in the left
            if constexpr (!hpx::is_sequenced_execution_policy_v<ExPolicy>)
            {
                // if we can check for overlapping ranges
                if constexpr (hpx::traits::is_contiguous_iterator_v<InIter1> &&
                    hpx::traits::is_contiguous_iterator_v<FwdIter>)
                {
                    auto dest_last = std::next(dest, count);
                    // if it is not overlapping in the left direction
                    if (!((first < dest_last) && (dest_last < last)))
                    {
                        // use parallel version
                        return parallel::util::get_second_element(
                            hpx::parallel::detail::uninitialized_relocate_n<
                                parallel::util::in_out_result<InIter1,
                                    FwdIter>>()
                                .call(HPX_FORWARD(ExPolicy, policy), first,
                                    count, dest));
                    }
                    // if it is we continue to use the sequential version
                }
                // else we assume that the ranges are overlapping, and continue
                // to use the sequential version
            }

            // sequential execution policy
            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_relocate_n<
                    parallel::util::in_out_result<InIter1, FwdIter>>()
                    .call(hpx::execution::seq, first, count, dest));
        }
    } uninitialized_relocate{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::uninitialized_relocate_backward
    inline constexpr struct uninitialized_relocate_backward_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_relocate_backward_t>
    {
        template <typename BiIter1, typename BiIter2>
        // clang-format off
            requires (
                hpx::traits::is_iterator_v<BiIter1> &&
                hpx::traits::is_iterator_v<BiIter2>
            )
        // clang-format on
        friend BiIter2 tag_fallback_invoke(uninitialized_relocate_backward_t,
            BiIter1 first, BiIter1 last,
            BiIter2 dest_last) noexcept(util::detail::relocation_traits<BiIter1,
            BiIter2>::is_noexcept_relocatable_v)
        {
            static_assert(hpx::traits::is_bidirectional_iterator_v<BiIter1> &&
                "The 'first' and 'last' arguments must meet the requirements "
                "of bidirectional iterators.");
            static_assert(hpx::traits::is_bidirectional_iterator_v<BiIter2>,
                "The 'dest_last' argument must meet the requirements of a "
                "bidirectional iterator.");
            static_assert(util::detail::relocation_traits<BiIter1,
                              BiIter2>::valid_relocation,
                "Relocating from this source type to this destination type is "
                "ill-formed");
            // if count is representing a negative value, we do nothing
            if (first == last)
            {
                return dest_last;
            }

            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_relocate_backward<
                    parallel::util::in_out_result<BiIter1, BiIter2>>()
                    .call(hpx::execution::seq, first, last, dest_last));
        }

        template <typename ExPolicy, typename BiIter1, typename BiIter2>
        // clang-format off
            requires (
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<BiIter1> &&
                hpx::traits::is_iterator_v<BiIter2>
            )
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            BiIter2>::type
        tag_fallback_invoke(uninitialized_relocate_backward_t,
            ExPolicy&& policy, BiIter1 first, BiIter1 last,
            BiIter2 dest_last) noexcept(util::detail::relocation_traits<BiIter1,
            BiIter2>::is_noexcept_relocatable_v)
        {
            static_assert(hpx::traits::is_bidirectional_iterator_v<BiIter1>,
                "The 'first' and 'last' arguments must meet the requirements "
                "of bidirectional iterators.");
            static_assert(hpx::traits::is_bidirectional_iterator_v<BiIter2>,
                "The 'dest' argument must meet the requirements of a "
                "bidirectional iterator.");
            static_assert(util::detail::relocation_traits<BiIter1,
                              BiIter2>::valid_relocation,
                "Relocating from this source type to this destination type is "
                "ill-formed");

            auto count = std::distance(first, last);

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return parallel::util::detail::algorithm_result<ExPolicy,
                    BiIter2>::get(HPX_MOVE(dest_last));
            }

            // If running in non-sequence execution policy, we must check
            // that the ranges are not overlapping in the right
            if constexpr (!hpx::is_sequenced_execution_policy_v<ExPolicy>)
            {
                // if we can check for overlapping ranges
                if constexpr (hpx::traits::is_contiguous_iterator_v<BiIter1> &&
                    hpx::traits::is_contiguous_iterator_v<BiIter2>)
                {
                    auto dest_first = std::prev(dest_last, count);
                    // if it is not overlapping in the right direction
                    if (!((first < dest_first) && (dest_first < last)))
                    {
                        // use parallel version
                        return parallel::util::get_second_element(
                            hpx::parallel::detail::
                                uninitialized_relocate_backward<parallel::util::
                                        in_out_result<BiIter1, BiIter2>>()
                                    .call(HPX_FORWARD(ExPolicy, policy), first,
                                        last, dest_last));
                    }
                    // if it is we continue to use the sequential version
                }
                // else we assume that the ranges are overlapping, and continue
                // to use the sequential version
            }

            // sequential execution policy
            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_relocate_backward<
                    parallel::util::in_out_result<BiIter1, BiIter2>>()
                    .call(hpx::execution::seq, first, last, dest_last));
        }
    } uninitialized_relocate_backward{};
}    // namespace hpx::experimental
#endif    // DOXYGEN
