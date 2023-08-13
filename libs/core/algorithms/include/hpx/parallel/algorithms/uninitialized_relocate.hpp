//  Copyright (c) 2014-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/uninitialized_relocate.hpp

#pragma once

#include <hpx/config.hpp>
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

        template <typename ExPolicy, typename Iter, typename FwdIter2>
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<Iter, FwdIter2>>::type
        parallel_uninitialized_relocate_n(
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

            return util::partitioner_with_cleanup<ExPolicy,
                util::in_out_result<Iter, FwdIter2>, partition_result_type>::
                call(
                    HPX_FORWARD(ExPolicy, policy), zip_iterator(first, dest),
                    count,
                    [policy](zip_iterator t, std::size_t part_size) mutable
                    -> partition_result_type {
                        using hpx::get;

                        auto iters = t.get_iterator_tuple();

                        Iter part_source = get<0>(iters);
                        FwdIter2 part_dest = get<1>(iters);

                        // returns (dest begin, dest end)
                        return std::make_pair(part_dest,
                            util::get_second_element(
                                hpx::parallel::util::uninit_relocate_n(
                                    HPX_FORWARD(ExPolicy, policy), part_source,
                                    part_size, part_dest)));
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
            template <typename ExPolicy, typename InIter, typename FwdIter2,
                HPX_CONCEPT_REQUIRES_(
                    hpx::is_sequenced_execution_policy_v<ExPolicy>)>
            static util::in_out_result<InIter, FwdIter2> sequential(
                ExPolicy&& policy, InIter first, std::size_t count,
                FwdIter2 dest)
            {
                return hpx::parallel::util::uninit_relocate_n(
                    HPX_FORWARD(ExPolicy, policy), first, count, dest);
            }

            template <typename ExPolicy, typename Iter, typename FwdIter2>
            static util::detail::algorithm_result_t<ExPolicy,
                util::in_out_result<Iter, FwdIter2>>
            parallel(
                ExPolicy&& policy, Iter first, std::size_t count, FwdIter2 dest)
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
            template <typename ExPolicy, typename InIter, typename FwdIter2,
                HPX_CONCEPT_REQUIRES_(
                    hpx::is_sequenced_execution_policy_v<ExPolicy>)>
            static util::in_out_result<InIter, FwdIter2> sequential(
                ExPolicy&& policy, InIter first, InIter last, FwdIter2 dest)
            {
                auto count = std::distance(first, last);

                return hpx::parallel::util::uninit_relocate_n(
                    HPX_FORWARD(ExPolicy, policy), first, count, dest);
            }

            template <typename ExPolicy, typename InIter, typename FwdIter2>
            static util::detail::algorithm_result_t<ExPolicy,
                util::in_out_result<InIter, FwdIter2>>
            parallel(
                ExPolicy&& policy, InIter first, InIter last, FwdIter2 dest)
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
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(hpx::uninitialized_relocate_n_t,
            InIter first, Size count,
            FwdIter dest) noexcept(std::is_same_v<hpx::traits::
                                                      pointer_relocate_category<
                                                          InIter, FwdIter>,
                                       hpx::traits::
                                           nothrow_relocatable_pointer_tag> ||
            std::is_same_v<
                hpx::traits::pointer_relocate_category<InIter, FwdIter>,
                hpx::traits::trivially_relocatable_pointer_tag>)
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "Required at least input iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");
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
        template <typename ExPolicy, typename FwdIter1, typename Size,
            typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter1> &&
                hpx::traits::is_forward_iterator_v<FwdIter2> &&
                std::is_integral_v<Size>
            )>
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_invoke(hpx::uninitialized_relocate_n_t, ExPolicy&& policy,
            FwdIter1 first, Size count, FwdIter2 dest) noexcept(
            std::is_same_v<
                hpx::traits::pointer_relocate_category<FwdIter1, FwdIter2>,
                hpx::traits::nothrow_relocatable_pointer_tag> ||
            std::is_same_v<
                hpx::traits::pointer_relocate_category<FwdIter1, FwdIter2>,
                hpx::traits::trivially_relocatable_pointer_tag>)
        // clang-format on
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::pointer_relocate_category<FwdIter1,
                              FwdIter2>::valid_relocation,
                "Relocating from this source type to this destination type is "
                "ill-formed");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter2>::get(HPX_MOVE(dest));
            }

            return parallel::util::get_second_element(
                hpx::parallel::detail::uninitialized_relocate_n<
                    parallel::util::in_out_result<FwdIter1, FwdIter2>>()
                    .call(HPX_FORWARD(ExPolicy, policy), first,
                        static_cast<std::size_t>(count), dest));
        }
    } uninitialized_relocate_n{};
}    // namespace hpx