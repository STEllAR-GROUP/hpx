//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution/traits/vector_pack_all_any_none.hpp>
#include <hpx/execution/traits/vector_pack_find.hpp>
#include <hpx/executors/datapar/execution_policy.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/parallel/algorithms/detail/adjacent_find.hpp>
#include <hpx/parallel/datapar/iterator_helpers.hpp>
#include <hpx/parallel/datapar/loop.hpp>
#include <hpx/parallel/datapar/zip_iterator.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_adjacent_find
    {
        template <typename InIter, typename Sent_, typename PredProj>
        static inline InIter call(
            InIter first, Sent_ last, PredProj&& pred_projected)
        {
            if (first == last)
                return last;

            InIter next = first;
            ++next;

            auto zip_iter = hpx::util::make_zip_iterator(first, next);
            std::size_t count = std::distance(first, last);
            util::cancellation_token<std::size_t> tok(count);

            call(0, zip_iter, count - 1, tok,
                std::forward<PredProj>(pred_projected));

            std::size_t adj_find_res = tok.get_data();
            if (adj_find_res != count)
                std::advance(first, adj_find_res);
            else
                first = last;

            return first;
        }

        template <typename ZipIter, typename Token, typename PredProj>
        static inline void call(std::size_t base_idx, ZipIter part_begin,
            std::size_t part_count, Token& tok, PredProj&& pred_projected)
        {
            util::loop_idx_n<ExPolicy>(base_idx, part_begin, part_count, tok,
                [&pred_projected, &tok](auto&& t, std::size_t i) {
                    using hpx::get;
                    auto msk = pred_projected(get<0>(t), get<1>(t));
                    int offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                        tok.cancel(i + offset);
                });
        }
    };

    template <typename ExPolicy, typename InIter, typename Sent_,
        typename PredProj,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_vectorpack_execution_policy<ExPolicy>::value&&
                hpx::parallel::util::detail::iterator_datapar_compatible<
                    InIter>::value)>
    inline InIter tag_invoke(sequential_adjacent_find_t<ExPolicy>, InIter first,
        Sent_ last, PredProj&& pred_projected)
    {
        return datapar_adjacent_find<ExPolicy>::call(
            first, last, std::forward<PredProj>(pred_projected));
    }

    template <typename ExPolicy, typename InIter, typename Sent_,
        typename PredProj,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_vectorpack_execution_policy<ExPolicy>::value &&
            !hpx::parallel::util::detail::iterator_datapar_compatible<
                InIter>::value)>
    inline InIter tag_invoke(sequential_adjacent_find_t<ExPolicy>, InIter first,
        Sent_ last, PredProj&& pred_projected)
    {
        return sequential_adjacent_find<typename ExPolicy::base_policy_type>(
            first, last, std::forward<PredProj>(pred_projected));
    }

    template <typename ExPolicy, typename ZipIter, typename Token,
        typename PredProj,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_vectorpack_execution_policy<ExPolicy>::value&&
                hpx::parallel::util::detail::iterator_datapar_compatible<
                    ZipIter>::value)>
    inline void tag_invoke(sequential_adjacent_find_t<ExPolicy>,
        std::size_t base_idx, ZipIter part_begin, std::size_t part_count,
        Token& tok, PredProj&& pred_projected)
    {
        return datapar_adjacent_find<ExPolicy>::call(base_idx, part_begin,
            part_count, tok, std::forward<PredProj>(pred_projected));
    }

    template <typename ExPolicy, typename ZipIter, typename Token,
        typename PredProj,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_vectorpack_execution_policy<ExPolicy>::value &&
            !hpx::parallel::util::detail::iterator_datapar_compatible<
                ZipIter>::value)>
    inline void tag_invoke(sequential_adjacent_find_t<ExPolicy>,
        std::size_t base_idx, ZipIter part_begin, std::size_t part_count,
        Token& tok, PredProj&& pred_projected)
    {
        return sequential_adjacent_find<typename ExPolicy::base_policy_type>(
            base_idx, part_begin, part_count, tok,
            std::forward<PredProj>(pred_projected));
    }
}}}}    // namespace hpx::parallel::v1::detail
#endif
