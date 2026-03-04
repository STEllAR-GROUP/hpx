//  Copyright (c) 2026 Arivoli Ramamoorthy
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/modules/execution.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/parallel/algorithms/detail/search.hpp>
#include <hpx/parallel/datapar/iterator_helpers.hpp>
#include <hpx/parallel/datapar/loop.hpp>
#include <hpx/parallel/datapar/zip_iterator.hpp>
#include <hpx/parallel/util/cancellation_token.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace hpx::parallel::detail {

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_CORE_EXPORT template <typename ExPolicy, typename Iter1,
        typename Iter2, typename Token, typename Pred, typename Proj1,
        typename Proj2>
        requires(hpx::is_vectorpack_execution_policy_v<ExPolicy>)
    HPX_HOST_DEVICE HPX_FORCEINLINE void tag_invoke(
        sequential_search_t<ExPolicy>, Iter1 it, Iter2 s_first,
        std::size_t base_idx, std::size_t part_size, std::size_t diff,
        std::size_t count, Token& tok, Pred&& op, Proj1&& proj1, Proj2&& proj2)
    {
        if constexpr (hpx::parallel::util::detail::iterator_datapar_compatible<
                          Iter1>::value &&
            hpx::parallel::util::detail::iterator_datapar_compatible<
                Iter2>::value)
        {
            std::size_t idx = 0;
            util::loop_idx_n<hpx::execution::parallel_policy>(base_idx, it,
                part_size, tok,
                [=, &tok, &op, &proj1, &proj2, &idx](
                    auto, std::size_t i) -> void {
                    auto begin =
                        hpx::util::zip_iterator(it + idx, s_first);
                    ++idx;
                    util::cancellation_token<> local_tok;
                    util::loop_n<hpx::execution::simd_policy>(begin, diff,
                        local_tok,
                        [&op, &proj1, &proj2,
                            &local_tok](auto t) -> void {
                            using hpx::get;
                            if (!hpx::parallel::traits::all_of(
                                    HPX_INVOKE(op,
                                        HPX_INVOKE(proj1, get<0>(*t)),
                                        HPX_INVOKE(proj2, get<1>(*t)))))
                            {
                                local_tok.cancel();
                            }
                        });
                    if (!local_tok.was_cancelled())
                        tok.cancel(i);
                });
        }
        else
        {
            using base_policy_type =
                decltype((hpx::execution::experimental::to_non_simd(
                    std::declval<ExPolicy>())));
            return sequential_search_t<base_policy_type>{}(it, s_first,
                base_idx, part_size, diff, count, tok, HPX_FORWARD(Pred, op),
                HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));
        }
    }

}    // namespace hpx::parallel::detail

#endif
