//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution/traits/vector_pack_all_any_none.hpp>
#include <hpx/execution/traits/vector_pack_find.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/parallel/algorithms/detail/find.hpp>
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
    struct datapar_find
    {
        template <typename Iterator, typename Sentinel, typename T,
            typename Proj>
        static inline constexpr Iterator call(
            Iterator first, Sentinel last, T const& val, Proj proj)
        {
            int offset = 0;
            util::cancellation_token<> tok;

            auto ret = util::loop_n<ExPolicy>(first, std::distance(first, last),
                tok, [&offset, &val, &tok, &proj](auto const& curr) {
                    auto msk = hpx::util::invoke(proj, *curr) == val;
                    offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                    {
                        tok.cancel();
                    }
                });

            if (tok.was_cancelled())
                std::advance(ret, offset);
            return ret;
        }

        template <typename FwdIter, typename Token, typename T, typename Proj>
        static inline constexpr void call(std::size_t base_idx,
            FwdIter part_begin, std::size_t part_count, Token& tok,
            T const& val, Proj&& proj)
        {
            util::loop_idx_n<ExPolicy>(base_idx, part_begin, part_count, tok,
                [&val, &proj, &tok](auto& v, std::size_t i) -> void {
                    auto msk = hpx::util::invoke(proj, v) == val;
                    int offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                        tok.cancel(i + offset);
                });
        }
    };

    template <typename ExPolicy, typename Iterator, typename Sentinel,
        typename T, typename Proj = util::projection_identity>
    inline constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iterator>
    tag_invoke(sequential_find_t<ExPolicy>, Iterator first, Sentinel last,
        T const& val, Proj proj = Proj())
    {
        return datapar_find<ExPolicy>::call(first, last, val, proj);
    }

    template <typename ExPolicy, typename FwdIter, typename Token, typename T,
        typename Proj>
    inline constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, void>
    tag_invoke(sequential_find_t<ExPolicy>, std::size_t base_idx,
        FwdIter part_begin, std::size_t part_count, Token& tok, T const& val,
        Proj&& proj)
    {
        return datapar_find<ExPolicy>::call(base_idx, part_begin, part_count,
            tok, val, std::forward<Proj>(proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_find_if
    {
        template <typename Iterator, typename Sentinel, typename Pred,
            typename Proj>
        static inline constexpr Iterator call(
            Iterator first, Sentinel last, Pred pred, Proj proj)
        {
            int offset = 0;
            util::cancellation_token<> tok;

            auto ret = util::loop_n<ExPolicy>(first, std::distance(first, last),
                tok, [&offset, &pred, &tok, &proj](auto const& curr) {
                    auto msk =
                        hpx::util::invoke(pred, hpx::util::invoke(proj, *curr));
                    offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                    {
                        tok.cancel();
                    }
                });

            if (tok.was_cancelled())
                std::advance(ret, offset);
            return ret;
        }

        template <typename FwdIter, typename Token, typename F, typename Proj>
        static inline constexpr void call(FwdIter part_begin,
            std::size_t part_count, Token& tok, F&& op, Proj&& proj)
        {
            util::loop_n<std::decay_t<ExPolicy>>(part_begin, part_count, tok,
                [&op, &tok, &proj](auto const& curr) {
                    auto msk =
                        hpx::util::invoke(op, hpx::util::invoke(proj, *curr));
                    if (hpx::parallel::traits::any_of(msk))
                    {
                        tok.cancel();
                    }
                });
        }

        template <typename FwdIter, typename Token, typename F, typename Proj>
        static inline constexpr void call(std::size_t base_idx,
            FwdIter part_begin, std::size_t part_count, Token& tok, F&& f,
            Proj&& proj)
        {
            util::loop_idx_n<ExPolicy>(base_idx, part_begin, part_count, tok,
                [&f, &proj, &tok](auto& v, std::size_t i) -> void {
                    auto msk = hpx::util::invoke(f, hpx::util::invoke(proj, v));
                    int offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                        tok.cancel(i + offset);
                });
        }
    };

    template <typename ExPolicy, typename Iterator, typename Sentinel,
        typename Pred, typename Proj = util::projection_identity>
    inline constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iterator>
    tag_invoke(sequential_find_if_t<ExPolicy>, Iterator first, Sentinel last,
        Pred pred, Proj proj = Proj())
    {
        return datapar_find_if<ExPolicy>::call(first, last, pred, proj);
    }

    template <typename ExPolicy, typename FwdIter, typename Token, typename F,
        typename Proj>
    inline constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, void>
    tag_invoke(sequential_find_if_t<ExPolicy>, FwdIter part_begin,
        std::size_t part_count, Token& tok, F&& op, Proj&& proj)
    {
        return datapar_find_if<ExPolicy>::call(part_begin, part_count, tok,
            std::forward<F>(op), std::forward<Proj>(proj));
    }

    template <typename ExPolicy, typename FwdIter, typename Token, typename F,
        typename Proj>
    inline constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, void>
    tag_invoke(sequential_find_if_t<ExPolicy>, std::size_t base_idx,
        FwdIter part_begin, std::size_t part_count, Token& tok, F&& op,
        Proj&& proj)
    {
        return datapar_find_if<ExPolicy>::call(base_idx, part_begin, part_count,
            tok, std::forward<F>(op), std::forward<Proj>(proj));
    }
    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct datapar_find_if_not
    {
        template <typename Iterator, typename Sentinel, typename Pred,
            typename Proj>
        static inline constexpr Iterator call(
            Iterator first, Sentinel last, Pred pred, Proj proj)
        {
            int offset = 0;
            util::cancellation_token<> tok;

            auto ret = util::loop_n<ExPolicy>(first, std::distance(first, last),
                tok, [&offset, &pred, &tok, &proj](auto const& curr) {
                    auto msk = !hpx::util::invoke(
                        pred, hpx::util::invoke(proj, *curr));
                    offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                    {
                        tok.cancel();
                    }
                });

            if (tok.was_cancelled())
                std::advance(ret, offset);
            return ret;
        }

        template <typename FwdIter, typename Token, typename F, typename Proj>
        static inline constexpr void call(FwdIter part_begin,
            std::size_t part_count, Token& tok, F&& op, Proj&& proj)
        {
            util::loop_n<std::decay_t<ExPolicy>>(part_begin, part_count, tok,
                [&op, &tok, &proj](auto const& curr) {
                    auto msk =
                        !hpx::util::invoke(op, hpx::util::invoke(proj, *curr));
                    if (hpx::parallel::traits::any_of(msk))
                    {
                        tok.cancel();
                    }
                });
        }

        template <typename FwdIter, typename Token, typename F, typename Proj>
        static inline constexpr void call(std::size_t base_idx,
            FwdIter part_begin, std::size_t part_count, Token& tok, F&& f,
            Proj&& proj)
        {
            util::loop_idx_n<ExPolicy>(base_idx, part_begin, part_count, tok,
                [&f, &proj, &tok](auto& v, std::size_t i) -> void {
                    auto msk =
                        !hpx::util::invoke(f, hpx::util::invoke(proj, v));
                    int offset = hpx::parallel::traits::find_first_of(msk);
                    if (offset != -1)
                        tok.cancel(i + offset);
                });
        }
    };

    template <typename ExPolicy, typename Iterator, typename Sentinel,
        typename Pred, typename Proj = util::projection_identity>
    inline constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, Iterator>
    tag_invoke(sequential_find_if_not_t<ExPolicy>, Iterator first,
        Sentinel last, Pred pred, Proj proj = Proj())
    {
        return datapar_find_if_not<ExPolicy>::call(first, last, pred, proj);
    }

    template <typename ExPolicy, typename FwdIter, typename Token, typename F,
        typename Proj>
    inline constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, void>
    tag_invoke(sequential_find_if_not_t<ExPolicy>, FwdIter part_begin,
        std::size_t part_count, Token& tok, F&& op, Proj&& proj)
    {
        return datapar_find_if_not<ExPolicy>::call(part_begin, part_count, tok,
            std::forward<F>(op), std::forward<Proj>(proj));
    }

    template <typename ExPolicy, typename FwdIter, typename Token, typename F,
        typename Proj>
    inline constexpr std::enable_if_t<
        hpx::is_vectorpack_execution_policy<ExPolicy>::value, void>
    tag_invoke(sequential_find_if_not_t<ExPolicy>, std::size_t base_idx,
        FwdIter part_begin, std::size_t part_count, Token& tok, F&& op,
        Proj&& proj)
    {
        return datapar_find_if_not<ExPolicy>::call(base_idx, part_begin,
            part_count, tok, std::forward<F>(op), std::forward<Proj>(proj));
    }
}}}}    // namespace hpx::parallel::v1::detail
#endif
