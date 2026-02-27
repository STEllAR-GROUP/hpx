//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//  Copyright (c) 2021 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/modules/type_support.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::parallel::detail {

    // provide implementation of std::find supporting iterators/sentinels
    struct sequential_find_t final
      : hpx::functional::detail::tag_fallback<sequential_find_t>
    {
    private:
        template <typename ExPolicy, typename Iterator, typename Sentinel,
            typename T, typename Proj = hpx::identity>
        friend constexpr Iterator tag_fallback_invoke(sequential_find_t,
            ExPolicy, Iterator first, Sentinel last, T const& value,
            Proj proj = Proj())
        {
            return util::loop_pred<ExPolicy>(
                first, last, [&value, &proj](auto const& curr) {
                    return HPX_INVOKE(proj, *curr) == value;
                });
        }

        template <typename ExPolicy, typename FwdIter, typename Token,
            typename T, typename Proj>
        friend constexpr void tag_fallback_invoke(sequential_find_t, ExPolicy,
            std::size_t base_idx, FwdIter part_begin, std::size_t part_count,
            Token& tok, T const& val, Proj&& proj)
        {
            util::loop_idx_n<ExPolicy>(base_idx, part_begin, part_count, tok,
                [&val, &proj, &tok](auto&& v, std::size_t i) -> void {
                    if (HPX_INVOKE(proj, v) == val)
                    {
                        tok.cancel(i);
                    }
                });
        }
    };

    inline constexpr sequential_find_t sequential_find = sequential_find_t{};

    // provide implementation of std::find_if supporting iterators/sentinels
    struct sequential_find_if_t final
      : hpx::functional::detail::tag_fallback<sequential_find_if_t>
    {
    private:
        template <typename ExPolicy, typename Iterator, typename Sentinel,
            typename Pred, typename Proj = hpx::identity>
        friend inline constexpr Iterator tag_fallback_invoke(
            sequential_find_if_t, ExPolicy, Iterator first, Sentinel last,
            Pred pred, Proj proj = Proj())
        {
            return util::loop_pred<ExPolicy>(
                first, last, [&pred, &proj](auto const& curr) {
                    return HPX_INVOKE(pred, HPX_INVOKE(proj, *curr));
                });
        }

        template <typename ExPolicy, typename FwdIter, typename Token,
            typename F, typename Proj>
        friend inline constexpr void tag_fallback_invoke(sequential_find_if_t,
            ExPolicy, FwdIter part_begin, std::size_t part_count, Token& tok,
            F&& op, Proj&& proj)
        {
            util::loop_n<std::decay_t<ExPolicy>>(part_begin, part_count, tok,
                [&op, &tok, &proj](auto const& curr) {
                    if (HPX_INVOKE(op, HPX_INVOKE(proj, *curr)))
                    {
                        tok.cancel();
                    }
                });
        }

        template <typename ExPolicy, typename FwdIter, typename Token,
            typename F, typename Proj>
        friend inline constexpr void tag_fallback_invoke(sequential_find_if_t,
            ExPolicy, std::size_t base_idx, FwdIter part_begin,
            std::size_t part_count, Token& tok, F&& f, Proj&& proj)
        {
            util::loop_idx_n<ExPolicy>(base_idx, part_begin, part_count, tok,
                [&f, &proj, &tok](auto&& v, std::size_t i) -> void {
                    if (HPX_INVOKE(f, HPX_INVOKE(proj, v)))
                    {
                        tok.cancel(i);
                    }
                });
        }
    };

    inline constexpr sequential_find_if_t sequential_find_if =
        sequential_find_if_t{};

    // provide implementation of std::find_if_not supporting iterators/sentinels
    struct sequential_find_if_not_t final
      : hpx::functional::detail::tag_fallback<sequential_find_if_not_t>
    {
    private:
        template <typename ExPolicy, typename Iterator, typename Sentinel,
            typename Pred, typename Proj = hpx::identity>
        friend inline constexpr Iterator tag_fallback_invoke(
            sequential_find_if_not_t, ExPolicy, Iterator first, Sentinel last,
            Pred pred, Proj proj = Proj())
        {
            return util::loop_pred<ExPolicy>(
                first, last, [&pred, &proj](auto const& curr) {
                    return !HPX_INVOKE(pred, HPX_INVOKE(proj, *curr));
                });
        }

        template <typename ExPolicy, typename FwdIter, typename Token,
            typename F, typename Proj>
        friend inline constexpr void tag_fallback_invoke(
            sequential_find_if_not_t, ExPolicy, FwdIter part_begin,
            std::size_t part_count, Token& tok, F&& op, Proj&& proj)
        {
            util::loop_n<std::decay_t<ExPolicy>>(part_begin, part_count, tok,
                [&op, &tok, &proj](auto const& curr) {
                    if (!HPX_INVOKE(op, HPX_INVOKE(proj, *curr)))
                    {
                        tok.cancel();
                    }
                });
        }

        template <typename ExPolicy, typename FwdIter, typename Token,
            typename F, typename Proj>
        friend inline constexpr void tag_fallback_invoke(
            sequential_find_if_not_t, ExPolicy, std::size_t base_idx,
            FwdIter part_begin, std::size_t part_count, Token& tok, F&& f,
            Proj&& proj)
        {
            util::loop_idx_n<ExPolicy>(base_idx, part_begin, part_count, tok,
                [&f, &proj, &tok](auto&& v, std::size_t i) -> void {
                    if (!HPX_INVOKE(f, HPX_INVOKE(proj, v)))
                    {
                        tok.cancel(i);
                    }
                });
        }
    };

    inline constexpr sequential_find_if_not_t sequential_find_if_not =
        sequential_find_if_not_t{};

    // provide implementation of std::find_end supporting iterators/sentinels
    HPX_CXX_CORE_EXPORT template <typename Iter1, typename Sent1,
        typename Iter2, typename Sent2, typename Pred, typename Proj1,
        typename Proj2>
    constexpr Iter1 sequential_search(Iter1 first1, Sent1 last1, Iter2 first2,
        Sent2 last2, Pred&& op, Proj1&& proj1, Proj2&& proj2)
    {
        for (/**/; /**/; ++first1)
        {
            Iter1 it1 = first1;
            for (Iter2 it2 = first2; /**/; (void) ++it1, ++it2)
            {
                if (it2 == last2)
                {
                    return first1;
                }
                if (it1 == last1)
                {
                    return last1;
                }
                if (!HPX_INVOKE(
                        op, HPX_INVOKE(proj1, *it1), HPX_INVOKE(proj2, *it2)))
                {
                    break;
                }
            }
        }
    }

    struct sequential_find_end_t final
      : hpx::functional::detail::tag_fallback<sequential_find_end_t>
    {
    private:
        template <typename ExPolicy, typename Iter1, typename Sent1,
            typename Iter2, typename Sent2, typename Pred, typename Proj1,
            typename Proj2>
        friend inline constexpr Iter1 tag_fallback_invoke(sequential_find_end_t,
            ExPolicy, Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2,
            Pred&& op, Proj1&& proj1, Proj2&& proj2)
        {
            if (first2 == last2)
            {
                return detail::advance_to_sentinel(first1, last1);
            }

            Iter1 result = last1;
            while (true)
            {
                Iter1 new_result = sequential_search(
                    first1, last1, first2, last2, op, proj1, proj2);

                if (new_result == last1)
                {
                    break;
                }
                else
                {
                    result = new_result;
                    first1 = result;
                    ++first1;
                }
            }
            return result;
        }

        template <typename ExPolicy, typename Iter1, typename Iter2,
            typename Token, typename Pred, typename Proj1, typename Proj2>
        friend inline constexpr void tag_fallback_invoke(sequential_find_end_t,
            ExPolicy, Iter1 it, Iter2 first2, std::size_t base_idx,
            std::size_t part_size, std::size_t diff, Token& tok, Pred&& op,
            Proj1&& proj1, Proj2&& proj2)
        {
            util::loop_idx_n<ExPolicy>(base_idx, it, part_size, tok,
                [=, &tok, &op, &proj1, &proj2](auto t, std::size_t i) -> void {
                    // Note: replacing the invoke() with HPX_INVOKE()
                    // below makes gcc generate errors
                    if (hpx::invoke(op, hpx::invoke(proj1, t),
                            hpx::invoke(proj2, *first2)))
                    {
                        std::size_t local_count = 1;
                        auto mid = t;
                        auto mid2 = first2;
                        ++mid;
                        ++mid2;

                        for (; local_count != diff;
                            ++local_count, ++mid, ++mid2)
                        {
                            // Note: replacing the invoke() with HPX_INVOKE()
                            // below makes gcc generate errors
                            if (!hpx::invoke(op, hpx::invoke(proj1, mid),
                                    hpx::invoke(proj2, *mid2)))
                            {
                                break;
                            }
                        }

                        if (local_count == diff)
                        {
                            tok.cancel(i);
                        }
                    }
                });
        }
    };

    inline constexpr sequential_find_end_t sequential_find_end =
        sequential_find_end_t{};

    struct sequential_find_first_of_t final
      : hpx::functional::detail::tag_fallback<sequential_find_first_of_t>
    {
        template <typename ExPolicy, typename InIter1, typename InIter2,
            typename Pred, typename Proj1, typename Proj2>
        friend inline constexpr InIter1 tag_fallback_invoke(
            sequential_find_first_of_t, ExPolicy, InIter1 first, InIter1 last,
            InIter2 s_first, InIter2 s_last, Pred&& op, Proj1&& proj1,
            Proj2&& proj2)
        {
            if (first == last)
                return last;

            util::compare_projected<Pred, Proj1, Proj2> cmp(
                HPX_FORWARD(Pred, op), HPX_FORWARD(Proj1, proj1),
                HPX_FORWARD(Proj2, proj2));

            for (/* */; first != last; ++first)
            {
                for (InIter2 iter = s_first; iter != s_last; ++iter)
                {
                    if (HPX_INVOKE(cmp, *first, *iter))
                    {
                        return first;
                    }
                }
            }
            return last;
        }

        template <typename ExPolicy, typename FwdIter, typename FwdIter2,
            typename Token, typename Pred, typename Proj1, typename Proj2>
        friend inline constexpr void tag_fallback_invoke(
            sequential_find_first_of_t, ExPolicy, FwdIter it, FwdIter2 s_first,
            FwdIter2 s_last, std::size_t base_idx, std::size_t part_size,
            Token& tok, Pred&& op, Proj1&& proj1, Proj2&& proj2)
        {
            util::loop_idx_n<ExPolicy>(base_idx, it, part_size, tok,
                [&tok, &s_first, &s_last, &op, &proj1, &proj2](
                    auto v, std::size_t i) -> void {
                    util::compare_projected<Pred, Proj1, Proj2> cmp(
                        HPX_FORWARD(Pred, op), HPX_FORWARD(Proj1, proj1),
                        HPX_FORWARD(Proj2, proj2));

                    for (FwdIter2 iter = s_first; iter != s_last; ++iter)
                    {
                        if (HPX_INVOKE(cmp, v, *iter))
                        {
                            tok.cancel(i);
                        }
                    }
                });
        }
    };

    inline constexpr sequential_find_first_of_t sequential_find_first_of =
        sequential_find_first_of_t{};

    ///////////////////////////////////////////////////////////////////////////
    // find_last
    struct sequential_find_last_t final
      : hpx::functional::detail::tag_fallback<sequential_find_last_t>
    {
    private:
        template <typename ExPolicy, typename Iterator, typename Sentinel,
            typename T, typename Proj = hpx::identity>
        friend constexpr Iterator tag_fallback_invoke(sequential_find_last_t,
            ExPolicy, Iterator first, Sentinel last, T const& value,
            Proj&& proj = Proj())
        {
            auto u_last = detail::advance_to_sentinel(first, last);
            if constexpr (std::bidirectional_iterator<Iterator>)
            {
                auto it = u_last;
                while (it != first)
                {
                    --it;
                    if (HPX_INVOKE(proj, *it) == value)
                    {
                        return it;
                    }
                }
                return u_last;
            }
            else
            {
                auto result = u_last;
                for (auto it = first; it != u_last; ++it)
                {
                    if (HPX_INVOKE(proj, *it) == value)
                    {
                        result = it;
                    }
                }
                return result;
            }
        }

        template <typename ExPolicy, typename FwdIter, typename Token,
            typename T, typename Proj>
        friend constexpr void tag_fallback_invoke(sequential_find_last_t,
            ExPolicy, std::size_t base_idx, FwdIter part_begin,
            std::size_t part_count, Token& tok, T const& val, Proj&& proj)
        {
            util::loop_idx_n<ExPolicy>(base_idx, part_begin, part_count, tok,
                [&val, &proj, &tok](auto&& v, std::size_t i) -> void {
                    if (HPX_INVOKE(proj, v) == val)
                    {
                        tok.cancel(i);
                    }
                });
        }
    };

    inline constexpr sequential_find_last_t sequential_find_last =
        sequential_find_last_t{};

    ///////////////////////////////////////////////////////////////////////////
    // find_last_if
    struct sequential_find_last_if_t final
      : hpx::functional::detail::tag_fallback<sequential_find_last_if_t>
    {
    private:
        template <typename ExPolicy, typename Iterator, typename Sentinel,
            typename Pred, typename Proj = hpx::identity>
        friend inline constexpr Iterator tag_fallback_invoke(
            sequential_find_last_if_t, ExPolicy, Iterator first, Sentinel last,
            Pred pred, Proj&& proj = Proj())
        {
            auto u_last = detail::advance_to_sentinel(first, last);
            if constexpr (std::bidirectional_iterator<Iterator>)
            {
                auto it = u_last;
                while (it != first)
                {
                    --it;
                    if (HPX_INVOKE(pred, HPX_INVOKE(proj, *it)))
                    {
                        return it;
                    }
                }
                return u_last;
            }
            else
            {
                auto result = u_last;
                for (auto it = first; it != u_last; ++it)
                {
                    if (HPX_INVOKE(pred, HPX_INVOKE(proj, *it)))
                    {
                        result = it;
                    }
                }
                return result;
            }
        }

        template <typename ExPolicy, typename FwdIter, typename Token,
            typename F, typename Proj>
        friend inline constexpr void tag_fallback_invoke(
            sequential_find_last_if_t, ExPolicy, std::size_t base_idx,
            FwdIter part_begin, std::size_t part_count, Token& tok, F&& f,
            Proj&& proj)
        {
            util::loop_idx_n<ExPolicy>(base_idx, part_begin, part_count, tok,
                [&f, &proj, &tok](auto&& v, std::size_t i) -> void {
                    if (HPX_INVOKE(f, HPX_INVOKE(proj, v)))
                    {
                        tok.cancel(i);
                    }
                });
        }
    };

    inline constexpr sequential_find_last_if_t sequential_find_last_if =
        sequential_find_last_if_t{};

    ///////////////////////////////////////////////////////////////////////////
    // find_last_if_not
    struct sequential_find_last_if_not_t final
      : hpx::functional::detail::tag_fallback<sequential_find_last_if_not_t>
    {
    private:
        template <typename ExPolicy, typename Iterator, typename Sentinel,
            typename Pred, typename Proj = hpx::identity>
        friend inline constexpr Iterator tag_fallback_invoke(
            sequential_find_last_if_not_t, ExPolicy, Iterator first,
            Sentinel last, Pred pred, Proj&& proj = Proj())
        {
            auto u_last = detail::advance_to_sentinel(first, last);
            if constexpr (std::bidirectional_iterator<Iterator>)
            {
                auto it = u_last;
                while (it != first)
                {
                    --it;
                    if (!HPX_INVOKE(pred, HPX_INVOKE(proj, *it)))
                    {
                        return it;
                    }
                }
                return u_last;
            }
            else
            {
                auto result = u_last;
                for (auto it = first; it != u_last; ++it)
                {
                    if (!HPX_INVOKE(pred, HPX_INVOKE(proj, *it)))
                    {
                        result = it;
                    }
                }
                return result;
            }
        }

        template <typename ExPolicy, typename FwdIter, typename Token,
            typename F, typename Proj>
        friend inline constexpr void tag_fallback_invoke(
            sequential_find_last_if_not_t, ExPolicy, std::size_t base_idx,
            FwdIter part_begin, std::size_t part_count, Token& tok, F&& f,
            Proj&& proj)
        {
            util::loop_idx_n<ExPolicy>(base_idx, part_begin, part_count, tok,
                [&f, &proj, &tok](auto&& v, std::size_t i) -> void {
                    if (!HPX_INVOKE(f, HPX_INVOKE(proj, v)))
                    {
                        tok.cancel(i);
                    }
                });
        }
    };

    inline constexpr sequential_find_last_if_not_t sequential_find_last_if_not =
        sequential_find_last_if_not_t{};
}    // namespace hpx::parallel::detail
