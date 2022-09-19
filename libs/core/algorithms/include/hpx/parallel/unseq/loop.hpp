//  Copyright (c) 2022 A Kishore Kumar
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        struct unseq_loop_n
        {
            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static InIter call(
                InIter it, std::size_t count, F&& f)
            {
                // clang-format off
                HPX_IVDEP HPX_UNROLL HPX_VECTORIZE
                for (std::size_t i = 0; i < count; i++)
                {
                    HPX_INVOKE(f, it);
                    ++it;
                }
                // clang-format on
                return it;
            }

            template <typename InIter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static InIter call(
                InIter it, std::size_t count, CancelToken& tok,
                F&& f)
            {
                // clang-format off
                HPX_IVDEP HPX_UNROLL HPX_VECTORIZE
                for (std::size_t i = 0; i < count; i++)
                {
                    HPX_INVOKE(f, it);
                    if (tok.was_cancelled())
                        return it;
                    ++it;
                }
                // clang-format on
                return it;
            }
        };

        struct unseq_loop_n_ind
        {
            template <typename InIter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static InIter call(
                InIter it, std::size_t count, F&& f)
            {
                // clang-format off
                HPX_IVDEP HPX_UNROLL HPX_VECTORIZE
                for (std::size_t i = 0; i < count; i++)
                {
                    HPX_INVOKE(f, *it);
                    ++it;
                }
                // clang-format on
                return it;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct unseq_loop
        {
            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<Begin>, Begin>::type
            call(Begin it, End end, F&& f)
            {
                return unseq_loop_n::call(
                    it, std::distance(it, end), HPX_FORWARD(F, f));
            }

            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<Begin>, Begin>::type
            call(Begin it, End end, F&& f)
            {
                for (/* */; it != end; it++)
                {
                    HPX_INVOKE(f, it);
                }
            }

            template <typename Begin, typename End, typename CancelToken,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<Begin>, Begin>::type
            call(Begin it, End end, CancelToken& tok,
                F&& f)
            {
                return unseq_loop_n::call(
                    it, std::distance(it, end), tok, HPX_FORWARD(F, f));
            }

            template <typename Begin, typename End, typename CancelToken,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<Begin>, Begin>::type
            call(Begin it, End end, CancelToken& tok,
                F&& f)
            {
                for (/* */; it != end; ++it)
                {
                    HPX_INVOKE(f, it);
                    if (tok.was_cancelled())
                        return it;
                }
                return it;
            }
        };

        struct unseq_loop_ind
        {
            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<Begin>, Begin>::type
            call(Begin it, End end, F&& f)
            {
                return unseq_loop_n_ind::call(
                    it, std::distance(it, end), HPX_FORWARD(F, f));
            }

            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<Begin>, Begin>::type
            call(Begin it, End end, F&& f)
            {
                for (/* */; it != end; ++it)
                {
                    HPX_INVOKE(f, *it);
                }
                return it;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct unseq_loop2
        {
            template <typename InIter1, typename InIter2, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                hpx::traits::is_random_access_iterator_v<InIter1> and
                    hpx::traits::is_random_access_iterator_v<InIter2>,
                std::pair<InIter1, InIter2>>::type
            call(InIter1 it1, InIter1 last1,
                InIter2 it2, F&& f)
            {
                std::size_t count = std::distance(it1, last1);
                // clang-format off
                HPX_IVDEP HPX_UNROLL HPX_VECTORIZE
                for (std::size_t i = 0; i < count; i++)
                {
                    HPX_INVOKE(f, it1, it2);
                    ++it1, ++it2;
                }
                // clang-format on
                return std::make_pair(HPX_MOVE(it1), HPX_MOVE(it2));
            }

            template <typename InIter1, typename InIter2, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static typename std::enable_if<
                !hpx::traits::is_random_access_iterator_v<InIter1> or
                    !hpx::traits::is_random_access_iterator_v<InIter2>,
                std::pair<InIter1, InIter2>>::type
            call(InIter1 it1, InIter1 last1,
                InIter2 it2, F&& f)
            {
                for (/* */; it1 != last1; it1++, it2++)
                {
                    HPX_INVOKE(f, it1, it2);
                }
                return std::make_pair(HPX_MOVE(it1), HPX_MOVE(it2));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct unseq_loop_idx_n
        {
            template <typename Iter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static Iter call(
                std::size_t base_idx, Iter it, std::size_t count,
                F&& f)
            {
                // clang-format off
                HPX_IVDEP HPX_UNROLL HPX_VECTORIZE
                for (std::size_t i = 0; i < count; i++)
                {
                    HPX_INVOKE(f, *it, base_idx);
                    ++it, ++base_idx;
                }
                // clang-format on
                return it;
            }

            template <typename Iter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static Iter call(
                std::size_t base_idx, Iter it, std::size_t count,
                CancelToken& tok, F&& f)
            {
                HPX_IVDEP HPX_UNROLL HPX_VECTORIZE for (std::size_t i = 0;
                                                        i < count; i++)
                {
                    if (tok.was_cancelled(base_idx))
                    {
                        break;
                    }
                    HPX_INVOKE(f, *it, base_idx);
                    ++it, ++base_idx;
                }
                return it;
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::unsequenced_policy,
        Begin begin, End end, F&& f)
    {
        return detail::unseq_loop::call(begin, end, HPX_FORWARD(F, f));
    }

    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::unsequenced_task_policy,
        Begin begin, End end, F&& f)
    {
        return detail::unseq_loop::call(begin, end, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Begin, typename End, typename CancelToken, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::unsequenced_policy,
        Begin begin, End end, CancelToken& tok, F&& f)
    {
        return detail::unseq_loop::call(
            begin, end, tok, HPX_FORWARD(F, f));
    }

    template <typename Begin, typename End, typename CancelToken, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_t, hpx::execution::unsequenced_task_policy,
        Begin begin, End end, CancelToken& tok, F&& f)
    {
        return detail::unseq_loop::call(
            begin, end, tok, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_ind_t, hpx::execution::unsequenced_policy,
        Begin begin, End end, F&& f)
    {
        return detail::unseq_loop_ind::call(
            begin, end, HPX_FORWARD(F, f));
    }

    template <typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE Begin tag_invoke(
        hpx::parallel::util::loop_ind_t,
        hpx::execution::unsequenced_task_policy, Begin begin, End end, F&& f)
    {
        return detail::unseq_loop_ind::call(
            begin, end, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter1, typename Iter2, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>,
        std::pair<Iter1, Iter2>>::type
    tag_invoke(hpx::parallel::util::loop2_t<ExPolicy>, Iter1 first1,
        Iter1 last1, Iter2 first2, F&& f)
    {
        return detail::unseq_loop2::call(
            first1, last1, first2, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>, Iter>::type
    tag_invoke(hpx::parallel::util::loop_n_t<ExPolicy>, Iter it,
        std::size_t count, F&& f)
    {
        return hpx::parallel::util::detail::unseq_loop_n::call(
            it, count, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>, Iter>::type
    tag_invoke(hpx::parallel::util::loop_n_t<ExPolicy>, Iter it,
        std::size_t count, CancelToken& tok, F&& f)
    {
        return hpx::parallel::util::detail::unseq_loop_n::call(
            it, count, tok, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>, Iter>::type
    tag_invoke(hpx::parallel::util::loop_n_ind_t<ExPolicy>, Iter it,
        std::size_t count, F&& f)
    {
        return hpx::parallel::util::detail::unseq_loop_n_ind::call(
            it, count, HPX_FORWARD(F, f));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>, Iter>::type
    tag_invoke(hpx::parallel::util::loop_idx_n_t<ExPolicy>,
        std::size_t base_idx, Iter it, std::size_t count, F&& f)
    {
        return hpx::parallel::util::detail::unseq_loop_idx_n::call(
            base_idx, it, count, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr typename std::enable_if<
        hpx::is_unsequenced_execution_policy_v<ExPolicy>, Iter>::type
    tag_invoke(hpx::parallel::util::loop_idx_n_t<ExPolicy>,
        std::size_t base_idx, Iter it, std::size_t count, CancelToken& tok,
        F&& f)
    {
        return hpx::parallel::util::detail::unseq_loop_idx_n::call(
            base_idx, it, count, tok, HPX_FORWARD(F, f));
    }
}}}    // namespace hpx::parallel::util