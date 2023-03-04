//  Copyright (c) 2022 Srinivas Yadav
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/parallel/util/loop.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx::parallel::detail {

    template <typename ExPolicy>
    struct sequential_reduce_t final
      : hpx::functional::detail::tag_fallback<sequential_reduce_t<ExPolicy>>
    {
    private:
        template <typename InIterB, typename InIterE, typename T,
            typename Reduce>
        friend constexpr T tag_fallback_invoke(sequential_reduce_t, ExPolicy&&,
            InIterB first, InIterE last, T init, Reduce&& r)
        {
            util::loop_ind<ExPolicy>(
                first, last, [&init, &r](auto const& v) mutable {
                    init = HPX_INVOKE(r, init, v);
                });
            return init;
        }

        template <typename T, typename FwdIterB, typename Reduce>
        friend constexpr T tag_fallback_invoke(sequential_reduce_t,
            FwdIterB part_begin, std::size_t part_size, T init, Reduce r)
        {
            util::loop_n_ind<ExPolicy>(
                part_begin, part_size, [&init, &r](auto const& v) mutable {
                    init = HPX_INVOKE(r, init, v);
                });
            return init;
        }

        template <typename Iter, typename Sent, typename T, typename Reduce,
            typename Convert>
        friend constexpr auto tag_fallback_invoke(sequential_reduce_t,
            ExPolicy&&, Iter first, Sent last, T init, Reduce&& r,
            Convert&& conv)
        {
            util::loop_ind<std::decay_t<ExPolicy>>(
                first, last, [&init, &r, &conv](auto const& v) mutable {
                    init = HPX_INVOKE(r, init, HPX_INVOKE(conv, v));
                });
            return init;
        }

        template <typename T, typename Iter, typename Reduce, typename Convert>
        friend constexpr auto tag_fallback_invoke(sequential_reduce_t,
            Iter part_begin, std::size_t part_size, T init, Reduce r,
            Convert conv)
        {
            util::loop_n_ind<ExPolicy>(part_begin, part_size,
                [&init, &r, &conv](auto const& v) mutable {
                    init = HPX_INVOKE(r, init, HPX_INVOKE(conv, v));
                });
            return init;
        }

        template <typename Iter1, typename Sent, typename Iter2, typename T,
            typename Reduce, typename Convert>
        friend constexpr T tag_fallback_invoke(sequential_reduce_t,
            Iter1 first1, Sent last1, Iter2 first2, T init, Reduce&& r,
            Convert&& conv)
        {
            util::loop2<ExPolicy>(first1, last1, first2,
                [&init, &r, &conv](Iter1 it1, Iter2 it2) mutable {
                    init = HPX_INVOKE(r, init, HPX_INVOKE(conv, *it1, *it2));
                });
            return init;
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr sequential_reduce_t<ExPolicy> sequential_reduce =
        sequential_reduce_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename... Args>
    HPX_HOST_DEVICE HPX_FORCEINLINE auto sequential_reduce(Args&&... args)
    {
        return sequential_reduce_t<ExPolicy>{}(std::forward<Args>(args)...);
    }
#endif

}    // namespace hpx::parallel::detail
