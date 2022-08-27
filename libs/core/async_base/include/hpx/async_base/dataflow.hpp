//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail {
    template <typename FD, typename Enable = void>
    struct dataflow_dispatch;
}}}    // namespace hpx::lcos::detail

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    template <typename F, typename... Ts>
    HPX_FORCEINLINE auto dataflow(F&& f, Ts&&... ts) -> decltype(
        lcos::detail::dataflow_dispatch<typename std::decay<F>::type>::call(
            hpx::util::internal_allocator<>{}, HPX_FORWARD(F, f),
            HPX_FORWARD(Ts, ts)...))
    {
        return lcos::detail::dataflow_dispatch<typename std::decay<F>::type>::
            call(hpx::util::internal_allocator<>{}, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
    }

    template <typename Allocator, typename F, typename... Ts>
    HPX_FORCEINLINE auto dataflow_alloc(
        Allocator const& alloc, F&& f, Ts&&... ts)
        -> decltype(
            lcos::detail::dataflow_dispatch<typename std::decay<F>::type>::call(
                alloc, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
    {
        return lcos::detail::dataflow_dispatch<
            typename std::decay<F>::type>::call(alloc, HPX_FORWARD(F, f),
            HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx

// #endif
