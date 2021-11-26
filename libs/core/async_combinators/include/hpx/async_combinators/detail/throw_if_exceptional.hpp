//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/traits/acquire_shared_state.hpp>

#include <array>
#include <cstddef>
#include <vector>

// This defines facilities that check a set of futures whether any of those are
// exceptional, rethrowing the exception if needed.
namespace hpx::detail {

    template <typename Future>
    void rethrow_if_needed(Future const& f)
    {
        auto shared_state = hpx::traits::detail::get_shared_state(f);
        if (shared_state->has_exception())
        {
            shared_state->get_result_void();    // throws stored exception
        }
    }

    template <typename Future>
    void throw_if_exceptional(std::vector<Future> const& values)
    {
        for (auto const& f : values)
        {
            rethrow_if_needed(f);
        }
    }

    template <typename Future, std::size_t N>
    void throw_if_exceptional(std::array<Future, N> const& values)
    {
        for (auto const& f : values)
        {
            rethrow_if_needed(f);
        }
    }

    template <typename... Ts>
    void throw_if_exceptional(Ts const&... ts)
    {
        int const _sequencer[] = {0, (rethrow_if_needed(ts), 0)...};
        (void) _sequencer;
    }
}    // namespace hpx::detail
