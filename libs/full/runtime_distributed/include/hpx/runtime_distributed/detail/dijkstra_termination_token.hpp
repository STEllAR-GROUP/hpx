//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

namespace hpx::components::server::detail {

    // Pure ring-walk decision logic used while forwarding the Dijkstra
    // termination detection token. Starting at \a locality_id, this walks
    // backwards through the locality ring (locality_id - 1, locality_id - 2,
    // ...) until either the token has been handed off successfully or the walk
    // reaches \a initiating_locality_id. \a locality_id is updated in-place to
    // reflect where the walk stopped, matching the historical behavior of this
    // loop (a subsequent call by the same caller resumes the walk from that
    // point instead of restarting at the top of the ring).
    //
    // The actual network operation is injected through \a send_token so that
    // this ring-walk logic can be exercised without any dependency on AGAS,
    // parcels, or the applier. \a send_token must be invocable as
    // bool(std::uint32_t target_locality_id) and should return true if the
    // token was successfully handed off to that locality.
    //
    // Returns true if the token was handed off to some locality strictly
    // between \a locality_id and \a initiating_locality_id, false if the walk
    // reached \a initiating_locality_id without a successful send (the caller
    // is then responsible for the fallback of sending directly to
    // \a initiating_locality_id).
    HPX_CXX_EXPORT template <typename SendToken>
    bool dijkstra_forward_token(std::uint32_t& locality_id,
        std::uint32_t const initiating_locality_id, SendToken&& send_token)
    {
        while (locality_id > 0 && locality_id != initiating_locality_id)
        {
            if (send_token(--locality_id))
            {
                return true;
            }
        }
        return false;
    }
}    // namespace hpx::components::server::detail
