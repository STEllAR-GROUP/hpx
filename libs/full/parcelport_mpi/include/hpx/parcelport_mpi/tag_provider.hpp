//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c)      2023 Jiakun Yan
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_MPI)
#include <hpx/assert.hpp>
#include <hpx/modules/synchronization.hpp>

#include <atomic>
#include <deque>
#include <limits>
#include <mutex>

namespace hpx::parcelset::policies::mpi {

    struct tag_provider
    {
        tag_provider()
          : next_tag(0)
        {
        }

        [[nodiscard]] int get_next_tag() noexcept
        {
            // Tag 0 is reserved for header message
            int tag = next_tag.fetch_add(1, std::memory_order_relaxed) %
                    (util::mpi_environment::MPI_MAX_TAG - 1) +
                1;
            return tag;
        }

        std::atomic<int> next_tag;
    };
}    // namespace hpx::parcelset::policies::mpi

#endif
