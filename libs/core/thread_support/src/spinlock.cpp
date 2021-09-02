////////////////////////////////////////////////////////////////////////////////
//  Copyright 2008, 2020 Peter Dimov
//  Copyright (c) 2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/local/config/compiler_fence.hpp>
#include <hpx/thread_support/spinlock.hpp>

#include <chrono>
#include <thread>

namespace hpx { namespace util { namespace detail {

    void spinlock::yield_k(unsigned k) noexcept
    {
        // Experiments on Windows and Fedora 32 show that a single pause,
        // followed by an immediate sleep, is best.

        if (k == 0)
        {
            HPX_SMT_PAUSE;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }

}}}    // namespace hpx::util::detail
