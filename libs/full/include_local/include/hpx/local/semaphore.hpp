//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/synchronization/counting_semaphore.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
// C++20 counting semaphores

namespace hpx {

    // Semaphores are lightweight synchronization primitives used to constrain
    // concurrent access to a shared resource. They are widely used to
    // implement other synchronization primitives and, whenever both are
    // applicable, can be more efficient than condition variables.

    // A counting semaphore is a semaphore object that models a non - negative
    // resource count. A binary semaphore is a semaphore object that has only
    // two states.
    //
    // Class template counting_semaphore maintains an internal counter that is
    // initialized when the semaphore is created. The counter is decremented
    // when a thread acquires the semaphore, and is incremented when a thread
    // releases the semaphore. If a thread tries to acquire the semaphore when
    // the counter is zero, the thread will block until another thread
    // increments the counter by releasing the semaphore.
    template <std::ptrdiff_t LeastMaxValue = PTRDIFF_MAX>
    using counting_semaphore =
        hpx::lcos::local::cpp20_counting_semaphore<LeastMaxValue>;

    // A binary semaphore should be more efficient than the default
    // implementation of a counting semaphore with a unit resource count.
    using binary_semaphore = hpx::lcos::local::cpp20_binary_semaphore<>;

}    // namespace hpx
