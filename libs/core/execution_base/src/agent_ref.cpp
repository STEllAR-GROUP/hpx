//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution_base/agent_ref.hpp>
#include <hpx/execution_base/this_thread.hpp>
#include <hpx/lock_registration/detail/register_locks.hpp>
#include <hpx/modules/format.hpp>

#include <cstddef>
#include <ostream>

namespace hpx::execution_base {

    void agent_ref::yield(const char* desc) const
    {
        HPX_ASSERT(*this == hpx::execution_base::this_thread::agent());

        // verify that there are no more registered locks for this OS-thread
        util::verify_no_locks();
        impl_->yield(desc);
    }

    void agent_ref::yield_k(std::size_t k, const char* desc) const
    {
        HPX_ASSERT(*this == hpx::execution_base::this_thread::agent());

        // verify that there are no more registered locks for this OS-thread
        util::verify_no_locks();
        impl_->yield_k(k, desc);
    }

    void agent_ref::suspend(const char* desc) const
    {
        HPX_ASSERT(*this == hpx::execution_base::this_thread::agent());

        // verify that there are no more registered locks for this OS-thread
        util::verify_no_locks();
        impl_->suspend(desc);
    }

    void agent_ref::resume(
        hpx::threads::thread_priority priority, const char* desc) const
    {
        HPX_ASSERT(*this != hpx::execution_base::this_thread::agent());
        impl_->resume(priority, desc);
    }

    void agent_ref::abort(const char* desc) const
    {
        HPX_ASSERT(*this != hpx::execution_base::this_thread::agent());
        impl_->abort(desc);
    }

    void agent_ref::sleep_for(
        hpx::chrono::steady_duration const& sleep_duration,
        const char* desc) const
    {
        HPX_ASSERT(*this == hpx::execution_base::this_thread::agent());
        impl_->sleep_for(sleep_duration, desc);
    }

    void agent_ref::sleep_until(
        hpx::chrono::steady_time_point const& sleep_time,
        const char* desc) const
    {
        HPX_ASSERT(*this == hpx::execution_base::this_thread::agent());
        impl_->sleep_until(sleep_time, desc);
    }

    std::ostream& operator<<(std::ostream& os, agent_ref const& a)
    {
        hpx::util::format_to(os, "agent_ref{{{}}}", a.impl_->description());
        return os;
    }
}    // namespace hpx::execution_base
