////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/modules/lock_registration.hpp>
#include <hpx/modules/thread_support.hpp>
#if defined(HPX_HAVE_MODULE_TRACY)
#include <hpx/modules/tracy.hpp>
#endif

#include <string>
#include <utility>

namespace hpx::util {

    // Lockable spinlock class
    HPX_CXX_EXPORT struct spinlock
    {
    public:
        spinlock(spinlock const&) = delete;
        spinlock(spinlock&&) = delete;
        spinlock& operator=(spinlock const&) = delete;
        spinlock& operator=(spinlock&&) = delete;

    private:
        hpx::util::detail::spinlock m;
#if defined(HPX_HAVE_MODULE_TRACY)
        hpx::tracy::lock_data context_;
#endif

    public:
        spinlock() noexcept
        {
            HPX_ITT_SYNC_CREATE(this, "util::spinlock", nullptr);
#if defined(HPX_HAVE_MODULE_TRACY)
            context_ = hpx::tracy::create("hpx::spinlock");
#endif
        }

        explicit spinlock(char const* desc) noexcept
        {
            HPX_ITT_SYNC_CREATE(this, "util::spinlock", desc);
#if defined(HPX_HAVE_MODULE_TRACY)
            context_ =
                hpx::tracy::create(std::string("util::spinlock#") + desc);
#endif
        }

        ~spinlock()
        {
            HPX_ITT_SYNC_DESTROY(this);
#if defined(HPX_HAVE_MODULE_TRACY)
            hpx::tracy::destroy(context_);
#endif
        }

        void lock() noexcept(
            noexcept(util::register_lock(std::declval<spinlock*>())))
        {
            HPX_ITT_SYNC_PREPARE(this);
#if defined(HPX_HAVE_MODULE_TRACY)
            bool const run_after = hpx::tracy::lock_prepare(context_);
#endif
            m.lock();

            HPX_ITT_SYNC_ACQUIRED(this);
#if defined(HPX_HAVE_MODULE_TRACY)
            if (run_after)
                hpx::tracy::lock_acquired(context_);
#endif
            util::register_lock(this);
        }

        bool try_lock() noexcept(
            noexcept(util::register_lock(std::declval<spinlock*>())))
        {
            HPX_ITT_SYNC_PREPARE(this);
#if defined(HPX_HAVE_MODULE_TRACY)
            bool const run_after = hpx::tracy::lock_prepare(context_);
#endif

            if (m.try_lock())
            {
                HPX_ITT_SYNC_ACQUIRED(this);
#if defined(HPX_HAVE_MODULE_TRACY)
                if (run_after)
                    hpx::tracy::lock_acquired(context_, true);
#endif
                util::register_lock(this);
                return true;
            }
            HPX_ITT_SYNC_CANCEL(this);
#if defined(HPX_HAVE_MODULE_TRACY)
            if (run_after)
                hpx::tracy::lock_acquired(context_, false);
#endif
            return false;
        }

        void unlock() noexcept(
            noexcept(util::unregister_lock(std::declval<spinlock*>())))
        {
            HPX_ITT_SYNC_RELEASING(this);

            m.unlock();

            HPX_ITT_SYNC_RELEASED(this);
#if defined(HPX_HAVE_MODULE_TRACY)
            hpx::tracy::lock_released(context_);
#endif
            util::unregister_lock(this);
        }
    };
}    // namespace hpx::util
