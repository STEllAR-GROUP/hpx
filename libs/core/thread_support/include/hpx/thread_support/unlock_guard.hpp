//  Copyright (c) 2007-2008 Chirag Dekate, Hartmut Kaiser
//  Copyright (c) 2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    // This is a helper structure to make sure a lock gets unlocked and locked
    // again in a scope.
    template <typename Mutex>
    class unlock_guard
    {
    public:
        HPX_NON_COPYABLE(unlock_guard);

    public:
        using mutex_type = Mutex;

        explicit unlock_guard(Mutex& m)
          : m_(m)
        {
            m_.unlock();
        }

        ~unlock_guard()
        {
            m_.lock();
        }

    private:
        Mutex& m_;
    };
}}    // namespace hpx::util
