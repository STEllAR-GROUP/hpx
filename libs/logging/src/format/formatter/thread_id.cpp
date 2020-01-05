// thread_id.cpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

#include <hpx/logging/format/formatters.hpp>

#include <hpx/config.hpp>
#include <hpx/logging/message.hpp>

#include <memory>
#include <string>

#if defined(HPX_WINDOWS)
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace hpx { namespace util { namespace logging { namespace formatter {

    thread_id::~thread_id() = default;

    struct thread_id_impl : thread_id
    {
        void operator()(message& msg) override
        {
            msg.format("{}",
#if defined(HPX_WINDOWS)
                ::GetCurrentThreadId()
#else
                pthread_self()
#endif
            );
        }
    };

    std::shared_ptr<thread_id> thread_id::make()
    {
        return std::make_shared<thread_id_impl>();
    }

}}}}    // namespace hpx::util::logging::formatter
