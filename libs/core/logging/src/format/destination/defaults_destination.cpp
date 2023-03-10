// destination_defaults.hpp

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

#include <hpx/logging/format/destinations.hpp>

#include <hpx/config.hpp>
#include <hpx/logging/message.hpp>

#include <iostream>
#include <memory>

#ifdef HPX_WINDOWS
#include <windows.h>
#endif

namespace hpx::util::logging::destination {

    cout::~cout() = default;

    struct cout_impl final : cout
    {
        void operator()(message const& msg) override
        {
            std::cout << msg.full_string();
        }
    };

    std::unique_ptr<cout> cout::make()
    {
        return std::make_unique<cout_impl>();
    }

    cerr::~cerr() = default;

    struct cerr_impl final : cerr
    {
        void operator()(message const& msg) override
        {
            std::cerr << msg.full_string();
        }
    };

    std::unique_ptr<cerr> cerr::make()
    {
        return std::make_unique<cerr_impl>();
    }

    stream::~stream() = default;

    struct stream_impl final : stream
    {
        explicit stream_impl(std::ostream* stream_ptr)
          : stream(stream_ptr)
        {
        }

        void operator()(message const& msg) override
        {
            if (ptr)
                *ptr << msg.full_string();
        }
    };

    std::unique_ptr<stream> stream::make(std::ostream* stream_ptr)
    {
        return std::make_unique<stream_impl>(stream_ptr);
    }

    dbg_window::~dbg_window() = default;

    struct dbg_window_impl final : dbg_window
    {
        void operator()(message const& msg) override
        {
#ifdef HPX_WINDOWS
            ::OutputDebugStringA(msg.full_string().c_str());
#else
            // non windows - dump to console
            std::cout << msg.full_string();
#endif
        }
    };

    std::unique_ptr<dbg_window> dbg_window::make()
    {
        return std::make_unique<dbg_window_impl>();
    }
}    // namespace hpx::util::logging::destination
