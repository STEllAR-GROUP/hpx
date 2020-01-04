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

#ifndef JT28092007_destination_defaults_HPP_DEFINED
#define JT28092007_destination_defaults_HPP_DEFINED

#include <hpx/config.hpp>
#include <hpx/logging/format/destination/file.hpp>
#include <hpx/logging/manipulator.hpp>

#include <iostream>

namespace hpx { namespace util { namespace logging { namespace destination {

    /**
    @brief Writes the string to console
*/
    struct cout : manipulator
    {
        void operator()(const message& msg) override
        {
            std::cout << msg.full_string();
        }
    };

    /**
    @brief Writes the string to cerr
*/
    struct cerr : manipulator
    {
        void operator()(const message& msg) override
        {
            std::cerr << msg.full_string();
        }
    };

    /**
    @brief writes to stream.

    @note:
    The stream must outlive this object! Or, clear() the stream,
    before the stream is deleted.
*/
    struct stream : manipulator
    {
        HPX_NON_COPYABLE(stream);

        typedef std::ostream stream_type;

        explicit stream(stream_type* p)
        {
            set_stream(p);
        }

        /**
        @brief resets the stream. Further output will be written to this stream
    */
        void set_stream(stream_type* p)
        {
            ptr = p;
        }

        /**
        @brief clears the stream. Further output will be ignored
    */
        void clear()
        {
            set_stream(nullptr);
        }

        void operator()(const message& msg) override
        {
            if (ptr)
                *ptr << msg.full_string();
        }

    private:
        std::ostream* ptr;
    };

    /**
    @brief Writes the string to output debug window

    For non-Windows systems, this is the console.
*/
    struct dbg_window : manipulator
    {
        void operator()(const message& msg) override
        {
#ifdef HPX_WINDOWS
            ::OutputDebugStringA(msg.full_string().c_str());
#else
            // non windows - dump to console
            std::cout << msg.full_string();
#endif
        }
    };

}}}}    // namespace hpx::util::logging::destination

#endif
