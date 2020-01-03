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
#include <hpx/logging/detail/manipulator.hpp>
#include <hpx/logging/format/destination/file.hpp>
#include <iostream>

namespace hpx { namespace util { namespace logging { namespace destination {

    /**
    @brief Writes the string to console
*/
    struct cout : is_generic
    {
        void operator()(const message& msg) const
        {
            std::cout << msg.full_string();
        }

        bool operator==(const cout&) const
        {
            return true;
        }
    };

    /**
    @brief Writes the string to cerr
*/
    struct cerr : is_generic
    {
        void operator()(const message& msg) const
        {
            std::cerr << msg.full_string();
        }

        bool operator==(const cerr&) const
        {
            return true;
        }
    };

    /**
    @brief writes to stream.

    @note:
    The stream must outlive this object! Or, clear() the stream,
    before the stream is deleted.
*/
    struct stream
      : is_generic
      , non_const_context<std::ostream*>
    {
        typedef std::ostream stream_type;
        typedef non_const_context<stream_type*> non_const_context_base;

        stream(stream_type* s)
          : non_const_context_base(s)
        {
        }
        stream(stream_type& s)
          : non_const_context_base(&s)
        {
        }

        void operator()(const message& msg) const
        {
            if (non_const_context_base::context())
                *non_const_context_base::context() << msg.full_string();
        }

        bool operator==(const stream& other) const
        {
            return non_const_context_base::context() !=
                other.non_const_context_base::context();
        }

        /**
        @brief resets the stream. Further output will be written to this stream
    */
        void set_stream(stream_type* p)
        {
            non_const_context_base::context() = p;
        }

        /**
        @brief clears the stream. Further output will be ignored
    */
        void clear()
        {
            set_stream(nullptr);
        }
    };

    /**
    @brief Writes the string to output debug window

    For non-Windows systems, this is the console.
*/
    struct dbg_window : is_generic
    {
        void operator()(const message& msg) const
        {
#ifdef HPX_WINDOWS
            ::OutputDebugStringA(msg.full_string().c_str());
#else
            // non windows - dump to console
            std::cout << msg.full_string();
#endif
        }

        bool operator==(const dbg_window&) const
        {
            return true;
        }
    };

}}}}    // namespace hpx::util::logging::destination

#endif
