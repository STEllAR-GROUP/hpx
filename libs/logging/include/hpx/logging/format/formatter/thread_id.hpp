// formatter_thread_id.hpp

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

#ifndef JT28092007_formatter_thread_id_HPP_DEFINED
#define JT28092007_formatter_thread_id_HPP_DEFINED

#include <hpx/config.hpp>
#include <hpx/format.hpp>
#include <hpx/logging/detail/manipulator.hpp>    // is_generic

#include <string>

namespace hpx { namespace util { namespace logging { namespace formatter {

    /**
@brief Writes the thread_id to the log

@param convert [optional] In case there needs to be a conversion between
std::(w)string and the string that holds your logged message. See convert_format.
*/
    struct thread_id : is_generic
    {
        void operator()(message& msg) const
        {
            std::string out = hpx::util::format("{}",
#if defined(HPX_WINDOWS)
                ::GetCurrentThreadId()
#else
                pthread_self()
#endif
            );

            msg.prepend_string(out);
        }

        bool operator==(const thread_id&) const
        {
            return true;
        }
    };

}}}}    // namespace hpx::util::logging::formatter

#endif
