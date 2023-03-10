// message.hpp

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

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/format.hpp>

#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace hpx::util::logging {

    /**
        @brief Optimizes the formatting for prepending and/or appending
        strings to the original message

        It keeps all the modified message in one string.
        Useful if some formatter needs to access the whole
        string at once.

        reserve() - the size that is reserved for prepending
        (similar to string::reserve function)

        Note : as strings are prepended, reserve() shrinks.
    */
    class message
    {
    public:
        message() = default;

        /**
            @param msg - the message that is originally cached
         */
        explicit message([[maybe_unused]] std::stringstream msg) noexcept
          : m_full_msg_computed(false)
#if defined(HPX_COMPUTE_HOST_CODE)
          , m_str(HPX_MOVE(msg))
#endif
        {
        }

        message(message&& other) noexcept
          : m_full_msg_computed(other.m_full_msg_computed)
#if defined(HPX_COMPUTE_HOST_CODE)
          , m_full_msg(HPX_MOVE(other.m_full_msg))
#endif
#if defined(HPX_COMPUTE_HOST_CODE)
          , m_str(HPX_MOVE(other.m_str))
#endif
        {
            other.m_full_msg_computed = false;
        }

        message& operator=(message&& other) noexcept
        {
            m_full_msg_computed = other.m_full_msg_computed;
            other.m_full_msg_computed = false;
#if defined(HPX_COMPUTE_HOST_CODE)
            m_full_msg = HPX_MOVE(other.m_full_msg);
#endif
#if defined(HPX_COMPUTE_HOST_CODE)
            m_str = HPX_MOVE(other.m_str);
#endif
            return *this;
        }

        template <typename T>
        message& operator<<(T&& v)
        {
            m_str << HPX_FORWARD(T, v);
            m_full_msg_computed = false;
            return *this;
        }

        template <typename... Args>
        message& format(std::string_view format_str, Args const&... args)
        {
            util::format_to(m_str, format_str, args...);
            m_full_msg_computed = false;
            return *this;
        }

        /**
            returns the full string
        */
        std::string const& full_string() const
        {
            if (!m_full_msg_computed)
            {
                m_full_msg_computed = true;
                m_full_msg = m_str.str();
            }
            return m_full_msg;
        }

        bool empty() const noexcept
        {
            return full_string().empty();
        }

        friend std::ostream& operator<<(std::ostream& os, message const& value)
        {
            return os << value.m_str.rdbuf();
        }

    private:
        // caching
        mutable bool m_full_msg_computed = false;
        mutable std::string m_full_msg;

        std::stringstream m_str;
    };
}    // namespace hpx::util::logging
