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

#ifndef HPX_LOGGING_MESSAGE_HPP
#define HPX_LOGGING_MESSAGE_HPP

#include <hpx/config.hpp>

#include <boost/utility/string_ref.hpp>

#include <cstddef>
#include <string>
#include <utility>

namespace hpx { namespace util { namespace logging {

    /**
        @brief Optimizes the formatting for prepending and/or appending strings to
        the original message

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
        /**
        @param reserve - how many chars to have space to prepend by default
         */
        message(std::size_t reserve_)
          : m_reserve(reserve_)
          , m_full_msg_computed(false)
        {
        }

        /**
        @param msg - the message that is originally cached
        @param reserve - how many chars to have space to prepend by default
         */
        message(std::string const& msg, std::size_t reserve_ = 10)
          : m_reserve(reserve_)
          , m_full_msg_computed(false)
        {
            set_string(msg);
        }

        message(message&& other)
          : m_reserve(other.m_reserve)
          , m_str(std::move(other.m_str))
          , m_full_msg_computed(other.m_full_msg_computed)
          , m_full_msg(std::move(other.m_full_msg))
        {
            other.m_reserve = 10;
            other.m_full_msg_computed = false;
        }

        message()
          : m_reserve(10)
          , m_full_msg_computed(false)
        {
        }

        void HPX_EXPORT set_string(std::string const& str);

        std::size_t reserve() const
        {
            return m_reserve;
        }

        void reserve(std::size_t new_size)
        {
            resize_string(new_size);
        }

        void HPX_EXPORT prepend_string(boost::string_ref str);

        /**
            returns the full string
        */
        std::string const& full_string() const
        {
            if (!m_full_msg_computed)
            {
                m_full_msg_computed = true;
                m_full_msg = m_str.substr(m_reserve, m_str.size() - m_reserve);
            }
            return m_full_msg;
        }

    private:
        void HPX_EXPORT resize_string(std::size_t reserve_);

        // if true, string was already set
        bool is_string_set() const
        {
            return !m_str.empty();
        }

    private:
        std::size_t m_reserve;
        std::string m_str;

        // caching
        mutable bool m_full_msg_computed;
        mutable std::string m_full_msg;
    };

}}}    // namespace hpx::util::logging

#endif /*HPX_LOGGING_MESSAGE_HPP*/
