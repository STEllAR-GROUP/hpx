// message.cpp

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

#include <hpx/logging/message.hpp>

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace hpx { namespace util { namespace logging {

    void message::set_string(std::string const& str)
    {
        m_str.resize(str.size() + m_reserve);
        std::copy(str.begin(), str.end(),
            m_str.begin() + static_cast<std::ptrdiff_t>(m_reserve));
        m_full_msg_computed = false;
    }

    void message::prepend_string(boost::string_ref str)
    {
        std::size_t len = str.length();
        if (m_reserve < len)
        {
            std::size_t new_reserve = len + 10;
            resize_string(new_reserve);
        }

        HPX_ASSERT(m_reserve >= len);

        std::ptrdiff_t start_idx = static_cast<std::ptrdiff_t>(m_reserve - len);
        m_reserve -= len;

        std::copy(str.data(), str.data() + len, m_str.begin() + start_idx);
        m_full_msg_computed = false;
    }

    void message::resize_string(std::size_t reserve_)
    {
        if (is_string_set())
        {
            std::size_t to_add = reserve_ - m_reserve;
            std::size_t new_size = m_str.size() + to_add;

            // I'm creating a new string instead of resizing the existing one
            // this is because the new string could be of lower size
            std::string new_str(reserve_, 0);
            std::size_t used_size = m_str.size() - m_reserve;
            new_str.insert(new_str.end(),
                m_str.begin() + static_cast<std::ptrdiff_t>(m_reserve),
                m_str.begin() +
                    static_cast<std::ptrdiff_t>(m_reserve + used_size));

            HPX_ASSERT(new_size == reserve_ + used_size);

            new_str.resize(new_size, 0);
            std::swap(new_str, m_str);
        }

        m_reserve = reserve_;
    }

}}}    // namespace hpx::util::logging
