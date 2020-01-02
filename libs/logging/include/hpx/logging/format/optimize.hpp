// optimize.hpp

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

#ifndef JT28092007_optimize_HPP_DEFINED
#define JT28092007_optimize_HPP_DEFINED

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/logging/detail/fwd.hpp>

#include <cstddef>
#include <cstring>
#include <cwchar>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace util { namespace logging {

    /**
    @brief Gathering the message: contains optimizers for formatting and/or destinations:
    for example, caching techniques
*/
    namespace optimize {

        /**
        @brief Optimizes the formatting for prepending and/or appending strings to
        the original message

        It keeps all the modified message in one string.
        Useful if some formatter needs to access the whole
        string at once.

        reserve_prepend() - the size that is reserved for prepending
        (similar to string::reserve function)
        reserve_append() - the size that is reserved for appending
        (similar to string::reserve function)

        Note : as strings are prepended, reserve_prepend() shrinks.
        Same goes for append.
    */
        struct cache_string_one_str
        {
            typedef cache_string_one_str self_type;

            /**
        @param reserve_prepend - how many chars to have space to prepend by default
        @param reserve_append - how many chars to have space to append by default
        @param grow_size - in case we add a string and there's no room for it,
                           with how much should we grow? We'll
                           grow this much in addition to the added string
                           - in the needed direction
         */
            cache_string_one_str(std::size_t reserve_prepend_,
                std::size_t reserve_append_, std::size_t grow_size_ = 10)
              : m_reserve_prepend(reserve_prepend_)
              , m_reserve_append(reserve_append_)
              , m_grow_size(grow_size_)
              , m_full_msg_computed(false)
            {
            }

            /**
        @param msg - the message that is originally cached
        @param reserve_prepend - how many chars to have space to prepend by default
        @param reserve_append - how many chars to have space to append by default
        @param grow_size - in case we add a string and there's no room for it,
                           with how much should we grow? We'll
                           grow this much in addition to the added string
                           - in the needed direction
         */
            cache_string_one_str(std::string const& msg,
                std::size_t reserve_prepend_ = 10,
                std::size_t reserve_append_ = 10, std::size_t grow_size_ = 10)
              : m_reserve_prepend(reserve_prepend_)
              , m_reserve_append(reserve_append_)
              , m_grow_size(grow_size_)
              , m_full_msg_computed(false)
            {
                set_string(msg);
            }

            cache_string_one_str(cache_string_one_str&& other)
              : m_reserve_prepend(other.m_reserve_prepend)
              , m_reserve_append(other.m_reserve_prepend)
              , m_grow_size(other.m_grow_size)
              , m_str(std::move(other.m_str))
              , m_full_msg_computed(other.m_full_msg_computed)
              , m_full_msg(std::move(other.m_full_msg))
            {
                other.m_reserve_prepend = 10;
                other.m_reserve_append = 10;
                other.m_grow_size = 10;
                other.m_full_msg_computed = false;
            }

            cache_string_one_str()
              : m_reserve_prepend(10)
              , m_reserve_append(10)
              , m_grow_size(10)
              , m_full_msg_computed(false)
            {
            }

            void set_string(std::string const& str)
            {
                m_str.resize(str.size() + m_reserve_prepend + m_reserve_append);
                std::copy(str.begin(), str.end(),
                    m_str.begin() +
                        static_cast<std::ptrdiff_t>(m_reserve_prepend));
                m_full_msg_computed = false;
            }

            std::size_t reserve_prepend() const
            {
                return m_reserve_prepend;
            }
            std::size_t reserve_append() const
            {
                return m_reserve_append;
            }
            std::size_t grow_size() const
            {
                return m_grow_size;
            }

            void reserve_prepend(std::size_t new_size)
            {
                resize_string(new_size, m_reserve_append);
            }

            void reserve_append(std::size_t new_size)
            {
                resize_string(m_reserve_prepend, new_size);
            }

            void grow_size(std::size_t new_size)
            {
                m_grow_size = new_size;
            }

        private:
            static std::size_t str_len(const char* str)
            {
                return strlen(str);
            }
            static std::size_t str_len(const wchar_t* str)
            {
                return wcslen(str);
            }

        public:
            void prepend_string(const char* str)
            {
                std::size_t len = str_len(str);
                if (m_reserve_prepend < len)
                {
                    std::size_t new_reserve_prepend = len + m_grow_size;
                    resize_string(new_reserve_prepend, m_reserve_append);
                }

                HPX_ASSERT(m_reserve_prepend >= len);

                std::ptrdiff_t start_idx =
                    static_cast<std::ptrdiff_t>(m_reserve_prepend - len);
                m_reserve_prepend -= len;

                std::copy(str, str + len, m_str.begin() + start_idx);
                m_full_msg_computed = false;
            }

            /**
            @brief pre-pends a string (inserts it at the beginning)
        */
            void prepend_string(std::string const& str)
            {
                if (m_reserve_prepend < str.size())
                {
                    std::size_t new_reserve_prepend = str.size() + m_grow_size;
                    resize_string(new_reserve_prepend, m_reserve_append);
                }

                HPX_ASSERT(m_reserve_prepend >= str.size());

                std::ptrdiff_t start_idx =
                    static_cast<std::ptrdiff_t>(m_reserve_prepend - str.size());
                m_reserve_prepend -= str.size();

                std::copy(str.begin(), str.end(), m_str.begin() + start_idx);
                m_full_msg_computed = false;
            }

            /**
            writes the current cached contents to a stream
        */
            template <class stream_type>
            void to_stream(stream_type& stream) const
            {
                stream.write(m_str.begin() + m_reserve_prepend,
                    m_str.size() - m_reserve_prepend - m_reserve_append);
            }

            /**
            returns the full string
        */
            std::string const& full_string() const
            {
                if (!m_full_msg_computed)
                {
                    m_full_msg_computed = true;
                    m_full_msg = m_str.substr(m_reserve_prepend,
                        m_str.size() - m_reserve_prepend - m_reserve_append);
                }
                return m_full_msg;
            }

            operator std::string const&() const
            {
                return full_string();
            }

        private:
            void resize_string(
                std::size_t reserve_prepend_, std::size_t reserve_append_)
            {
                if (is_string_set())
                {
                    std::size_t to_add = reserve_prepend_ + reserve_append_ -
                        m_reserve_prepend - m_reserve_append;
                    std::size_t new_size = m_str.size() + to_add;

                    // I'm creating a new string instead of resizing the existing one
                    // this is because the new string could be of lower size
                    std::string new_str(reserve_prepend_, 0);
                    std::size_t used_size =
                        m_str.size() - m_reserve_prepend - m_reserve_append;
                    new_str.insert(new_str.end(),
                        m_str.begin() +
                            static_cast<std::ptrdiff_t>(m_reserve_prepend),
                        m_str.begin() +
                            static_cast<std::ptrdiff_t>(
                                m_reserve_prepend + used_size));

                    HPX_ASSERT(new_size ==
                        reserve_prepend_ + used_size + reserve_append_);

                    new_str.resize(new_size, 0);
                    std::swap(new_str, m_str);
                }

                m_reserve_prepend = reserve_prepend_;
                m_reserve_append = reserve_append_;
            }

            // if true, string was already set
            bool is_string_set() const
            {
                return !m_str.empty();
            }

        private:
            std::size_t m_reserve_prepend;
            std::size_t m_reserve_append;
            std::size_t m_grow_size;
            std::string m_str;

            // caching
            mutable bool m_full_msg_computed;
            mutable std::string m_full_msg;
        };

    }    // namespace optimize
}}}      // namespace hpx::util::logging

#endif
