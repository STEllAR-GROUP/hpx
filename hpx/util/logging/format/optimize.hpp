// optimize.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

#ifndef JT28092007_optimize_HPP_DEFINED
#define JT28092007_optimize_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/assert.hpp>
#include <hpx/util/logging/detail/fwd.hpp>
#include <map>
#include <memory>
#include <vector>

#include <string.h>
#include <wchar.h>

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
    template<class string_type_ = hpx::util::logging::hold_string_type >
    struct cache_string_one_str {
        typedef cache_string_one_str<string_type_> self_type;
        typedef string_type_ string_type;

        /**
        @param reserve_prepend - how many chars to have space to prepend by default
        @param reserve_append - how many chars to have space to append by default
        @param grow_size - in case we add a string and there's no room for it,
                           with how much should we grow? We'll
                           grow this much in addition to the added string
                           - in the needed direction
         */
        cache_string_one_str(std::size_t reserve_prepend_, std::size_t reserve_append_,
            std::size_t grow_size_ = 10)
                : m_reserve_prepend(reserve_prepend_), m_reserve_append(reserve_append_),
            m_grow_size(grow_size_), m_full_msg_computed(false) {}

        /**
        @param msg - the message that is originally cached
        @param reserve_prepend - how many chars to have space to prepend by default
        @param reserve_append - how many chars to have space to append by default
        @param grow_size - in case we add a string and there's no room for it,
                           with how much should we grow? We'll
                           grow this much in addition to the added string
                           - in the needed direction
         */
        cache_string_one_str(const string_type & msg, std::size_t reserve_prepend_ = 10,
            std::size_t reserve_append_ = 10, std::size_t grow_size_ = 10)
                : m_reserve_prepend(reserve_prepend_), m_reserve_append(reserve_append_),
            m_grow_size(grow_size_), m_full_msg_computed(false) {
            set_string(msg);
        }

        cache_string_one_str() : m_reserve_prepend(10), m_reserve_append(10),
            m_grow_size(10), m_full_msg_computed(false) {}

        void set_string(const string_type & str) {
            m_str.resize(str.size() + m_reserve_prepend + m_reserve_append);
            std::copy( str.begin(), str.end(), m_str.begin() +
                static_cast<typename string_type::difference_type>(m_reserve_prepend));
            m_full_msg_computed = false;
        }

        std::size_t reserve_prepend() const { return m_reserve_prepend; }
        std::size_t reserve_append() const { return m_reserve_append; }
        std::size_t grow_size() const { return m_grow_size; }

        void reserve_prepend(std::size_t new_size) {
            resize_string(new_size, m_reserve_append);
        }

        void reserve_append(std::size_t new_size) {
            resize_string(m_reserve_prepend, new_size);
        }

        void grow_size(std::size_t new_size) {
            m_grow_size = new_size;
        }

    private:
        static std::size_t str_len(const char* str)
        { return strlen(str); }
        static std::size_t str_len(const wchar_t* str)
        { return wcslen(str); }
    public:

        void prepend_string(const char_type* str) {
            std::size_t len = str_len(str);
            if ( m_reserve_prepend < len) {
                std::size_t new_reserve_prepend = len + m_grow_size ;
                resize_string( new_reserve_prepend, m_reserve_append);
            }

            HPX_ASSERT(m_reserve_prepend >= len );

            typename string_type::difference_type start_idx =
                static_cast<typename string_type::difference_type>(m_reserve_prepend
                    - len);
            m_reserve_prepend -= len;

            std::copy(str, str + len, m_str.begin() + start_idx);
            m_full_msg_computed = false;
        }
        void append_string(const char_type* str) {
            std::size_t len = str_len(str);
            if ( m_reserve_append < len) {
                std::size_t new_reserve_append = len + m_grow_size ;
                resize_string( m_reserve_prepend, new_reserve_append);
            }

            HPX_ASSERT(m_reserve_append >= len );

            typename string_type::difference_type start_idx =
                static_cast<typename string_type::difference_type>(m_str.size()
                    - m_reserve_append);

            std::copy(str, str + len, m_str.begin() + start_idx);
            m_reserve_append -= len;
            m_full_msg_computed = false;
        }



        /**
            @brief pre-pends a string (inserts it at the beginning)
        */
        void prepend_string(const string_type & str) {
            if ( m_reserve_prepend < str.size()) {
                std::size_t new_reserve_prepend = str.size() + m_grow_size ;
                resize_string( new_reserve_prepend, m_reserve_append);
            }

            HPX_ASSERT(m_reserve_prepend >= str.size() );

            typename string_type::difference_type start_idx =
                static_cast<typename string_type::difference_type>(m_reserve_prepend
                    - str.size());
            m_reserve_prepend -= str.size();

            std::copy(str.begin(), str.end(), m_str.begin() + start_idx);
            m_full_msg_computed = false;
        }

        /**
            @brief appends a string (inserts it at the end)
        */
        void append_string(const string_type & str) {
            if ( m_reserve_append < str.size()) {
                std::size_t new_reserve_append = str.size() + m_grow_size ;
                resize_string( m_reserve_prepend, new_reserve_append);
            }

            HPX_ASSERT(m_reserve_append >= str.size());

            typename string_type::difference_type start_idx =
                static_cast<typename string_type::difference_type>(m_str.size()
                    - m_reserve_append);

            std::copy(str.begin(), str.end(), m_str.begin() + start_idx);
            m_reserve_append -= str.size();
            m_full_msg_computed = false;
        }

        /**
            writes the current cached contents to a stream
        */
        template<class stream_type> void to_stream(stream_type & stream) const {
            stream.write( m_str.begin() + m_reserve_prepend, m_str.size()
                - m_reserve_prepend - m_reserve_append);
        }

        /**
            returns the full string
        */
        const string_type & full_string() const {
            if ( !m_full_msg_computed) {
                m_full_msg_computed = true;
                m_full_msg = m_str.substr(m_reserve_prepend, m_str.size()
                    - m_reserve_prepend - m_reserve_append );
            }
            return m_full_msg;
        }

        operator const string_type&() const { return full_string(); }

    private:
        void resize_string(std::size_t reserve_prepend_, std::size_t reserve_append_) {
            if ( is_string_set() ) {
                std::size_t to_add = reserve_prepend_ + reserve_append_
                    - m_reserve_prepend - m_reserve_append ;
                std::size_t new_size = m_str.size() + to_add;

                // I'm creating a new string instead of resizing the existing one
                // this is because the new string could be of lower size
                string_type new_str(reserve_prepend_, 0);
                std::size_t used_size = m_str.size() - m_reserve_prepend
                    - m_reserve_append;
                new_str.insert( new_str.end(), m_str.begin() +
                        static_cast<typename string_type::difference_type>
                    (m_reserve_prepend), m_str.begin() +
                        static_cast<typename string_type::difference_type>
                    (m_reserve_prepend + used_size));

                HPX_ASSERT(new_size == reserve_prepend_ + used_size + reserve_append_);

                new_str.resize( new_size, 0);
                std::swap(new_str, m_str);
            }

            m_reserve_prepend = reserve_prepend_;
            m_reserve_append = reserve_append_;
        }

        // if true, string was already set
        bool is_string_set() const {
            return !m_str.empty();
        }
    private:
        std::size_t m_reserve_prepend;
        std::size_t m_reserve_append;
        std::size_t m_grow_size;
        string_type m_str;

        // caching
        mutable bool m_full_msg_computed;
        mutable string_type m_full_msg;
    };


    template<class stream, class string> inline stream& operator <<(stream & out,
        const cache_string_one_str<string> & val) {
        out << val.full_string();
        return out;
    }



    /**
        @brief This holds 3 strings - one for prepend, one for modification,
        and one for appending

        When you prepend or append, you can also specify an extra argument
        - an identifier.
        This identifier uniquely identifies the prepended or appended message.

        Afterwards, you can prepend/append only by specifying an identifier
        - which will identify a previously
        appended or prepended message
    */
    template<class string_type_ = hpx::util::logging::hold_string_type,
    class ptr_type = void* > struct cache_string_several_str {
    private:
        typedef string_type_ string_type;
        typedef std::shared_ptr<string_type> string_ptr;

        struct cached_msg {
            cached_msg() : prepended(true), id( ptr_type() ), is_new(true) {}
            cached_msg(const string_type & str, bool prepended_)
              : msg(new string_type(str)), prepended(prepended_), id( ptr_type() ),
                is_new(true) {}

            // when within the collection - it can never be null
            // when within the array - if null, use it from the collection
            string_ptr msg;
            // if true, prepended; if false, appended
            bool prepended;
            // who wrote the message?
            ptr_type id;
            // easily identify a message if it's new or it's been written before
            bool is_new;
        };

    public:

        /**
            constructs an object

            @param reserve_ [optional, default = 512] When creating the full msg,
            how much should we reserve?
        */
        cache_string_several_str(int reserve_ = 512)
            : m_full_msg_computed(false) {
            m_full_msg.reserve(reserve_);
        }

        /**
            constructs an object

            @param reserve_ [optional, default = 512] When creating the full msg,
            how much should we reserve?
        */
        cache_string_several_str(const string_type& msg, int reserve_ = 512)
            : m_msg(msg), m_full_msg_computed(false) {
            m_full_msg.reserve(reserve_);
        }

        /**
            sets the string with a swap (that is, you pass a non-const refererence,
            and we do a swap)
        */
        void set_string_swap(string_type & msg) {
            std::swap(msg, m_msg);
            m_full_msg_computed = false;
        }

        /**
            @brief sets the string
        */
        void set_string(const string_type & msg) {
            m_msg = msg;
            m_full_msg_computed = false;
        }

        /**
            @brief pre-pends a string (inserts it at the beginning)
        */
        void prepend_string(const string_type & str ) {
            m_cur_msg.push_back( cached_msg(str, true) );
            m_full_msg_computed = false;
        }

        /**
            @brief appends a string (inserts it at the end)
        */
         void append_string(const string_type & str ) {
            m_cur_msg.push_back( cached_msg(str, false) );
            m_full_msg_computed = false;
        }

        /**
            Specifies the id of the last message
        */
        void set_last_id(ptr_type id) {
            m_cur_msg.back().id = id;
        }

        /**
            @brief Reuses a pre-pended or appended string. The message was already cached
        */
        void reuse(ptr_type id ) {
            // make sure you first call restart() before reusing a formatter.
            // In your code - this means calling set_route(). .... .clear(),
            // and the writing to destinations
            HPX_ASSERT( m_cached.find(id) != m_cached.end() );

            m_cur_msg.push_back( m_cached[id] );
            m_cur_msg.back().is_new = false;
            m_full_msg_computed = false;
        }

        /**
            @brief computes (if necessary) and returns the full string
        */
        const string_type & full_string() const {
            if ( !m_full_msg_computed) {
                m_full_msg_computed = true;

                m_full_msg.erase();
                for ( typename array::const_iterator b = m_cur_msg.begin(),
                    e = m_cur_msg.end(); b != e; ++b)
                    if ( b->prepended)
                        m_full_msg += *(b->msg);

                m_full_msg += m_msg;

                for ( typename array::const_iterator b = m_cur_msg.begin(),
                    e = m_cur_msg.end(); b != e; ++b)
                    if ( !b->prepended)
                        m_full_msg += *(b->msg);
            }
            return m_full_msg;
        }

        /**
            @brief computes (if necessary) and returns the full string
        */
        operator const string_type&() const { return full_string(); }


        /**
            @brief This restarts writing the messages.
            Whatever is cached can be used again
        */
        void restart() {
            m_full_msg_computed = false;

            for ( typename array::const_iterator b = m_cur_msg.begin(),
                e = m_cur_msg.end(); b != e; ++b)
                if ( b->is_new)
                    m_cached[ b->id ] = *b;
            m_cur_msg.clear();
        }

    private:
        string_type m_msg;
        mutable bool m_full_msg_computed;
        mutable string_type m_full_msg;

        typedef std::map<ptr_type, cached_msg> coll;
        typedef std::vector<cached_msg> array;
        coll m_cached;
        array m_cur_msg;
    };

    template<class stream, class string, class ptr_type>
    inline stream& operator <<(stream & out,
        const cache_string_several_str<string,ptr_type> & val)
    {
        out << val.full_string();
        return out;
    }

}}}}

#endif

