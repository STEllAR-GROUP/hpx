// cache_before_init.hpp

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

#ifndef JT28092007_cache_before_init_HPP_DEFINED
#define JT28092007_cache_before_init_HPP_DEFINED

#include <hpx/config.hpp>
#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/format/optimize.hpp>
#include <hpx/util/logging/writer/named_write.hpp>

#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace util { namespace logging { namespace detail {

//////////////////////////////////////////////////////////////////
// Messages that were logged before initializing the log - Caching them

/**
    The library will make sure your logger derives from this in case you want to
    cache messages that are logged before logs are initialized.

    Note:
    - you should initialize your logs ASAP
    - before logs are initialized
    - cache can be turned off ONLY ONCE
*/
struct HPX_EXPORT cache_before_init {
    HPX_NON_COPYABLE(cache_before_init);

private:
    typedef std::vector<msg_type> message_array;

public:
    cache_before_init() : m_is_caching_off(false) {}

    bool is_cache_turned_off() const noexcept {
        return m_is_caching_off; // cache has been turned off
    }

    void turn_cache_off(const writer::named_write & writer_) {
        if ( m_is_caching_off )
            return; // already turned off

        m_is_caching_off = true;

        // dump messages
        message_array msgs;
        std::swap( m_cache, msgs );

        for ( auto& msg : msgs ) {
            writer_( msg );
        }
    }

    void add_msg(msg_type msg) const {
        m_cache.push_back( std::move( msg ) );
    }

private:
    mutable message_array m_cache;
    mutable bool m_is_caching_off;
};

}}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
