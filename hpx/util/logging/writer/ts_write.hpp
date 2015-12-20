// ts_write.hpp

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


#ifndef JT28092007_ts_write_HPP_DEFINED
#define JT28092007_ts_write_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/forward_constructor.hpp>
#include <hpx/util/logging/detail/manipulator.hpp>

#if !defined(HPX_HAVE_LOG_NO_TS)

namespace hpx { namespace util { namespace logging { namespace writer {

    namespace detail {
        struct ts_write_context {
            mutable hpx::util::logging::threading::mutex cs;
        };
    }

/**
<tt>\#include <hpx/util/logging/format.hpp> </tt>

Performs all writes in a thread-safe manner.
In other words, makes sure that all operator() calls of base_type are called
in a thread-safe manner.

To transform a writer into thread-safe writer, simply surround the writer with ts_write:

Example:

@code
// not thread-safe
logger< gather::ostream_like::return_str<>, write_to_cout> g_l();

// thread-safe
logger< gather::ostream_like::return_str<>, ts_write<write_to_cout> > g_l();


// not thread-safe
logger<
    gather::ostream_like::return_cache_str<> ,
    format_write< format_base, destination_base> > g_l();

// thread-safe
logger<
    gather::ostream_like::return_cache_str<> ,
    ts_write< format_write< format_base, destination_base > > > g_l();
@endcode

Depending on your scenario, you could prefer on_dedicated_thread class.

@sa on_dedicated_thread
*/
    template<class base_type> struct ts_write : base_type,
        ::hpx::util::logging::manipulator::non_const_context<detail::ts_write_context> {
        typedef ::hpx::util::logging::manipulator::non_const_context<detail
            ::ts_write_context> non_const_context_base;

        HPX_LOGGING_FORWARD_CONSTRUCTOR(ts_write,base_type)

        template<class msg_type> void operator()(msg_type msg) const {
            typedef hpx::util::logging::threading::mutex::scoped_lock lock;
            lock lk(non_const_context_base::context().cs);

            base_type::operator()(msg);
        }
    };

}}}}

#endif

#endif

