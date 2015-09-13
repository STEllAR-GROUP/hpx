// use_format_write.hpp

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

#ifndef JT28092007_format_write_detail_HPP_DEFINED
#error do not include this directly. Include hpx/util/logging/format.hpp instead
#endif

#ifndef JT28092007_use_format_write_HPP_DEFINED
#define JT28092007_use_format_write_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/format/optimize.hpp>
#include <hpx/util/logging/gather/ostream_like.hpp>
#include <hpx/util/logging/detail/manipulator.hpp>
#include <hpx/util/logging/detail/find_gather.hpp>
#include <hpx/util/logging/detail/find_format_writer.hpp>

namespace hpx { namespace util { namespace logging {




/**
@brief Makes it easier to use a logger with format_write class

You just define your <tt>logger<...> </tt> class like this:

@code
typedef logger_format_write<format_base,destination_base> logger_type;
@endcode

instead of

@code
typedef logger_format_write<
        format_base, destination_base
        gather::ostream_like::return_str<>,
        writer::format_write<formatter_base,destination_base> > > logger_type;
@endcode

FIXME need to have more template params

@param format_base_type @ref misc_use_defaults "(optional)"
Your formatter base class
@param destination_base @ref misc_use_defaults "(optional)"
Your destination base class
@param thread_safety @ref misc_use_defaults "(optional)"
Thread-safety. Any of the writer::threading classes.
@param gather @ref misc_use_defaults "(optional)"
The class that @ref gather "gathers" the message
*/
template<class format_base, class destination_base, class thread_safety,
class gather, class lock_resource>
struct logger_format_write
    : logger<
            typename detail::format_find_gather<gather>::type ,
            typename detail::format_find_writer<format_base, destination_base,
    lock_resource, thread_safety>::type >
{
    typedef logger<
            typename detail::format_find_gather<gather>::type ,
            typename detail::format_find_writer<format_base,
        destination_base, lock_resource, thread_safety>::type > logger_base_type;

    HPX_LOGGING_FORWARD_CONSTRUCTOR(logger_format_write, logger_base_type)
};

}}}

#endif

