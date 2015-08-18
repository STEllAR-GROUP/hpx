// high_precision_time.hpp

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


#ifndef JT28092007_high_precision_tag_HPP_DEFINED
#define JT28092007_high_precision_tag_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/microsec_time_clock.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <hpx/util/logging/detail/manipulator.hpp> // is_generic
#include <hpx/util/logging/format/formatter/tags.hpp> // uses_tag
#include <hpx/util/logging/format/formatter/high_precision_time.hpp>
// high_precision_time_t
#include <hpx/util/logging/format/formatter/convert_format.hpp>
// do_convert_format


namespace hpx { namespace util { namespace logging {

namespace tag {


/** @brief tag that holds the current time (with high precision) context information

@code
#include <hpx/util/logging/tag/high_precision_time.hpp>
@endcode

See @ref hpx::util::logging::tag "how to use tags".
*/
struct high_precision_time {
    high_precision_time() : val( ::boost::posix_time::microsec_clock::local_time() ) {}
    ::boost::posix_time::ptime val;
};

}

namespace formatter { namespace tag {

/** @brief Dumps current high_precision_time information
(corresponds to hpx::util::logging::tag::high_precision_time tag class)

@code
#include <hpx/util/logging/tag/high_precision_time.hpp>
@endcode

Similar to hpx::util::logging::formatter::high_precision_time_t class
- only that this one uses tags.

See @ref hpx::util::logging::tag "how to use tags".
*/
template<class convert = do_convert_format::prepend>
struct high_precision_time_t : is_generic, uses_tag< high_precision_time_t<convert>,
    ::hpx::util::logging::tag::high_precision_time > {
    typedef convert convert_type;
    typedef hpx::util::logging::formatter::high_precision_time_t<convert>
        high_precision_time_write_type;
    high_precision_time_write_type m_writer;

    high_precision_time_t(const hold_string_type & format) : m_writer(format) {}

    template<class msg_type, class tag_type>
    void write_tag(msg_type & str, const tag_type & tag) const {
        m_writer.write_high_precision_time(str, tag.val);
    }

    bool operator==(const high_precision_time_t & other) const {
        return m_writer == other.m_writer ;
    }

};

/** @brief high_precision_time_t with default values. See high_precision_time_t

@copydoc high_precision_time_t
*/
typedef high_precision_time_t<> high_precision_time;


}}

}}}

#endif

