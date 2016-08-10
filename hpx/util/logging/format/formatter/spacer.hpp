// spacer.hpp

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


#ifndef JT28092007_spacer_HPP_DEFINED
#define JT28092007_spacer_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/manipulator.hpp>    // is_generic
#include <hpx/util/logging/format/formatter/convert_format.hpp>

#include <type_traits>

namespace hpx { namespace util { namespace logging { namespace formatter {

namespace detail {

    template<class original_formatter, class msg_type,
    class string_type> inline void spacer_write_with_convert(msg_type & msg,
        const original_formatter & fmt, const string_type & prefix,
        const string_type & suffix, const do_convert_format::prepend*) {
        // prepend
        do_convert_format::prepend::write(suffix, msg);
        fmt(msg);
        do_convert_format::prepend::write(prefix, msg);
    }
    template<class original_formatter, class msg_type,
    class string_type> inline void spacer_write_with_convert(msg_type & msg,
        const original_formatter & fmt, const string_type & prefix,
        const string_type & suffix, const do_convert_format::append*) {
        // append
        do_convert_format::append::write(prefix, msg);
        fmt(msg);
        do_convert_format::append::write(suffix, msg);
    }
    template<class original_formatter, class msg_type, class string_type,
    class convert> inline
        void spacer_write_with_convert(msg_type & msg, const original_formatter & fmt,
            const string_type & prefix, const string_type & suffix, const convert*) {
        // custom conversion - prefix before suffix
        convert::write(prefix, msg);
        fmt(msg);
        convert::write(suffix, msg);
    }

    // note: pass original_formatter here
    // - so that original_formatter::operator() gets called,
    // not the spacer_t's operator()
    template<class original_formatter, class convert, class msg_type,
    class string_type> inline void spacer_write(msg_type & msg,
        const original_formatter & fmt, const string_type & prefix,
        const string_type & suffix) {
        spacer_write_with_convert(msg, fmt, prefix, suffix, 0);
    }
}

/** @brief Prepends some info, and appends some info to an existing formatter

The syntax is simple: construct a spacer by passing the original formatter,
and the text to space (prepend and append).
Use:
- @c % to mean the original formatter text
- anything before @c "%" is prepended before
- anything after @c "%" is appended after

Examples:

@code
// prefix "[" before index, and append "] " after it.
formatter::spacer( formatter::idx(), "[%] ");

// prefix "{T" before thread_id, and append "} " after it
formatter::spacer( formatter::thread_id(), "{T%} ");
@endcode

When adding a spacer formatter, you'll do something similar to:

@code
g_l()->writer().add_formatter( formatter::spacer( formatter::idx(), "[%] ") );
@endcode

However, to make this even simpler, I allow an ever easier syntax:

@code
// equivalent to the above
g_l()->writer().add_formatter( formatter::idx(), "[%] " );
@endcode


*/
template<class convert, class original_formatter,
    bool is_generic_formatter> struct spacer_t : original_formatter {
    // "fixed" formatter - it has a msg_type typedef
    typedef typename original_formatter::param param;
    typedef original_formatter spacer_base;
    typedef hold_string_type string_type;

    spacer_t(const original_formatter & fmt, const char_type * format_str)
        : spacer_base(fmt) {
        parse_format( format_str);
    }

    void operator()(param msg) const {
        detail::spacer_write<spacer_base,convert>(msg,
            *this, m_prefix, m_suffix);
    }
private:
    void parse_format(const string_type & format_str) {
        typedef typename string_type::size_type size_type;
        size_type msg_idx = format_str.find('%');
        if ( msg_idx != string_type::npos) {
            m_prefix = format_str.substr(0, msg_idx);
            m_suffix = format_str.substr( msg_idx + 1);
        }
        else
            // no suffix
            m_prefix = format_str;
    }

private:
    string_type m_prefix, m_suffix;
};

// specialize for generic formatters
template<class convert, class original_formatter>
struct spacer_t<convert,original_formatter,true> : original_formatter {
    // generic formatter
    typedef original_formatter spacer_base;
    typedef hold_string_type string_type;

    spacer_t(const original_formatter & fmt, const char_type * format_str)
        : spacer_base(fmt) {
        parse_format( format_str);
    }

    template<class msg_type> void operator()(msg_type & msg) const {
        detail::spacer_write<spacer_base,convert>(msg, *this, m_prefix, m_suffix);
    }
private:
    void parse_format(const string_type & format_str) {
        typedef typename string_type::size_type size_type;
        size_type msg_idx = format_str.find('%');
        if ( msg_idx != string_type::npos) {
            m_prefix = format_str.substr(0, msg_idx);
            m_suffix = format_str.substr( msg_idx + 1);
        }
        else
            // no suffix
            m_prefix = format_str;
    }

private:
    hold_string_type m_prefix, m_suffix;
};


namespace detail {
    template<class original_formatter,int> struct find_spacer_generic {
        // generic
        typedef typename original_formatter::convert_type convert;
        typedef spacer_t<convert, original_formatter, true> type;
    };
    template<class original_formatter>
    struct find_spacer_generic<original_formatter,0> {
        // not generic
        typedef do_convert_format::prepend convert ;
        typedef spacer_t<convert, original_formatter, false> type;
    };

    template<class original_formatter>
    struct find_spacer : find_spacer_generic<original_formatter,
        std::is_base_of<is_generic,original_formatter>::value> {
    };
}

/**
    @copydoc spacer_t
*/
template<class original_formatter>
typename detail::find_spacer<original_formatter>
::type spacer(const original_formatter & fmt, const char_type* format_str) {
    typedef typename detail::find_spacer<original_formatter>::type spacer_type;
    return spacer_type(fmt, format_str);
}

}}}}

#endif

