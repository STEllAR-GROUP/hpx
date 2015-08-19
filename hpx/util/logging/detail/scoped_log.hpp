// scoped_log.hpp

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


#ifndef JT28092007_scoped_log_HPP_DEFINED
#define JT28092007_scoped_log_HPP_DEFINED

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <algorithm>
#include <iosfwd>

namespace hpx { namespace util { namespace logging {


#ifndef HPX_LOG_USE_WCHAR_T

#define BOOST_SCOPED_LOG_WITH_CLASS_NAME(logger, msg, class_name) \
struct class_name { \
    class_name()  { logger ( "start of " msg ) ;} \
    ~class_name() { logger ( "  end of " msg ) ; } \
} HPX_LOG_CONCATENATE(log_, __LINE__);

#define BOOST_SCOPED_LOG(logger, msg) \
 BOOST_SCOPED_LOG_WITH_CLASS_NAME(logger, msg, \
 HPX_LOG_CONCATENATE(boost_scoped_log,__LINE__) )

#else
// unicode
#define BOOST_SCOPED_LOG_WITH_CLASS_NAME(logger, msg, class_name) \
struct class_name { \
    class_name()  { logger ( L"start of " msg ) ;} \
    ~class_name() { logger ( L"  end of " msg ) ; } \
} HPX_LOG_CONCATENATE(log_, __LINE__);

#define BOOST_SCOPED_LOG(logger, msg) \
 BOOST_SCOPED_LOG_WITH_CLASS_NAME(logger, msg, \
 HPX_LOG_CONCATENATE(boost_scoped_log,__LINE__) )

#endif

// default scoped write - in case your gather
//    class .read_msg().out() returns an STL ostream
template<class char_type, class char_traits>
inline void scoped_write_msg(const hold_string_type & str,
    std::basic_ostream<char_type, char_traits> & out) {
    out << str;
}

namespace detail {

    template<class gather_msg = default_> struct scoped_gather_base {
        typedef typename detail::find_gather_if_default<gather_msg>
            ::msg_type msg_type;
        virtual void do_gather(const msg_type & ) = 0;
    };

    /**
        when doing scoped logging, we use this as a trick to find out if
        a logger is enabled.
        That is, we want to do the overhead of gathering the
        message to happen only if logging is enabled
    */
    template<class ostream_type = std::basic_ostringstream<char_type> ,
    class gather_msg = default_ > struct scoped_logger { //-V690

        typedef scoped_gather_base<gather_msg> scoped_gather;
        scoped_logger(scoped_gather & do_gather) : m_gather(do_gather) {}
        scoped_logger(const scoped_logger & other) : m_out( other.m_out.str() ),
            m_gather( other.m_gather) {}

        template<class type> scoped_logger & operator<<(const type& val) {
            m_out << val;
            return *this;
        }

        // when we enter here, we know the logger is enabled
        hold_string_type gathered_info() {
            hold_string_type str = m_out.str();
            m_gather.do_gather(str);
            return HPX_LOG_STR("start of ") + str;
        }

    private:
        ostream_type m_out;
        scoped_gather & m_gather;
    };

    template<class gather_type, class ostream_type> inline gather_type & operator,
        (gather_type & g, scoped_logger<ostream_type> & val) {
        scoped_write_msg( val.gathered_info(), g);
        return g;
    }
}



#define BOOST_SCOPED_LOG_CTX_IMPL(logger_macro, operator_, class_name) \
struct class_name : ::hpx::util::logging::detail::scoped_gather_base<> { \
    class_name() : m_is_enabled(false) { } \
    ~class_name() {  if ( m_is_enabled) \
    logger_macro operator_ HPX_LOG_STR("  end of ") operator_ m_str ; } \
    void do_gather(const msg_type & str) { m_str = str; m_is_enabled = true; } \
    msg_type m_str; \
    bool m_is_enabled; \
} HPX_LOG_CONCATENATE(log_, __LINE__); logger_macro , \
 ::hpx::util::logging::detail::scoped_logger<>( HPX_LOG_CONCATENATE(log_, __LINE__) )



// note: to use BOOST_SCOPED_LOG_CTX, you need to #include
// <hpx/util/logging/gather/ostream_like.hpp>
//       This is included by default, in #include <hpx/util/logging/format_fwd.hpp>
#define BOOST_SCOPED_LOG_CTX(logger) \
BOOST_SCOPED_LOG_CTX_IMPL(logger, << , HPX_LOG_CONCATENATE(boost_scoped_log,__LINE__) )


}}}

#endif

