//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LOGGING
#define HPX_PARCELSET_POLICIES_LOGGING

#include <sstream>
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
//
#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread.hpp>
//
#include <hpx/config/parcelport_defines.hpp>
//
#include <boost/preprocessor.hpp>

//
// useful macros for formatting log messages
//
#define nhex(n) "0x" << std::setfill('0') << std::setw(n) << std::noshowbase << std::hex
#define hexpointer(p) nhex(16) << (uintptr_t)(p) << " "
#define hexuint64(p)  nhex(16) << (uintptr_t)(p) << " "
#define hexuint32(p)  nhex(8)  << (uint32_t)(p) << " "
#define hexlength(p)  nhex(6)  << (uintptr_t)(p) << " "
#define hexnumber(p)  nhex(4)  << p << " "
#define hexbyte(p)    nhex(2)  << static_cast<int>(p) << " "
#define decimal(n)    std::setfill('0') << std::setw(n) << std::noshowbase << std::dec
#define decnumber(p)  std::dec << p << " "
#define dec4(p)       decimal(4) << p << " "
#define ipaddress(p)  std::dec << (int)(reinterpret_cast<const uint8_t*>(&p))[0] << "." \
                               << (int)(reinterpret_cast<const uint8_t*>(&p))[1] << "." \
                               << (int)(reinterpret_cast<const uint8_t*>(&p))[2] << "." \
                               << (int)(reinterpret_cast<const uint8_t*>(&p))[3] << " "
#define sockaddress(p) ipaddress(((struct sockaddr_in*)(p))->sin_addr.s_addr)

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric {
namespace detail {

    struct rdma_thread_print_helper {};

    inline std::ostream& operator<<(std::ostream& os, const rdma_thread_print_helper&)
    {
        if (hpx::threads::get_self_id()==hpx::threads::invalid_thread_id) {
            os << "------------------ ";
        }
        else {
            hpx::threads::thread_data *dummy =
                hpx::this_thread::get_id().native_handle().get();
            os << hexpointer(dummy);
        }
        os << nhex(12) << std::this_thread::get_id();
        return os;
    }

}}}}}

#define THREAD_ID "" \
    << hpx::parcelset::policies::libfabric::detail::rdma_thread_print_helper()

//
// If any log options are ON then we must include the boost log files
//
#if defined(HPX_PARCELPORT_VERBS_HAVE_DEV_MODE)     || \
    defined(HPX_PARCELPORT_VERBS_HAVE_LOGGING)      || \
    defined(HPX_PARCELPORT_LIBFABRIC_HAVE_DEV_MODE) || \
    defined(HPX_PARCELPORT_LIBFABRIC_HAVE_LOGGING)

#  include <boost/log/trivial.hpp>
#  include <boost/log/expressions/formatter.hpp>
#  include <boost/log/expressions/formatters.hpp>
#  include <boost/log/expressions/formatters/stream.hpp>
#  include <boost/log/expressions.hpp>
#  include <boost/log/sources/severity_logger.hpp>
#  include <boost/log/sources/record_ostream.hpp>
#  include <boost/log/utility/formatting_ostream.hpp>
#  include <boost/log/utility/manipulators/to_log.hpp>
#  include <boost/log/utility/setup/console.hpp>
#  include <boost/log/utility/setup/common_attributes.hpp>
#endif

//
// Full Logging enabled
//
#if defined(HPX_PARCELPORT_LIBFABRIC_HAVE_LOGGING) || \
    defined(HPX_PARCELPORT_VERBS_HAVE_LOGGING)

#  define LOG_TRACE_MSG(x) BOOST_LOG_TRIVIAL(trace)   << THREAD_ID << " " << x;
#  define LOG_DEBUG_MSG(x) BOOST_LOG_TRIVIAL(debug)   << THREAD_ID << " " << x;
#  define LOG_INFO_MSG(x)  BOOST_LOG_TRIVIAL(info)    << THREAD_ID << " " << x;
#  define LOG_WARN_MSG(x)  BOOST_LOG_TRIVIAL(warning) << THREAD_ID << " " << x;
#  define LOG_ERROR_MSG(x) BOOST_LOG_TRIVIAL(error)   << THREAD_ID << " " << x;
#  define LOG_FATAL_MSG(x) BOOST_LOG_TRIVIAL(fatal)   << THREAD_ID << " " << x;
//
#  define LOG_EXCLUSIVE(x) x
//
#  define FUNC_START_DEBUG_MSG LOG_TRACE_MSG("*** Enter " << __func__);
#  define FUNC_END_DEBUG_MSG   LOG_TRACE_MSG("### Exit  " << __func__);

#  define LOG_FORMAT_MSG(x)                                    \
    (dynamic_cast<std::ostringstream &> (                      \
        std::ostringstream().seekp(0, std::ios_base::cur) << x \
        << __FILE__ << " " << std::dec << __LINE__ )).str()

#  define LOG_TIMED_INIT(name)                                                      \
    using namespace std::chrono;                                                    \
    static time_point<system_clock> log_timed_start_ ## name = system_clock::now(); \

#  define LOG_TIMED_MSG(name, level, delay, x)             \
    time_point<system_clock> log_timed_now_ ## name =      \
        system_clock::now();                               \
    duration<double> log_timed_elapsed_ ## name =          \
      log_timed_now_ ## name - log_timed_start_ ## name;   \
    if (log_timed_elapsed_ ## name.count()>delay) {        \
        LOG_DEVEL_MSG(x);                                  \
        log_timed_start_ ## name = log_timed_now_ ## name; \
    }

#  define LOG_TIMED_BLOCK(name, level, delay, x)           \
    time_point<system_clock> log_timed_now_ ## name =      \
        system_clock::now();                               \
    duration<double> log_timed_elapsed_ ## name =          \
      log_timed_now_ ## name - log_timed_start_ ## name;   \
    if (log_timed_elapsed_ ## name.count()>delay) {        \
        log_timed_start_ ## name = log_timed_now_ ## name; \
        x;                                                 \
    }

//
// Logging disabled, #define all macros (except for error), to be empty
//
#else
#  define LOG_DEVEL_MSG(x)
#  define LOG_DEBUG_MSG(x)
#  define LOG_TRACE_MSG(x)
#  define LOG_INFO_MSG(x)
#  define LOG_WARN_MSG(x)
#  define LOG_ERROR_MSG(x) std::cout << "00: <ERROR> " << THREAD_ID << " " \
	<< x << " " << __FILE__ << " " << std::dec << __LINE__ << std::endl;
#  define LOG_FORMAT_MSG(x) ""
#  define LOG_EXCLUSIVE(x)

#  define FUNC_START_DEBUG_MSG
#  define FUNC_END_DEBUG_MSG

#  define LOG_TIMED_INIT(name)
#  define LOG_TIMED_MSG(name, level, delay, x)
#  define LOG_TIMED_BLOCK(name, level, delay, x)
#endif
//
// Log DEVEL : just enable the LOG_DEVEL macro to bypass most log output
// but still show some that have been specially marked
//
#if defined(HPX_PARCELPORT_VERBS_HAVE_DEV_MODE) || \
    defined(HPX_PARCELPORT_LIBFABRIC_HAVE_DEV_MODE)
#  undef  LOG_TRACE_MSG
#  undef  LOG_DEVEL_MSG
#  undef  LOG_DEBUG_MSG
#  undef  LOG_EXCLUSIVE
#  define LOG_EXCLUSIVE(x) x
#  define LOG_TRACE_MSG(x) BOOST_LOG_TRIVIAL(trace) << "" << THREAD_ID << " " << x;
#  define LOG_DEVEL_MSG(x) BOOST_LOG_TRIVIAL(debug) << "" << THREAD_ID << " " << x;
#  define LOG_DEBUG_MSG(x) BOOST_LOG_TRIVIAL(debug) << "" << THREAD_ID << " " << x;
#  define LOG_BOOST_MSG(x,y) BOOST_LOG_TRIVIAL(debug) << "" << THREAD_ID << " " << boost::format(x) y;

#  define LOG_TIMED_INIT(name)                                                      \
    using namespace std::chrono;                                                    \
    static time_point<system_clock> log_timed_start_ ## name = system_clock::now(); \

#  define LOG_TIMED_MSG(name, level, delay, x)             \
    time_point<system_clock> log_timed_now_ ## name =      \
        system_clock::now();                               \
    duration<double> log_timed_elapsed_ ## name =          \
      log_timed_now_ ## name - log_timed_start_ ## name;   \
    if (log_timed_elapsed_ ## name.count()>delay) {        \
        LOG_DEVEL_MSG(x);                                  \
        log_timed_start_ ## name = log_timed_now_ ## name; \
    }

#  define LOG_TIMED_BLOCK(name, level, delay, x)           \
    time_point<system_clock> log_timed_now_ ## name =      \
        system_clock::now();                               \
    duration<double> log_timed_elapsed_ ## name =          \
      log_timed_now_ ## name - log_timed_start_ ## name;   \
    if (log_timed_elapsed_ ## name.count()>delay) {        \
        log_timed_start_ ## name = log_timed_now_ ## name; \
        x;                                                 \
    }

#endif

//
// Utility to allow automatic printing of enum names in log messages
//
// example of usage
// DEFINE_ENUM_WITH_STRING_CONVERSIONS(test_type, (test1)(test2)(test3))

#  define X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE(r, data, elem)  \
    case elem : return BOOST_PP_STRINGIZE(elem);

#  define DEFINE_ENUM_WITH_STRING_CONVERSIONS(name, enumerators)              \
    enum name {                                                               \
        BOOST_PP_SEQ_ENUM(enumerators)                                        \
    };                                                                        \
                                                                              \
    static const char* ToString(name v) {                                     \
        switch (v) {                                                          \
            BOOST_PP_SEQ_FOR_EACH(                                            \
                X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE,          \
                name,                                                         \
                enumerators                                                   \
            )                                                                 \
            default: return "[Unknown " BOOST_PP_STRINGIZE(name) "]";         \
        }                                                                     \
    }

#endif
