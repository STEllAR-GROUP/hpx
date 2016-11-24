//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_RDMA_LOGGING
#define HPX_PARCELSET_POLICIES_VERBS_RDMA_LOGGING

#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
//
#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread.hpp>
//
#include <hpx/config/parcelport_verbs_defines.hpp>
//
#include <boost/log/trivial.hpp>
//
#define HPX_PARCELPORT_VERBS_ENABLE_DEVEL_MSG

//
// useful macros for formatting log messages
//
#define nhex(n) "0x" << std::setfill('0') << std::setw(n) << std::noshowbase << std::hex
#define hexpointer(p) nhex(12) << (uintptr_t)(p) << " "
#define hexuint32(p)  nhex(8)  << (uint32_t)(p) << " "
#define hexlength(p)  nhex(6)  << (uintptr_t)(p) << " "
#define hexnumber(p)  nhex(4)  << p << " "
#define decnumber(p)  "" << std::dec << p << " "
#define ipaddress(p)  "" << std::dec << (int) ((uint8_t*) &p)[0] << "." \
                                     << (int) ((uint8_t*) &p)[1] << "." \
                                     << (int) ((uint8_t*) &p)[2] << "." \
                                     << (int) ((uint8_t*) &p)[3] << " "
#define sockaddress(p) ipaddress(((struct sockaddr_in*)(p))->sin_addr.s_addr)

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs {
namespace detail {

    struct rdma_thread_print_helper {};

    inline std::ostream& operator<<(std::ostream& os, const rdma_thread_print_helper&)
    {
        if (hpx::threads::get_self_id()==hpx::threads::invalid_thread_id) {
            os << "-------------- ";
        }
        else {
            hpx::threads::thread_data *dummy = hpx::this_thread::get_id().native_handle().get();
            os << hexpointer(dummy);
        }
        os << nhex(12) << std::this_thread::get_id();
        return os;
    }

}}}}}

#define THREAD_ID "" << hpx::parcelset::policies::verbs::detail::rdma_thread_print_helper()

// This is a special log message that will be output even when logging is not enabled
// it should only be used in development as a way of triggering selected messages
// without enabling all of them
#ifdef HPX_PARCELPORT_VERBS_ENABLE_DEVEL_MSG
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
#  define LOG_DEVEL_MSG(x) BOOST_LOG_TRIVIAL(debug) << "" << THREAD_ID << " " << x;
#else
#  define LOG_DEVEL_MSG(x)
#endif

//
// Logging disabled, #define all macros to be empty
//
#ifndef HPX_PARCELPORT_VERBS_HAVE_LOGGING
#  define LOG_DEBUG_MSG(x)
#  define LOG_TRACE_MSG(x)
#  define LOG_INFO_MSG(x)
#  define LOG_WARN_MSG(x)
#  define LOG_ERROR_MSG(x) std::cout << "ERROR: " << x << " " \
    << __FILE__ << " " << __LINE__ << std::endl;
#  define LOG_EXCLUSIVE(x)
//
#  define FUNC_START_DEBUG_MSG
#  define FUNC_END_DEBUG_MSG

#  define LOG_TIMED_INIT(name)                                                      \
    using namespace std::chrono;                                                    \
    static time_point<system_clock> log_timed_start_ ## name = system_clock::now(); \

#  define LOG_TIMED_MSG(name, level, delay, x)
#  define LOG_TIMED_BLOCK(name, level, delay, x)

#else
//
// Logging enabled
//

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

#  include <boost/preprocessor.hpp>


#  define LOG_TRACE_MSG(x) BOOST_LOG_TRIVIAL(trace)   << THREAD_ID << " " << x;
#  define LOG_DEBUG_MSG(x) BOOST_LOG_TRIVIAL(debug)   << THREAD_ID << " " << x;
#  define LOG_INFO_MSG(x)  BOOST_LOG_TRIVIAL(info)    << THREAD_ID << " " << x;
#  define LOG_WARN_MSG(x)  BOOST_LOG_TRIVIAL(warning) << THREAD_ID << " " << x;
#  define LOG_ERROR_MSG(x) BOOST_LOG_TRIVIAL(error)   << THREAD_ID << " " << x;
#  define LOG_FATAL_MSG(x) BOOST_LOG_TRIVIAL(fatal)   << THREAD_ID << " " << x;
//
#  define LOG_EXCLUSIVE(x) x
//
#  define FUNC_START_DEBUG_MSG LOG_DEBUG_MSG("*** Enter " << __func__);
#  define FUNC_END_DEBUG_MSG   LOG_DEBUG_MSG("### Exit  " << __func__);

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

#  define X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE(r, data, elem)  \
    case elem : return BOOST_PP_STRINGIZE(elem);

#  define DEFINE_ENUM_WITH_STRING_CONVERSIONS(name, enumerators)              \
    enum name {                                                               \
        BOOST_PP_SEQ_ENUM(enumerators)                                        \
    };                                                                        \
                                                                              \
    inline const char* ToString(name v)                                       \
    {                                                                         \
        switch (v)                                                            \
        {                                                                     \
            BOOST_PP_SEQ_FOR_EACH(                                            \
                X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE,          \
                name,                                                         \
                enumerators                                                   \
            )                                                                 \
            default: return "[Unknown " BOOST_PP_STRINGIZE(name) "]";         \
        }                                                                     \
    }

    // example of usage
    // DEFINE_ENUM_WITH_STRING_CONVERSIONS(test_type, (test1)(test2)(test3))

#endif

#endif
