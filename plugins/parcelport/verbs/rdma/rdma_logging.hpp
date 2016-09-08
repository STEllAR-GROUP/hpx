//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_VERBS_RDMA_LOGGING
#define HPX_PARCELSET_POLICIES_VERBS_RDMA_LOGGING

#include <iostream>
#include <iomanip>
#include <thread>
//
#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread.hpp>
//
#include <boost/log/trivial.hpp>

//
// useful macros for formatting messages
//
#define hexpointer(p) "0x" << std::setfill('0') << std::setw(12) << std::noshowbase << std::hex << (uintptr_t)(p) << " "
#define hexuint32(p)  "0x" << std::setfill('0') << std::setw( 8) << std::noshowbase << std::hex << (uint32_t)(p) << " "
#define hexlength(p)  "0x" << std::setfill('0') << std::setw( 6) << std::noshowbase << std::hex << (uintptr_t)(p) << " "
#define hexnumber(p)  "0x" << std::setfill('0') << std::setw( 4) << std::noshowbase << std::hex << p << " "
#define decnumber(p)  "" << std::dec << p << " "
#define ipaddress(p)  "" << std::dec << (int) ((uint8_t*) &p)[0] << "." << (int) ((uint8_t*) &p)[1] << \
    "." << (int) ((uint8_t*) &p)[2] << "." << (int) ((uint8_t*) &p)[3] << " "

namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs {

    struct RdmaThreadPrintHelper {};

    inline std::ostream& operator<<(std::ostream& os, const RdmaThreadPrintHelper&)
    {
        if (hpx::threads::get_self_id()==hpx::threads::invalid_thread_id) {
            os << "-------------- ";
        }
        else {
            hpx::threads::thread_data *dummy = hpx::this_thread::get_id().native_handle().get();
            os << hexpointer(dummy);
        }
        os << "0x" << std::setfill('0') << std::setw(12) << std::noshowbase
            << std::hex << std::this_thread::get_id();
        return os;
    }

}}}}

#define THREAD_ID "" << hpx::parcelset::policies::verbs::RdmaThreadPrintHelper()

// This is a special log message that will be output even when logging is not enabled
// it should only be used in development as a way of triggering selected messages
// without enabling all of them
//#define LOG_DEVEL_MSG(x) BOOST_LOG_TRIVIAL(debug) << THREAD_ID << " " << x;
#define LOG_DEVEL_MSG(x)

//
// Logging disabled, #define all macros to be empty
//
#ifndef HPX_PARCELPORT_VERBS_HAVE_LOGGING
#  define LOG_DEBUG_MSG(x)
#  define LOG_TRACE_MSG(x)
#  define LOG_INFO_MSG(x)
#  define LOG_WARN_MSG(x)
#  define LOG_ERROR_MSG(x) std::cout << x << " " << __FILE__ << " " << __LINE__ << std::endl;
#  define LOG_SETUP_VAR(x)
//
#  define FUNC_START_DEBUG_MSG
#  define FUNC_END_DEBUG_MSG

#else
//
// Logging enabled
//
/*
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
*/
#  include <boost/preprocessor.hpp>


#  define LOG_TRACE_MSG(x) BOOST_LOG_TRIVIAL(trace)   << THREAD_ID << " " << x;
#  define LOG_DEBUG_MSG(x) BOOST_LOG_TRIVIAL(debug)   << THREAD_ID << " " << x;
#  define LOG_INFO_MSG(x)  BOOST_LOG_TRIVIAL(info)    << THREAD_ID << " " << x;
#  define LOG_WARN_MSG(x)  BOOST_LOG_TRIVIAL(warning) << THREAD_ID << " " << x;
#  define LOG_ERROR_MSG(x) BOOST_LOG_TRIVIAL(error)   << THREAD_ID << " " << x;
#  define LOG_FATAL_MSG(x) BOOST_LOG_TRIVIAL(fatal)   << THREAD_ID << " " << x;
//
#  define LOG_SETUP_VAR(x) x
//
#  define FUNC_START_DEBUG_MSG LOG_DEBUG_MSG("**************** Enter " << __func__);
#  define FUNC_END_DEBUG_MSG   LOG_DEBUG_MSG("################ Exit  " << __func__);

#  define X_DEFINE_ENUM_WITH_STRING_CONVERSIONS_TOSTRING_CASE(r, data, elem)    \
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

   DEFINE_ENUM_WITH_STRING_CONVERSIONS(BGCIOS_type, (BGV_RDMADROP)(BGV_RDMA_REG)(BGV_RDMA_RMV)(BGV_WORK_CMP)(BGV_RECV_EVT))

#endif

#endif
