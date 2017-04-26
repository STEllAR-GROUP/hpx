//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_AGAS_LOGGING_APR_10_2008_1032AM)
#define HPX_UTIL_AGAS_LOGGING_APR_10_2008_1032AM

#include <hpx/config.hpp>

#include <string>
#include <vector>

#if defined(HPX_HAVE_LOGGING)

#include <hpx/util/logging/format/named_write.hpp>
#include <hpx/util/logging/format_fwd.hpp>

#include <boost/current_function.hpp>

HPX_LOG_FORMAT_MSG(optimize::cache_string_one_str<>)

///////////////////////////////////////////////////////////////////////////////
// specific logging
#define LTM_(lvl)   LHPX_(lvl, "  [TM] ")   /* thread manager */
#define LRT_(lvl)   LHPX_(lvl, "  [RT] ")   /* runtime support */
#define LOSH_(lvl)  LHPX_(lvl, " [OSH] ")   /* one size heap */
#define LERR_(lvl)  LHPX_(lvl, " [ERR] ")   /* exceptions */
#define LLCO_(lvl)  LHPX_(lvl, " [LCO] ")   /* lcos */
#define LPCS_(lvl)  LHPX_(lvl, " [PCS] ")   /* performance counters */
#define LAS_(lvl)   LHPX_(lvl, "  [AS] ")   /* addressing service */
#define LBT_(lvl)   LHPX_(lvl, "  [BT] ")   /* bootstrap */
#define LSEC_(lvl)  LHPX_(lvl, " [SEC] ")   /* security */

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT std::string levelname(int level);

    ///////////////////////////////////////////////////////////////////////////
    typedef hpx::util::logging::named_logger<>::type logger_type;
    typedef hpx::util::logging::level::holder filter_type;

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(agas_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(agas_logger, logger_type)

    #define LAGAS_(lvl)                                                       \
        HPX_LOG_USE_LOG_IF_LEVEL(hpx::util::agas_logger(),                    \
            hpx::util::agas_level(), lvl)                                     \
        << hpx::util::levelname(::hpx::util::logging::level::lvl) << " "      \
    /**/

    #define LAGAS_ENABLED(lvl)                                                \
        hpx::util::agas_level()->is_enabled(::hpx::util::logging::level::lvl) \
    /**/

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(parcel_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(parcel_logger, logger_type)

    #define LPT_(lvl)                                                         \
        HPX_LOG_USE_LOG_IF_LEVEL(hpx::util::parcel_logger(),                  \
            hpx::util::parcel_level(), lvl)                                   \
        << hpx::util::levelname(::hpx::util::logging::level::lvl) << " "      \
    /**/

    #define LPT_ENABLED(lvl)                                                  \
        hpx::util::parcel_level()->is_enabled(::hpx::util::logging::level::lvl)\
    /**/

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(timing_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(timing_logger, logger_type)

    #define LTIM_(lvl)                                                        \
        HPX_LOG_USE_LOG_IF_LEVEL(hpx::util::timing_logger(),                  \
            hpx::util::timing_level(), lvl)                                   \
        << hpx::util::levelname(::hpx::util::logging::level::lvl) << " "      \
    /**/
    #define LPROGRESS_                                                        \
        HPX_LOG_USE_LOG_IF_LEVEL(hpx::util::timing_logger(),                  \
            hpx::util::timing_level(), fatal) << " "                          \
        << __FILE__ << ":" << __LINE__ << " " << BOOST_CURRENT_FUNCTION << " "\
    /**/

    #define LTIM_ENABLED(lvl)                                                 \
        hpx::util::timing_level()->is_enabled(::hpx::util::logging::level::lvl)\
    /**/

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(hpx_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(hpx_logger, logger_type)

    #define LHPX_(lvl, cat)                                                   \
        HPX_LOG_USE_LOG_IF_LEVEL(hpx::util::hpx_logger(),                     \
            hpx::util::hpx_level(), lvl)                                      \
        << hpx::util::levelname(::hpx::util::logging::level::lvl)             \
        << (cat)                                                              \
    /**/

    #define LHPX_ENABLED(lvl)                                                 \
        hpx::util::hpx_level()->is_enabled(::hpx::util::logging::level::lvl)  \
    /**/

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(app_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(app_logger, logger_type)

    #define LAPP_(lvl)                                                        \
        HPX_LOG_USE_LOG_IF_LEVEL(hpx::util::app_logger(),                     \
            hpx::util::app_level(), lvl)                                      \
        << hpx::util::levelname(::hpx::util::logging::level::lvl) << " "      \
    /**/

    #define LAPP_ENABLED(lvl)                                                 \
        hpx::util::app_level()->is_enabled(::hpx::util::logging::level::lvl)  \
    /**/

    ///////////////////////////////////////////////////////////////////////////
    // special debug logging channel
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(debuglog_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(debuglog_logger, logger_type)

    #define LDEB_                                                             \
        HPX_LOG_USE_LOG_IF_LEVEL(hpx::util::debuglog_logger(),                \
            hpx::util::debuglog_level(), error)                               \
        << hpx::util::levelname(::hpx::util::logging::level::error) << " "    \
    /**/

    #define LDEB_ENABLED                                                      \
        hpx::util::debuglog_level()->is_enabled(                              \
            ::hpx::util::logging::level::error)                               \
    /**/

    ///////////////////////////////////////////////////////////////////////////
    // errors are logged in a special manner (always to cerr and additionally,
    // if enabled to 'normal' logging destination as well)
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(hpx_error_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(hpx_error_logger, logger_type)

    #define LFATAL_                                                           \
        HPX_LOG_USE_LOG_IF_LEVEL(hpx::util::hpx_error_logger(),               \
            hpx::util::hpx_error_level(), fatal)                              \
        << hpx::util::levelname(::hpx::util::logging::level::fatal)           \
        << (" [ERR] ")                                                        \
    /**/

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // get the data to use to pre-fill the runtime_configuration instance
        // with logging specific data
        std::vector<std::string> const& get_logging_data();
    }

    ///////////////////////////////////////////////////////////////////////////
    typedef hpx::util::logging::named_logger<>::type logger_type;
    typedef hpx::util::logging::level::holder filter_type;

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(agas_console_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(agas_console_logger, logger_type)

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(parcel_console_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(parcel_console_logger, logger_type)

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(timing_console_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(timing_console_logger, logger_type)

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(hpx_console_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(hpx_console_logger, logger_type)

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(app_console_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(app_console_logger, logger_type)

    ///////////////////////////////////////////////////////////////////////////
    // special debug logging channel
    HPX_EXPORT HPX_DECLARE_LOG_FILTER(debuglog_console_level, filter_type)
    HPX_EXPORT HPX_DECLARE_LOG(debuglog_console_logger, logger_type)
}}

///////////////////////////////////////////////////////////////////////////////
#define LAGAS_CONSOLE_(lvl)                                                   \
    HPX_LOG_USE_LOG(hpx::util::agas_console_logger(),                         \
        read_msg().gather().out(),                                            \
        hpx::util::agas_console_level()->is_enabled(                          \
            static_cast<hpx::util::logging::level::type>(lvl)))               \
/**/

#define LPT_CONSOLE_(lvl)                                                     \
    HPX_LOG_USE_LOG(hpx::util::parcel_console_logger(),                       \
        read_msg().gather().out(),                                            \
        hpx::util::parcel_console_level()->is_enabled(                        \
            static_cast<hpx::util::logging::level::type>(lvl)))               \
/**/

#define LTIM_CONSOLE_(lvl)                                                    \
    HPX_LOG_USE_LOG(hpx::util::timing_console_logger(),                       \
        read_msg().gather().out(),                                            \
        hpx::util::timing_console_level()->is_enabled(                        \
            static_cast<hpx::util::logging::level::type>(lvl)))               \
/**/

#define LHPX_CONSOLE_(lvl)                                                    \
    HPX_LOG_USE_LOG(hpx::util::hpx_console_logger(),                          \
        read_msg().gather().out(),                                            \
        hpx::util::hpx_console_level()->is_enabled(                           \
            static_cast<hpx::util::logging::level::type>(lvl)))               \
/**/

#define LAPP_CONSOLE_(lvl)                                                    \
    HPX_LOG_USE_LOG(hpx::util::app_console_logger(),                          \
        read_msg().gather().out(),                                            \
        hpx::util::app_console_level()->is_enabled(                           \
            static_cast<hpx::util::logging::level::type>(lvl)))               \
/**/

#define LDEB_CONSOLE_                                                         \
    HPX_LOG_USE_LOG(hpx::util::debuglog_console_logger(),                     \
        read_msg().gather().out(),                                            \
        hpx::util::debuglog_console_level()->is_enabled(                      \
            hpx::util::logging::level::error))                                \
/**/

// helper type to forward logging during bootstrap to two destinations
struct bootstrap_logging { constexpr bootstrap_logging() {} };

template <typename T>
bootstrap_logging const& operator<< (bootstrap_logging const& l, T const& t)
{
    LBT_(info) << t;
    LPROGRESS_ << t;
    return l;
}

constexpr bootstrap_logging lbt_;

#else
// logging is disabled all together

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // get the data to use to pre-fill the runtime_configuration instance
    // with logging specific data
    HPX_EXPORT std::vector<std::string> get_logging_data();

    struct dummy_log_impl { constexpr dummy_log_impl() {} };
    constexpr dummy_log_impl dummy_log;

    template <typename T>
    dummy_log_impl const& operator<<(dummy_log_impl const& l, T&&)
    {
        return l;
    }

    struct bootstrap_logging { constexpr bootstrap_logging() {} };
    constexpr bootstrap_logging lbt_;

    template <typename T>
    bootstrap_logging const& operator<< (bootstrap_logging const& l, T&&)
    {
        return l;
    }

    #define LAGAS_(lvl)           if(true) {} else hpx::util::detail::dummy_log
    #define LPT_(lvl)             if(true) {} else hpx::util::detail::dummy_log
    #define LTIM_(lvl)            if(true) {} else hpx::util::detail::dummy_log
    #define LPROGRESS_            if(true) {} else hpx::util::detail::dummy_log
    #define LHPX_(lvl, cat)       if(true) {} else hpx::util::detail::dummy_log
    #define LAPP_(lvl)            if(true) {} else hpx::util::detail::dummy_log
    #define LDEB_                 if(true) {} else hpx::util::detail::dummy_log

    #define LFATAL_               if(true) {} else hpx::util::detail::dummy_log

    #define LAGAS_CONSOLE_(lvl)   if(true) {} else hpx::util::detail::dummy_log
    #define LPT_CONSOLE_(lvl)     if(true) {} else hpx::util::detail::dummy_log
    #define LTIM_CONSOLE_(lvl)    if(true) {} else hpx::util::detail::dummy_log
    #define LHPX_CONSOLE_(lvl)    if(true) {} else hpx::util::detail::dummy_log
    #define LAPP_CONSOLE_(lvl)    if(true) {} else hpx::util::detail::dummy_log
    #define LDEB_CONSOLE_         if(true) {} else hpx::util::detail::dummy_log

    #define LAGAS_ENABLED(lvl)    (false)
    #define LPT_ENABLED(lvl)      (false)
    #define LTIM_ENABLED(lvl)     (false)
    #define LHPX_ENABLED(lvl)     (false)
    #define LAPP_ENABLED(lvl)     (false)
    #define LDEB_ENABLED          (false)
}}}

#endif
#endif


