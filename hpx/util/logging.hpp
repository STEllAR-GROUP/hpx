//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_AGAS_LOGGING_APR_10_2008_1032AM)
#define HPX_UTIL_AGAS_LOGGING_APR_10_2008_1032AM

#include <string>
#include <hpx/hpx_fwd.hpp>

#if !defined(HPX_NO_LOGGING)

#include <boost/logging/format/named_write.hpp>
#include <boost/logging/format_fwd.hpp>

BOOST_LOG_FORMAT_MSG(optimize::cache_string_one_str<>)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT std::string levelname(int level);

    ///////////////////////////////////////////////////////////////////////////
    typedef boost::logging::named_logger<>::type logger_type;
    typedef boost::logging::level::holder filter_type;

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT BOOST_DECLARE_LOG_FILTER(agas_level, filter_type)
    HPX_EXPORT BOOST_DECLARE_LOG(agas_logger, logger_type)

    #define LAGAS_(lvl)                                                       \
        BOOST_LOG_USE_LOG_IF_LEVEL(hpx::util::agas_logger(),                  \
            hpx::util::agas_level(), lvl)                                     \
        << hpx::util::levelname(::boost::logging::level::lvl) << " "          \
    /**/

    #define LAGAS_ENABLED(lvl)                                                \
        hpx::util::agas_level()->is_enabled(::boost::logging::level::lvl)     \
    /**/

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT BOOST_DECLARE_LOG_FILTER(timing_level, filter_type)
    HPX_EXPORT BOOST_DECLARE_LOG(timing_logger, logger_type)

    #define LTIM_(lvl)                                                        \
        BOOST_LOG_USE_LOG_IF_LEVEL(hpx::util::timing_logger(),                \
            hpx::util::timing_level(), lvl)                                   \
        << hpx::util::levelname(::boost::logging::level::lvl) << " "          \
    /**/

    #define LTIM_ENABLED(lvl)                                                 \
        hpx::util::timing_level()->is_enabled(::boost::logging::level::lvl)   \
    /**/

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT BOOST_DECLARE_LOG_FILTER(hpx_level, filter_type)
    HPX_EXPORT BOOST_DECLARE_LOG(hpx_logger, logger_type)

    #define LHPX_(lvl, cat)                                                   \
        BOOST_LOG_USE_LOG_IF_LEVEL(hpx::util::hpx_logger(),                   \
            hpx::util::hpx_level(), lvl)                                      \
        << hpx::util::levelname(::boost::logging::level::lvl)                 \
        << (cat)                                                              \
    /**/

    #define LHPX_ENABLED(lvl)                                                 \
        hpx::util::hpx_level()->is_enabled(::boost::logging::level::lvl)      \
    /**/

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT BOOST_DECLARE_LOG_FILTER(app_level, filter_type)
    HPX_EXPORT BOOST_DECLARE_LOG(app_logger, logger_type)

    #define LAPP_(lvl)                                                        \
        BOOST_LOG_USE_LOG_IF_LEVEL(hpx::util::app_logger(),                   \
            hpx::util::app_level(), lvl)                                      \
        << hpx::util::levelname(::boost::logging::level::lvl) << " "          \
    /**/

    #define LAPP_ENABLED(lvl)                                                 \
        hpx::util::app_level()->is_enabled(::boost::logging::level::lvl)      \
    /**/

    ///////////////////////////////////////////////////////////////////////////
    // errors are logged in a special manner (always to cerr and additionally,
    // if enabled to 'normal' logging destination as well)
    HPX_EXPORT BOOST_DECLARE_LOG_FILTER(hpx_error_level, filter_type)
    HPX_EXPORT BOOST_DECLARE_LOG(hpx_error_logger, logger_type)

    #define LFATAL_                                                           \
        BOOST_LOG_USE_LOG_IF_LEVEL(hpx::util::hpx_error_logger(),             \
            hpx::util::hpx_error_level(), fatal)                              \
        << hpx::util::levelname(::boost::logging::level::fatal)               \
        << (" [ERR] ")                                                        \
    /**/

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // get the data to use to pre-fill the runtime_configuration instance
        // with logging specific data
        std::vector<std::string> const& get_logging_data();

        // the init_logging type will be used for initialization purposes only as
        // well
        struct init_logging
        {
            init_logging(runtime_configuration& ini, bool isconsole,
                naming::resolver_client& agas_client);
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    typedef boost::logging::named_logger<>::type logger_type;
    typedef boost::logging::level::holder filter_type;

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT BOOST_DECLARE_LOG_FILTER(agas_console_level, filter_type)
    HPX_EXPORT BOOST_DECLARE_LOG(agas_console_logger, logger_type)

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT BOOST_DECLARE_LOG_FILTER(timing_console_level, filter_type)
    HPX_EXPORT BOOST_DECLARE_LOG(timing_console_logger, logger_type)

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT BOOST_DECLARE_LOG_FILTER(hpx_console_level, filter_type)
    HPX_EXPORT BOOST_DECLARE_LOG(hpx_console_logger, logger_type)

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT BOOST_DECLARE_LOG_FILTER(app_console_level, filter_type)
    HPX_EXPORT BOOST_DECLARE_LOG(app_console_logger, logger_type)
}}

///////////////////////////////////////////////////////////////////////////////
#define LAGAS_CONSOLE_(lvl)                                                   \
    BOOST_LOG_USE_LOG(hpx::util::agas_console_logger(),                       \
        read_msg().gather().out(),                                            \
        hpx::util::agas_console_level()->is_enabled(                          \
            static_cast<boost::logging::level::type>(lvl)))                   \
/**/

#define LTIM_CONSOLE_(lvl)                                                    \
    BOOST_LOG_USE_LOG(hpx::util::timing_console_logger(),                     \
        read_msg().gather().out(),                                            \
        hpx::util::timing_console_level()->is_enabled(                        \
            static_cast<boost::logging::level::type>(lvl)))                   \
/**/

#define LHPX_CONSOLE_(lvl)                                                    \
    BOOST_LOG_USE_LOG(hpx::util::hpx_console_logger(),                        \
        read_msg().gather().out(),                                            \
        hpx::util::hpx_console_level()->is_enabled(                           \
            static_cast<boost::logging::level::type>(lvl)))                   \
/**/

#define LAPP_CONSOLE_(lvl)                                                    \
    BOOST_LOG_USE_LOG(hpx::util::app_console_logger(),                        \
        read_msg().gather().out(),                                            \
        hpx::util::app_console_level()->is_enabled(                           \
            static_cast<boost::logging::level::type>(lvl)))                   \
/**/

#else
// logging is disabled all together

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // get the data to use to pre-fill the runtime_configuration instance
    // with logging specific data
    std::vector<std::string> HPX_EXPORT get_logging_data();

    // the init_logging type will be used for initialization purposes only as
    // well
    struct init_logging
    {
        init_logging(runtime_configuration&, bool, naming::resolver_client&)
        {
        }
    };

    struct dummy_log_impl {};
    extern HPX_EXPORT dummy_log_impl dummy_log;

    template <typename T>
    inline dummy_log_impl& operator<<(dummy_log_impl& l, T) { return l; }

    #define LAGAS_(lvl)           if(true) {} else hpx::util::detail::dummy_log
    #define LTIM_(lvl)            if(true) {} else hpx::util::detail::dummy_log
    #define LHPX_(lvl, cat)       if(true) {} else hpx::util::detail::dummy_log
    #define LAPP_(lvl)            if(true) {} else hpx::util::detail::dummy_log
    #define LFATAL_               if(true) {} else hpx::util::detail::dummy_log

    #define LAGAS_CONSOLE_(lvl)   if(true) {} else hpx::util::detail::dummy_log
    #define LTIM_CONSOLE_(lvl)    if(true) {} else hpx::util::detail::dummy_log
    #define LHPX_CONSOLE_(lvl)    if(true) {} else hpx::util::detail::dummy_log
    #define LAPP_CONSOLE_(lvl)    if(true) {} else hpx::util::detail::dummy_log

    #define LAGAS_ENABLED(lvl)    (false)
    #define LTIM_ENABLED(lvl)     (false)
    #define LHPX_ENABLED(lvl)     (false)
    #define LAPP_ENABLED(lvl)     (false)
}}}

#endif

///////////////////////////////////////////////////////////////////////////////
// specific logging
#define LTM_(lvl)   LHPX_(lvl, "  [TM] ")   /* thread manager */
#define LRT_(lvl)   LHPX_(lvl, "  [RT] ")   /* runtime support */
#define LOSH_(lvl)  LHPX_(lvl, " [OSH] ")   /* one size heap */
#define LERR_(lvl)  LHPX_(lvl, " [ERR] ")   /* exceptions */
#define LPT_(lvl)   LHPX_(lvl, "  [PT] ")   /* parcel transport */
#define LLCO_(lvl)  LHPX_(lvl, " [LCO] ")   /* lcos */
#define LPCS_(lvl)  LHPX_(lvl, " [PCS] ")   /* performance counters */
#define LAS_(lvl)   LHPX_(lvl, "  [AS] ")   /* addressing service */
#define LBT_(lvl)   LHPX_(lvl, "  [BT] ")   /* bootstrap */

#endif


