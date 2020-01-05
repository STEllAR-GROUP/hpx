//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LOGGING_HPP
#define HPX_LOGGING_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_LOGGING)

#include <hpx/assertion/current_function.hpp>
#include <hpx/format.hpp>
#include <hpx/logging/level.hpp>
#include <hpx/logging/logging.hpp>

#include <string>

////////////////////////////////////////////////////////////////////////////////
// specific logging
#define LTM_(lvl) LHPX_(lvl, "  [TM] ")  /* thread manager */
#define LRT_(lvl) LHPX_(lvl, "  [RT] ")  /* runtime support */
#define LOSH_(lvl) LHPX_(lvl, " [OSH] ") /* one size heap */
#define LERR_(lvl) LHPX_(lvl, " [ERR] ") /* exceptions */
#define LLCO_(lvl) LHPX_(lvl, " [LCO] ") /* lcos */
#define LPCS_(lvl) LHPX_(lvl, " [PCS] ") /* performance counters */
#define LAS_(lvl) LHPX_(lvl, "  [AS] ")  /* addressing service */
#define LBT_(lvl) LHPX_(lvl, "  [BT] ")  /* bootstrap */

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {
    ////////////////////////////////////////////////////////////////////////////
    namespace detail {
        HPX_API_EXPORT hpx::util::logging::level get_log_level(
            std::string const& env, bool allow_always = false);
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT HPX_DECLARE_LOG(agas)

#define LAGAS_(lvl)                                                            \
    HPX_LOG_FORMAT(hpx::util::agas, ::hpx::util::logging::level::lvl, "{} ",   \
        ::hpx::util::logging::level::lvl) /**/

#define LAGAS_ENABLED(lvl)                                                     \
    hpx::util::agas_logger()->is_enabled(::hpx::util::logging::level::lvl) /**/

        ////////////////////////////////////////////////////////////////////////
        HPX_EXPORT HPX_DECLARE_LOG(parcel)

#define LPT_(lvl)                                                              \
    HPX_LOG_FORMAT(hpx::util::parcel, ::hpx::util::logging::level::lvl, "{} ", \
        ::hpx::util::logging::level::lvl) /**/

#define LPT_ENABLED(lvl)                                                       \
    hpx::util::parcel_logger()->is_enabled(                                    \
        ::hpx::util::logging::level::lvl) /**/

        ////////////////////////////////////////////////////////////////////////
        HPX_EXPORT HPX_DECLARE_LOG(timing)

#define LTIM_(lvl)                                                             \
    HPX_LOG_FORMAT(hpx::util::timing, ::hpx::util::logging::level::lvl, "{} ", \
        ::hpx::util::logging::level::lvl) /**/
#define LPROGRESS_                                                             \
    HPX_LOG_FORMAT(hpx::util::timing, ::hpx::util::logging::level::fatal,      \
        " {}:{} {} ", __FILE__, __LINE__, HPX_ASSERT_CURRENT_FUNCTION) /**/

#define LTIM_ENABLED(lvl)                                                      \
    hpx::util::timing_logger()->is_enabled(                                    \
        ::hpx::util::logging::level::lvl) /**/

        ////////////////////////////////////////////////////////////////////////
        HPX_EXPORT HPX_DECLARE_LOG(hpx)

#define LHPX_(lvl, cat)                                                        \
    HPX_LOG_FORMAT(hpx::util::hpx, ::hpx::util::logging::level::lvl, "{}{}",   \
        ::hpx::util::logging::level::lvl, (cat)) /**/

#define LHPX_ENABLED(lvl)                                                      \
    hpx::util::hpx_logger()->is_enabled(::hpx::util::logging::level::lvl) /**/

        ////////////////////////////////////////////////////////////////////////
        HPX_EXPORT HPX_DECLARE_LOG(app)

#define LAPP_(lvl)                                                             \
    HPX_LOG_FORMAT(hpx::util::app, ::hpx::util::logging::level::lvl, "{} ",    \
        ::hpx::util::logging::level::lvl) /**/

#define LAPP_ENABLED(lvl)                                                      \
    hpx::util::app_logger()->is_enabled(::hpx::util::logging::level::lvl) /**/

        ////////////////////////////////////////////////////////////////////////
        // special debug logging channel
        HPX_EXPORT HPX_DECLARE_LOG(debuglog)

#define LDEB_                                                                  \
    HPX_LOG_FORMAT(hpx::util::debuglog, ::hpx::util::logging::level::error,    \
        "{} ", ::hpx::util::logging::level::error) /**/

#define LDEB_ENABLED                                                           \
    hpx::util::debuglog_logger()->is_enabled(                                  \
        ::hpx::util::logging::level::error) /**/

        ////////////////////////////////////////////////////////////////////////
        // errors are logged in a special manner (always to cerr and additionally,
        // if enabled to 'normal' logging destination as well)
        HPX_EXPORT HPX_DECLARE_LOG(hpx_error)

#define LFATAL_                                                                \
    HPX_LOG_FORMAT(hpx::util::hpx_error, ::hpx::util::logging::level::fatal,   \
        "{} [ERR] ", ::hpx::util::logging::level::fatal) /**/

            HPX_EXPORT HPX_DECLARE_LOG(agas_console) HPX_EXPORT
        HPX_DECLARE_LOG(parcel_console) HPX_EXPORT
        HPX_DECLARE_LOG(timing_console) HPX_EXPORT
        HPX_DECLARE_LOG(hpx_console) HPX_EXPORT HPX_DECLARE_LOG(app_console)

        // special debug logging channel
        HPX_EXPORT HPX_DECLARE_LOG(debuglog_console)
}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
#define LAGAS_CONSOLE_(lvl)                                                    \
    HPX_LOG_USE_LOG(hpx::util::agas_console,                                   \
        static_cast<::hpx::util::logging::level>(lvl))                         \
    /**/

#define LPT_CONSOLE_(lvl)                                                      \
    HPX_LOG_USE_LOG(hpx::util::parcel_console,                                 \
        static_cast<::hpx::util::logging::level>(lvl))                         \
    /**/

#define LTIM_CONSOLE_(lvl)                                                     \
    HPX_LOG_USE_LOG(hpx::util::timing_console,                                 \
        static_cast<::hpx::util::logging::level>(lvl))                         \
    /**/

#define LHPX_CONSOLE_(lvl)                                                     \
    HPX_LOG_USE_LOG(                                                           \
        hpx::util::hpx_console, static_cast<::hpx::util::logging::level>(lvl)) \
    /**/

#define LAPP_CONSOLE_(lvl)                                                     \
    HPX_LOG_USE_LOG(                                                           \
        hpx::util::app_console, static_cast<::hpx::util::logging::level>(lvl)) \
    /**/

#define LDEB_CONSOLE_                                                          \
    HPX_LOG_USE_LOG(                                                           \
        hpx::util::debuglog_console, ::hpx::util::logging::level::error)       \
/**/

// helper type to forward logging during bootstrap to two destinations
struct bootstrap_logging
{
    constexpr bootstrap_logging() {}
};

template <typename T>
bootstrap_logging const& operator<<(bootstrap_logging const& l, T const& t)
{
    LBT_(info) << t;
    LPROGRESS_ << t;
    return l;
}

constexpr bootstrap_logging lbt_;

#else
// logging is disabled all together

namespace hpx { namespace util {
    namespace detail {
        struct dummy_log_impl
        {
            constexpr dummy_log_impl() noexcept {}

            template <typename T>
            dummy_log_impl const& operator<<(T&&) const noexcept
            {
                return *this;
            }

            template <typename... Args>
            dummy_log_impl const& format(char const*, Args const&...) const
                noexcept
            {
                return *this;
            }
        };
        constexpr dummy_log_impl dummy_log;
    }    // namespace detail

    // clang-format off

    #define LAGAS_(lvl)           if(true) {} else hpx::util::detail::dummy_log
    #define LPT_(lvl)             if(true) {} else hpx::util::detail::dummy_log
    #define LTIM_(lvl)            if(true) {} else hpx::util::detail::dummy_log
    #define LPROGRESS_            if(true) {} else hpx::util::detail::dummy_log
    #define LHPX_(lvl, cat)       if(true) {} else hpx::util::detail::dummy_log
    #define LAPP_(lvl)            if(true) {} else hpx::util::detail::dummy_log
    #define LDEB_                 if(true) {} else hpx::util::detail::dummy_log

    #define LTM_(lvl)             if(true) {} else hpx::util::detail::dummy_log
    #define LRT_(lvl)             if(true) {} else hpx::util::detail::dummy_log
    #define LOSH_(lvl)            if(true) {} else hpx::util::detail::dummy_log
    #define LERR_(lvl)            if(true) {} else hpx::util::detail::dummy_log
    #define LLCO_(lvl)            if(true) {} else hpx::util::detail::dummy_log
    #define LPCS_(lvl)            if(true) {} else hpx::util::detail::dummy_log
    #define LAS_(lvl)             if(true) {} else hpx::util::detail::dummy_log
    #define LBT_(lvl)             if(true) {} else hpx::util::detail::dummy_log

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

    // clang-format on

}}    // namespace hpx::util

struct bootstrap_logging
{
    constexpr bootstrap_logging() {}
};
constexpr bootstrap_logging lbt_;

template <typename T>
bootstrap_logging const& operator<<(bootstrap_logging const& l, T&&)
{
    return l;
}

#endif

#endif /*HPX_LOGGING_HPP*/
