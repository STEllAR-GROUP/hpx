//  Copyright (c) 2007-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/logging/config/defines.hpp>
#include <hpx/logging/macros.hpp>

namespace hpx {

    HPX_CXX_EXPORT enum class logging_destination {
        hpx = 0,
        timing = 1,
        agas = 2,
        parcel = 3,
        app = 4,
        debuglog = 5
    };

#define HPX_LOGGING_DESTINATION_UNSCOPED_ENUM_DEPRECATION_MSG                  \
    "The unscoped logging_destination names are deprecated. Please use "       \
    "logging_destination::<value> instead."

    HPX_DEPRECATED_V(
        1, 9, HPX_LOGGING_DESTINATION_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr logging_destination destination_hpx =
        logging_destination::hpx;
    HPX_DEPRECATED_V(
        1, 9, HPX_LOGGING_DESTINATION_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr logging_destination destination_timing =
        logging_destination::timing;
    HPX_DEPRECATED_V(
        1, 9, HPX_LOGGING_DESTINATION_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr logging_destination destination_agas =
        logging_destination::agas;
    HPX_DEPRECATED_V(
        1, 9, HPX_LOGGING_DESTINATION_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr logging_destination destination_parcel =
        logging_destination::parcel;
    HPX_DEPRECATED_V(
        1, 9, HPX_LOGGING_DESTINATION_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr logging_destination destination_app =
        logging_destination::app;
    HPX_DEPRECATED_V(
        1, 9, HPX_LOGGING_DESTINATION_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr logging_destination destination_debuglog =
        logging_destination::debuglog;

#undef HPX_LOGGING_DESTINATION_UNSCOPED_ENUM_DEPRECATION_MSG
}    // namespace hpx

#if defined(HPX_HAVE_LOGGING)

#include <hpx/logging/detail/macros.hpp>
#include <hpx/logging/level.hpp>
#include <hpx/logging/logging.hpp>
#include <hpx/modules/assertion.hpp>
#include <hpx/modules/format.hpp>

#include <string>

////////////////////////////////////////////////////////////////////////////////
// clang-format off
namespace hpx::util {

    ////////////////////////////////////////////////////////////////////////////
    namespace detail {

        HPX_CXX_EXPORT  [[nodiscard]] HPX_CORE_EXPORT
        hpx::util::logging::level get_log_level(std::string const& env,
            bool allow_always = false);
    }

    ////////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(hpx)
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(hpx_console)

    ////////////////////////////////////////////////////////////////////////////
    // errors are logged in a special manner (always to cerr and additionally,
    // if enabled to 'normal' logging destination as well)
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(hpx_error)

#if defined(HPX_LOGGING_HAVE_SEPARATE_DESTINATIONS)
    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(agas)
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(agas_console)

    ////////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(parcel)
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(parcel_console)

    ////////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(timing)
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(timing_console)
#endif

    ////////////////////////////////////////////////////////////////////////////
    // Application specific logging
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(app)
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(app_console)

    ////////////////////////////////////////////////////////////////////////////
    // special debug logging channel
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(debuglog)
    HPX_CXX_EXPORT  HPX_CORE_EXPORT HPX_DECLARE_LOG(debuglog_console)
}    // namespace hpx::util
// clang-format on

// helper type to forward logging during bootstrap to two destinations
HPX_CXX_EXPORT struct bootstrap_logging
{
    constexpr bootstrap_logging() noexcept = default;
};

HPX_CXX_EXPORT template <typename T>
bootstrap_logging const& operator<<(
    bootstrap_logging const& l, T const& t)    //-V835
{
    LBT_(info) << t;
    LPROGRESS_ << t;
    return l;    // NOLINT(bugprone-return-const-ref-from-parameter)
}

HPX_CXX_EXPORT inline constexpr bootstrap_logging lbt_{};

#else

// logging is disabled all together
namespace hpx::util::detail {

    HPX_CXX_EXPORT struct dummy_log_impl
    {
        constexpr dummy_log_impl() noexcept = default;

        template <typename T>
        dummy_log_impl const& operator<<(T&&) const noexcept
        {
            return *this;
        }

        template <typename... Args>
        dummy_log_impl const& format(char const*, Args const&...) const noexcept
        {
            return *this;
        }
    };

    HPX_CXX_EXPORT inline constexpr dummy_log_impl dummy_log;
}    // namespace hpx::util::detail

HPX_CXX_EXPORT struct bootstrap_logging
{
    constexpr bootstrap_logging() noexcept = default;
};

HPX_CXX_EXPORT inline constexpr bootstrap_logging lbt_{};

HPX_CXX_EXPORT template <typename T>
constexpr bootstrap_logging const& operator<<(
    bootstrap_logging const& l, T&&) noexcept
{
    return l;
}

#endif
