//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>

#include <string>

#if defined(HPX_HAVE_LOGGING)
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {

    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    // custom log destination: send generated strings to console
    struct HPX_LOCAL_EXPORT console_local : logging::destination::manipulator
    {
        console_local(logging::level level, logging_destination dest)
          : level_(level)
          , dest_(dest)
        {
        }

        void operator()(logging::message const& msg) override;

        friend bool operator==(
            console_local const& lhs, console_local const& rhs)
        {
            return lhs.dest_ == rhs.dest_;
        }

        logging::level level_;
        logging_destination dest_;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        struct log_settings
        {
            std::string level_;
            std::string dest_;
            std::string format_;
        };

        HPX_LOCAL_EXPORT void define_common_formatters(
            logging::writer::named_write& writer);

        HPX_LOCAL_EXPORT void define_formatters_local(
            logging::writer::named_write& writer);

        HPX_LOCAL_EXPORT log_settings get_log_settings(
            util::section const&, char const*);

        HPX_LOCAL_EXPORT void init_logging(runtime_configuration& ini,
            bool isconsole,
            void (*set_console_dest)(logging::writer::named_write&, char const*,
                logging::level, logging_destination),
            void (*define_formatters)(logging::writer::named_write&));

        HPX_LOCAL_EXPORT void init_logging_local(runtime_configuration&);
    }    // namespace detail

    /// \endcond

    //////////////////////////////////////////////////////////////////////////
    /// Enable logging for given destination
    HPX_LOCAL_EXPORT void enable_logging(logging_destination dest,
        std::string const& lvl = "5", std::string logdest = "",
        std::string logformat = "");

    /// Disable all logging for the given destination
    HPX_LOCAL_EXPORT void disable_logging(logging_destination dest);
}}    // namespace hpx::util

#else    // HPX_HAVE_LOGGING

namespace hpx { namespace util {
    namespace detail {

        HPX_LOCAL_EXPORT void warn_if_logging_requested(runtime_configuration&);
    }

    //////////////////////////////////////////////////////////////////////////
    HPX_LOCAL_EXPORT void enable_logging(logging_destination dest,
        std::string const& lvl = "5", std::string logdest = "",
        std::string logformat = "");

    HPX_LOCAL_EXPORT void disable_logging(logging_destination dest);
}}    // namespace hpx::util

#endif    // HPX_HAVE_LOGGING
