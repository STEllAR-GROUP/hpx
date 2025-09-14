//  Copyright (c) 2007-2025 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_LOGGING)
#include <hpx/logging/config/defines.hpp>
#include <hpx/logging/api.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/modules/format.hpp>

#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util {

    HPX_DEFINE_LOG(app, disable_all)
    HPX_DEFINE_LOG(app_console, disable_all)
    HPX_DEFINE_LOG(app_error, fatal)
    HPX_DEFINE_LOG(debuglog, disable_all)
    HPX_DEFINE_LOG(debuglog_console, disable_all)
    HPX_DEFINE_LOG(debuglog_error, fatal)
    HPX_DEFINE_LOG(hpx, disable_all)
    HPX_DEFINE_LOG(hpx_console, disable_all)
    HPX_DEFINE_LOG(hpx_error, fatal)

#if defined(HPX_LOGGING_HAVE_SEPARATE_DESTINATIONS)
    HPX_DEFINE_LOG(agas, disable_all)
    HPX_DEFINE_LOG(agas_console, disable_all)
    HPX_DEFINE_LOG(parcel, disable_all)
    HPX_DEFINE_LOG(parcel_console, disable_all)
    HPX_DEFINE_LOG(timing, disable_all)
    HPX_DEFINE_LOG(timing_console, disable_all)
#endif

    namespace detail {

        hpx::util::logging::level get_log_level(
            std::string const& env, bool allow_always)
        {
            try
            {
                int const env_val = hpx::util::from_string<int>(env);
                if (env_val < 0)
                    return hpx::util::logging::level::disable_all;

                switch (env_val)
                {
                case 0:
                    return allow_always ?
                        hpx::util::logging::level::always :
                        hpx::util::logging::level::disable_all;
                case 1:
                    return hpx::util::logging::level::fatal;
                case 2:
                    return hpx::util::logging::level::error;
                case 3:
                    return hpx::util::logging::level::warning;
                case 4:
                    return hpx::util::logging::level::info;
                default:
                    break;
                }
                return hpx::util::logging::level::debug;
            }
            catch (hpx::util::bad_lexical_cast const&)
            {
                return hpx::util::logging::level::disable_all;
            }
        }
    }    // namespace detail
}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
#include <hpx/logging/detail/logger.hpp>

namespace hpx::util::logging {

    void logger::turn_cache_off() const
    {
        if (m_is_caching_off)
            return;    // already turned off

        m_is_caching_off = true;

        // dump messages
        std::vector<message> msgs;
        std::swap(m_cache, msgs);

        for (auto& msg : msgs)
            m_writer(msg);
    }
}    // namespace hpx::util::logging

#include <hpx/config/warnings_suffix.hpp>

#endif    // HPX_HAVE_LOGGING
