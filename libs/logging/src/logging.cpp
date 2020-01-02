//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_LOGGING)
#include <hpx/filesystem.hpp>
#include <hpx/logging.hpp>
#include <hpx/logging/format/destination/defaults.hpp>
#include <hpx/logging/format/named_write.hpp>
#include <hpx/util/from_string.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
// 'class1' : inherits 'class2::member' via dominance
#pragma warning(disable : 4250)
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {
    HPX_DEFINE_LOG(agas, disable_all)
    HPX_DEFINE_LOG(agas_console, disable_all)
    HPX_DEFINE_LOG(app, disable_all)
    HPX_DEFINE_LOG(app_console, disable_all)
    HPX_DEFINE_LOG(app_error, fatal)
    HPX_DEFINE_LOG(debuglog, disable_all)
    HPX_DEFINE_LOG(debuglog_console, disable_all)
    HPX_DEFINE_LOG(debuglog_error, fatal)
    HPX_DEFINE_LOG(hpx, disable_all)
    HPX_DEFINE_LOG(hpx_console, disable_all)
    HPX_DEFINE_LOG(hpx_error, fatal)
    HPX_DEFINE_LOG(parcel, disable_all)
    HPX_DEFINE_LOG(parcel_console, disable_all)
    HPX_DEFINE_LOG(timing, disable_all)
    HPX_DEFINE_LOG(timing_console, disable_all)

    namespace detail {
        hpx::util::logging::level get_log_level(
            std::string const& env, bool allow_always)
        {
            try
            {
                int env_val = hpx::util::from_string<int>(env);
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
}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
#include <hpx/logging/detail/cache_before_init.hpp>

namespace hpx { namespace util { namespace logging { namespace detail {
    void cache_before_init::turn_cache_off(writer::named_write const& writer_)
    {
        if (m_is_caching_off)
            return;    // already turned off

        m_is_caching_off = true;

        // dump messages
        message_array msgs;
        std::swap(m_cache, msgs);

        for (auto& msg : msgs)
        {
            writer_(msg);
        }
    }
}}}}    // namespace hpx::util::logging::detail

///////////////////////////////////////////////////////////////////////////////
#include <hpx/logging/format/destination/file.hpp>

namespace hpx { namespace util { namespace logging { namespace destination {
    file::mutex_type file::mtx_ = BOOST_DETAIL_SPINLOCK_INIT;
}}}}    // namespace hpx::util::logging::destination

///////////////////////////////////////////////////////////////////////////////
#include <hpx/logging/format/destination/named.hpp>

namespace hpx { namespace util { namespace logging { namespace destination {
    namespace detail {
        void named_context::compute_write_steps()
        {
            m_info.write_steps.clear();

            std::istringstream in(m_info.format_string);
            std::string word;
            while (in >> word)
            {
                if (word[0] == '+')
                    word.erase(word.begin());
                else if (word[0] == '-')
                    // ignore this word
                    continue;

                if (m_info.name_to_destination.find(word) !=
                    m_info.name_to_destination.end())
                    m_info.write_steps.push_back(
                        m_info.name_to_destination.find(word)->second);
            }
        }
}}}}}    // namespace hpx::util::logging::destination::detail

///////////////////////////////////////////////////////////////////////////////
#include <hpx/logging/format/formatter/named_spacer.hpp>

namespace hpx { namespace util { namespace logging { namespace formatter {
    namespace detail {
        static std::string unescape(std::string escaped)
        {
            typedef std::size_t size_type;
            size_type idx_start = 0;
            while (true)
            {
                size_type found = escaped.find("%%", idx_start);
                if (found != std::string::npos)
                {
                    escaped.erase(
                        escaped.begin() + static_cast<std::ptrdiff_t>(found));
                    ++idx_start;
                }
                else
                    break;
            }
            return escaped;
        }

        void base_named_spacer_context::compute_write_steps()
        {
            typedef std::size_t size_type;

            m_info.write_steps.clear();
            std::string remaining = m_info.format_string;
            size_type start_search_idx = 0;
            while (!remaining.empty())
            {
                size_type idx = remaining.find('%', start_search_idx);
                if (idx != std::string::npos)
                {
                    // see if just escaped
                    if ((idx < remaining.size() - 1) &&
                        remaining[idx + 1] == '%')
                    {
                        // we found an escaped char
                        start_search_idx = idx + 2;
                        continue;
                    }

                    // up to here, this is a spacer string
                    start_search_idx = 0;
                    std::string spacer = unescape(remaining.substr(0, idx));
                    remaining = remaining.substr(idx + 1);
                    // find end of formatter name
                    idx = remaining.find('%');
                    format_base_type* fmt = nullptr;
                    if (idx != std::string::npos)
                    {
                        std::string name = remaining.substr(0, idx);
                        remaining = remaining.substr(idx + 1);
                        fmt = m_info.name_to_formatter[name];
                    }
                    // note: fmt could be null, in case
                    m_info.write_steps.push_back(write_step(spacer, fmt));
                }
                else
                {
                    // last part
                    m_info.write_steps.push_back(
                        write_step(unescape(remaining), nullptr));
                    remaining.clear();
                }
            }
        }
}}}}}    // namespace hpx::util::logging::formatter::detail

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif

#endif    // HPX_HAVE_LOGGING
