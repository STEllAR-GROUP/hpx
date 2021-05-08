//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_LOGGING)
#include <hpx/ini/ini.hpp>
#include <hpx/init_runtime_local/detail/init_logging.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/agas/addressing_service.hpp>
#include <hpx/runtime_components/console_logging.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <string>
#include <vector>

#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#endif

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(                                                               \
    disable : 4250)    // 'class1' : inherits 'class2::member' via dominance
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX component id of current thread
    struct thread_component_id : logging::formatter::manipulator
    {
        void operator()(std::ostream& to) const override
        {
            std::uint64_t component_id = threads::get_self_component_id();
            if (0 != component_id)
            {
                // called from inside a HPX thread
                util::format_to(to, "{:016x}", component_id);
            }
            else
            {
                // called from outside a HPX thread
                to << std::string(16, '-');
            }
        }
    };
#endif

    ///////////////////////////////////////////////////////////////////////////
    // custom log destination: send generated strings to console
    struct console : logging::destination::manipulator
    {
        console(logging::level level, logging_destination dest)
          : level_(level)
          , dest_(dest)
        {
        }

        void operator()(logging::message const& msg) override
        {
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
            components::console_logging(
                dest_, static_cast<std::size_t>(level_), msg.full_string());
#else
            switch (dest_)
            {
            default:
            case destination_hpx:
                LHPX_CONSOLE_(level_) << msg;
                break;

            case destination_timing:
                LTIM_CONSOLE_(level_) << msg;
                break;

            case destination_agas:
                LAGAS_CONSOLE_(level_) << msg;
                break;

            case destination_parcel:
                LPT_CONSOLE_(level_) << msg;
                break;

            case destination_app:
                LAPP_CONSOLE_(level_) << msg;
                break;

            case destination_debuglog:
                LDEB_CONSOLE_ << msg;
                break;
            }
#endif
        }

        bool operator==(console const& rhs) const
        {
            return dest_ == rhs.dest_;
        }

        logging::level level_;
        logging_destination dest_;
    };    // namespace util

    namespace detail {
        inline void define_formatters(logger_writer_type& writer)
        {
            writer.set_formatter("osthread", shepherd_thread_id());
            writer.set_formatter("locality", locality_prefix());
            writer.set_formatter("hpxthread", thread_id());
            writer.set_formatter("hpxphase", thread_phase());
            writer.set_formatter("hpxparent", parent_thread_id());
            writer.set_formatter("hpxparentphase", parent_thread_phase());
            writer.set_formatter("parentloc", parent_thread_locality());
#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
            writer.set_formatter("hpxcomponent", thread_component_id());
#endif
        }
    }    // namespace detail
}}       // namespace hpx::util

#else

#include <hpx/init_runtime_local/detail/init_logging.hpp>

namespace hpx { namespace util {
    struct console;
    namespace detail {
        inline void define_formatters() {}
    }    // namespace detail
}}       // namespace hpx::util

#endif
