//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2017 Shoshana Jakobovits
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/debugging/backtrace.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/runtime_local/config_entry.hpp>
#include <hpx/runtime_local/custom_exception_info.hpp>
#include <hpx/runtime_local/debugging.hpp>
#include <hpx/runtime_local/runtime_handlers.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <asio/io_context.hpp>

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>

namespace hpx { namespace detail {

    HPX_NORETURN void assertion_handler(
        hpx::assertion::source_location const& loc, const char* expr,
        std::string const& msg)
    {
        static thread_local bool handling_assertion = false;

        if (handling_assertion)
        {
            std::ostringstream strm;
            strm << "Trying to handle failed assertion while handling another "
                    "failed assertion!"
                 << std::endl;
            strm << "Assertion '" << expr << "' failed";
            if (!msg.empty())
            {
                strm << " (" << msg << ")";
            }

            strm << std::endl;
            strm << "{file}: " << loc.file_name << std::endl;
            strm << "{line}: " << loc.line_number << std::endl;
            strm << "{function}: " << loc.function_name << std::endl;

            std::cerr << strm.str();

            std::abort();
        }

        handling_assertion = true;

        hpx::util::may_attach_debugger("exception");

        std::ostringstream strm;
        strm << "Assertion '" << expr << "' failed";
        if (!msg.empty())
        {
            strm << " (" << msg << ")";
        }

        hpx::exception e(hpx::assertion_failure, strm.str());
        std::cerr << hpx::diagnostic_information(hpx::detail::get_exception(
                         e, loc.function_name, loc.file_name, loc.line_number))
                  << std::endl;
        std::abort();
    }

#if defined(HPX_HAVE_APEX)
    bool enable_parent_task_handler()
    {
        return !hpx::is_networking_enabled();
    }
#endif

    void test_failure_handler()
    {
        hpx::util::may_attach_debugger("test-failure");
    }

#if defined(HPX_HAVE_VERIFY_LOCKS)
    void registered_locks_error_handler()
    {
        std::string back_trace = hpx::util::trace(std::size_t(128));

        // throw or log, depending on config options
        if (get_config_entry("hpx.throw_on_held_lock", "1") == "0")
        {
            if (back_trace.empty())
            {
                LERR_(debug).format(
                    "suspending thread while at least one lock is being held "
                    "(stack backtrace was disabled at compile time)");
            }
            else
            {
                LERR_(debug).format("suspending thread while at least one lock "
                                    "is being held, stack backtrace: {}",
                    back_trace);
            }
        }
        else
        {
            if (back_trace.empty())
            {
                HPX_THROW_EXCEPTION(invalid_status, "verify_no_locks",
                    "suspending thread while at least one lock is "
                    "being held (stack backtrace was disabled at "
                    "compile time)");
            }
            else
            {
                HPX_THROW_EXCEPTION(invalid_status, "verify_no_locks",
                    "suspending thread while at least one lock is "
                    "being held, stack backtrace: {}",
                    back_trace);
            }
        }
    }

    bool register_locks_predicate()
    {
        return threads::get_self_ptr() != nullptr;
    }
#endif

    threads::thread_pool_base* get_default_pool()
    {
        hpx::runtime* rt = get_runtime_ptr();
        if (rt == nullptr)
        {
            HPX_THROW_EXCEPTION(invalid_status, "hpx::detail::get_default_pool",
                "The runtime system is not active");
        }

        return &rt->get_thread_manager().default_pool();
    }

    asio::io_context* get_default_timer_service()
    {
        hpx::runtime* rt = get_runtime_ptr();
        if (rt == nullptr)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::detail::get_default_timer_service",
                "The runtime system is not active");
        }

        return &get_thread_pool("timer-pool")->get_io_service();
    }

    threads::mask_cref_type get_pu_mask(
        threads::topology& /* topo */, std::size_t thread_num)
    {
        auto& rp = hpx::resource::get_partitioner();
        return rp.get_pu_mask(thread_num);
    }
}}    // namespace hpx::detail
