//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2017 Shoshana Jakobovits
//  Copyright (c) 2010-2011 Phillip LeBlanc, Dylan Stark
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/custom_exception_info.hpp>
#include <hpx/errors.hpp>
#include <hpx/logging.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime_handlers.hpp>
#include <hpx/util/backtrace.hpp>
#include <hpx/util/debugging.hpp>

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>

namespace hpx { namespace detail {

    HPX_NORETURN void assertion_handler(
        hpx::assertion::source_location const& loc, const char* expr,
        std::string const& msg)
    {
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

    bool enable_parent_task_handler()
    {
        return (hpx::get_initial_num_localities() == 1);
    }

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
                LERR_(debug) << "suspending thread while at least one lock is "
                                "being held (stack backtrace was disabled at "
                                "compile time)";
            }
            else
            {
                LERR_(debug) << "suspending thread while at least one lock is "
                             << "being held, stack backtrace: " << back_trace;
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
                    "being held, stack backtrace: " +
                        back_trace);
            }
        }
    }

    bool register_locks_predicate()
    {
        return threads::get_self_ptr() != nullptr;
    }
#endif
}}    // namespace hpx::detail
