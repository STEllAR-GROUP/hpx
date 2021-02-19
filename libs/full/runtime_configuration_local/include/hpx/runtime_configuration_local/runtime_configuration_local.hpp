//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/ini/ini.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/filesystem.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace local { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    // The runtime_configuration class is a wrapper for the runtime
    // configuration data allowing to extract configuration information in a
    // more convenient way
    class HPX_EXPORT runtime_configuration : public hpx::util::section
    {
    protected:
        std::string hpx_ini_file;
        std::vector<std::string> cmdline_ini_defs;

    public:
        // initialize and load configuration information
        runtime_configuration(char const* argv0, bool pre_initialize = true);

        // re-initialize all entries based on the additional information from
        // the given configuration file
        void reconfigure(std::string const& ini_file);

        // re-initialize all entries based on the additional information from
        // any explicit command line options
        void reconfigure(std::vector<std::string> const& ini_defs);

        // Load application specific configuration and merge it with the
        // default configuration loaded from hpx.ini
        bool load_application_configuration(
            char const* filename, error_code& ec = throws);

        // Can be set to true if we want to use the ITT notify tools API.
        bool get_itt_notify_mode() const;

        // Enable lock detection during suspension
        bool enable_lock_detection() const;

        // Enable minimal deadlock detection for HPX threads
        bool enable_minimal_deadlock_detection() const;
        bool enable_spinlock_deadlock_detection() const;
        std::size_t get_spinlock_deadlock_detection_limit() const;

#if defined(__linux) || defined(linux) || defined(__linux__) ||                \
    defined(__FreeBSD__)
        bool use_stack_guard_pages() const;
#endif

        // return trace_depth for stack-backtraces
        std::size_t trace_depth() const;

        // Returns the number of OS threads this locality is running.
        std::size_t get_os_thread_count() const;

        // Returns the command line that this locality was invoked with.
        std::string get_cmd_line() const;

        // Will return the default stack size to use for all HPX-threads.
        std::ptrdiff_t get_default_stack_size() const
        {
            return small_stacksize;
        }

        // Will return the requested stack size to use for an HPX-threads.
        std::ptrdiff_t get_stack_size(
            threads::thread_stacksize stacksize) const;

        // Return the configured sizes of any of the know thread pools
        std::size_t get_thread_pool_size(char const* poolname) const;

    protected:
        std::ptrdiff_t init_stack_size(char const* entryname,
            char const* defaultvaluestr, std::ptrdiff_t defaultvalue) const;

        std::ptrdiff_t init_small_stack_size() const;
        std::ptrdiff_t init_medium_stack_size() const;
        std::ptrdiff_t init_large_stack_size() const;
        std::ptrdiff_t init_huge_stack_size() const;

        virtual void pre_initialize_ini();
        void post_initialize_ini(std::string& hpx_ini_file,
            std::vector<std::string> const& cmdline_ini_defs);
        void pre_initialize_logging_ini();

        void reconfigure();

    protected:
        std::ptrdiff_t small_stacksize;
        std::ptrdiff_t medium_stacksize;
        std::ptrdiff_t large_stacksize;
        std::ptrdiff_t huge_stacksize;
        bool need_to_call_pre_initialize;
#if defined(__linux) || defined(linux) || defined(__linux__)
        char const* argv0;
#endif
    };
}}}    // namespace hpx::local::detail
