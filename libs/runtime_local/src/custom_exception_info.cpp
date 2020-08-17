//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/command_line_handling/command_line_handling.hpp>
#include <hpx/debugging/backtrace.hpp>
#include <hpx/execution_base/register_locks.hpp>
#include <hpx/futures/futures_factory.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/naming_base.hpp>
#include <hpx/modules/threading.hpp>
#include <hpx/modules/threadmanager.hpp>
#include <hpx/runtime_local/config_entry.hpp>
#include <hpx/runtime_local/custom_exception_info.hpp>
#include <hpx/runtime_local/debugging.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime_local/get_worker_thread_num.hpp>
#include <hpx/runtime_local/runtime_local.hpp>
#include <hpx/state.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/version.hpp>

#if defined(HPX_WINDOWS)
#include <process.h>
#elif defined(HPX_HAVE_UNISTD_H)
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#elif defined(__FreeBSD__)
HPX_EXPORT char** freebsd_environ = nullptr;
#elif !defined(HPX_WINDOWS)
extern char** environ;
#endif

namespace hpx {
    char const* get_runtime_state_name(state st);

    ///////////////////////////////////////////////////////////////////////////
    // For testing purposes we sometime expect to see exceptions, allow those
    // to go through without attaching a debugger.
    std::atomic<bool> expect_exception_flag(false);

    bool expect_exception(bool flag)
    {
        return expect_exception_flag.exchange(flag);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Extract the diagnostic information embedded in the given exception and
    // return a string holding a formatted message.
    std::string diagnostic_information(hpx::exception_info const& xi)
    {
        int const verbosity = util::from_string<int>(
            get_config_entry("hpx.exception_verbosity", "2"));

        std::ostringstream strm;
        strm << "\n";

        // add full build information
        if (verbosity >= 2)
        {
            strm << full_build_string();

            std::string const* env = xi.get<hpx::detail::throw_env>();
            if (env && !env->empty())
                strm << "{env}: " << *env;
        }

        if (verbosity >= 1)
        {
            std::string const* back_trace =
                xi.get<hpx::detail::throw_stacktrace>();
            if (back_trace && !back_trace->empty())
            {
                // FIXME: add indentation to stack frame information
                strm << "{stack-trace}: " << *back_trace << "\n";
            }

            std::uint32_t const* locality =
                xi.get<hpx::detail::throw_locality>();
            if (locality)
                strm << "{locality-id}: " << *locality << "\n";

            std::string const* hostname_ =
                xi.get<hpx::detail::throw_hostname>();
            if (hostname_ && !hostname_->empty())
                strm << "{hostname}: " << *hostname_ << "\n";

            std::int64_t const* pid_ = xi.get<hpx::detail::throw_pid>();
            if (pid_ && -1 != *pid_)
                strm << "{process-id}: " << *pid_ << "\n";

            bool thread_info = false;
            char const* const thread_prefix = "{os-thread}: ";
            std::size_t const* shepherd = xi.get<hpx::detail::throw_shepherd>();
            if (shepherd && std::size_t(-1) != *shepherd)
            {
                strm << thread_prefix << *shepherd;
                thread_info = true;
            }

            std::string thread_name = hpx::get_thread_name();
            if (!thread_info)
                strm << thread_prefix;
            else
                strm << ", ";
            strm << thread_name << "\n";

            std::size_t const* thread_id =
                xi.get<hpx::detail::throw_thread_id>();
            if (thread_id && *thread_id)
            {
                strm << "{thread-id}: ";
                hpx::util::format_to(strm, "{:016x}\n", *thread_id);
            }

            std::string const* thread_description =
                xi.get<hpx::detail::throw_thread_name>();
            if (thread_description && !thread_description->empty())
                strm << "{thread-description}: " << *thread_description << "\n";

            std::string const* state = xi.get<hpx::detail::throw_state>();
            if (state)
                strm << "{state}: " << *state << "\n";

            std::string const* auxinfo = xi.get<hpx::detail::throw_auxinfo>();
            if (auxinfo)
                strm << "{auxinfo}: " << *auxinfo << "\n";
        }

        std::string const* file = xi.get<hpx::detail::throw_file>();
        if (file)
            strm << "{file}: " << *file << "\n";

        long const* line = xi.get<hpx::detail::throw_line>();
        if (line)
            strm << "{line}: " << *line << "\n";

        std::string const* function = xi.get<hpx::detail::throw_function>();
        if (function)
            strm << "{function}: " << *function << "\n";

        // Try a cast to std::exception - this should handle boost.system
        // error codes in addition to the standard library exceptions.
        std::exception const* se = dynamic_cast<std::exception const*>(&xi);
        if (se)
            strm << "{what}: " << se->what() << "\n";

        return strm.str();
    }
}    // namespace hpx

namespace hpx { namespace util {
    // This is a local helper used to get the backtrace on a new new stack if
    // possible.
    std::string trace_on_new_stack(
        std::size_t frames_no = HPX_HAVE_THREAD_BACKTRACE_DEPTH)
    {
#if defined(HPX_HAVE_STACKTRACES)
        if (frames_no == 0)
        {
            return std::string();
        }

        backtrace bt(frames_no);

        auto* self = threads::get_self_ptr();
        if (nullptr == self ||
            self->get_thread_id() == threads::invalid_thread_id)
        {
            return bt.trace();
        }

        lcos::local::futures_factory<std::string()> p(
            [&bt]() { return bt.trace(); });

        error_code ec(lightweight);
        threads::thread_id_type tid = p.apply("hpx::util::trace_on_new_stack",
            launch::fork, threads::thread_priority_default,
            threads::thread_stacksize_medium, threads::thread_schedule_hint(),
            ec);
        if (ec)
            return "<couldn't retrieve stack backtrace>";

        // make sure this thread is executed last
        hpx::this_thread::yield_to(thread::id(std::move(tid)));

        return p.get_future().get(ec);
#else
        return "";
#endif
    }
}}    // namespace hpx::util

namespace hpx { namespace detail {
    void pre_exception_handler()
    {
        if (!expect_exception_flag.load(std::memory_order_relaxed))
        {
            hpx::util::may_attach_debugger("exception");
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // report an early or late exception and abort
    void report_exception_and_continue(std::exception_ptr const& e)
    {
        pre_exception_handler();

        std::cerr << diagnostic_information(e) << std::endl;
    }

    void report_exception_and_continue(hpx::exception const& e)
    {
        pre_exception_handler();

        std::cerr << diagnostic_information(e) << std::endl;
    }

    void report_exception_and_terminate(std::exception_ptr const& e)
    {
        report_exception_and_continue(e);
        std::abort();
    }

    void report_exception_and_terminate(hpx::exception const& e)
    {
        report_exception_and_continue(e);
        std::abort();
    }

    hpx::exception_info construct_exception_info(std::string const& func,
        std::string const& file, long line, std::string const& back_trace,
        std::uint32_t node, std::string const& hostname, std::int64_t pid,
        std::size_t shepherd, std::size_t thread_id,
        std::string const& thread_name, std::string const& env,
        std::string const& config, std::string const& state_name,
        std::string const& auxinfo)
    {
        return hpx::exception_info().set(
            hpx::detail::throw_stacktrace(back_trace),
            hpx::detail::throw_locality(node),
            hpx::detail::throw_hostname(hostname), hpx::detail::throw_pid(pid),
            hpx::detail::throw_shepherd(shepherd),
            hpx::detail::throw_thread_id(thread_id),
            hpx::detail::throw_thread_name(thread_name),
            hpx::detail::throw_function(func), hpx::detail::throw_file(file),
            hpx::detail::throw_line(line), hpx::detail::throw_env(env),
            hpx::detail::throw_config(config),
            hpx::detail::throw_state(state_name),
            hpx::detail::throw_auxinfo(auxinfo));
    }

    template <typename Exception>
    std::exception_ptr construct_exception(
        Exception const& e, hpx::exception_info info)
    {
        // create a std::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with all the local information we have
        try
        {
            throw_with_info(e, std::move(info));
        }
        catch (...)
        {
            return std::current_exception();
        }

        // need this return to silence a warning with icc
        HPX_ASSERT(false);    // -V779
        return std::exception_ptr();
    }

    template HPX_EXPORT std::exception_ptr construct_exception(
        hpx::exception const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        boost::system::system_error const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        std::exception const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        hpx::detail::std_exception const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        std::bad_exception const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        hpx::detail::bad_exception const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        std::bad_typeid const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        hpx::detail::bad_typeid const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        std::bad_cast const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        hpx::detail::bad_cast const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        std::bad_alloc const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        hpx::detail::bad_alloc const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        std::logic_error const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        std::runtime_error const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        std::out_of_range const&, hpx::exception_info info);
    template HPX_EXPORT std::exception_ptr construct_exception(
        std::invalid_argument const&, hpx::exception_info info);

    ///////////////////////////////////////////////////////////////////////////
    //  Figure out the size of the given environment
    inline std::size_t get_arraylen(char** array)
    {
        std::size_t count = 0;
        if (nullptr != array)
        {
            while (nullptr != array[count])
                ++count;    // simply count the environment strings
        }
        return count;
    }

    std::string get_execution_environment()
    {
        std::vector<std::string> env;

#if defined(HPX_WINDOWS)
        std::size_t len = get_arraylen(_environ);
        env.reserve(len);
        std::copy(&_environ[0], &_environ[len], std::back_inserter(env));
#elif defined(linux) || defined(__linux) || defined(__linux__) ||              \
    defined(__AIX__)
        std::size_t len = get_arraylen(environ);
        env.reserve(len);
        std::copy(&environ[0], &environ[len], std::back_inserter(env));
#elif defined(__FreeBSD__)
        std::size_t len = get_arraylen(freebsd_environ);
        env.reserve(len);
        std::copy(&freebsd_environ[0], &freebsd_environ[len],
            std::back_inserter(env));
#elif defined(__APPLE__)
        std::size_t len = get_arraylen(environ);
        env.reserve(len);
        std::copy(&environ[0], &environ[len], std::back_inserter(env));
#else
#error "Don't know, how to access the execution environment on this platform"
#endif

        std::sort(env.begin(), env.end());

        static constexpr char const* ignored_env_patterns[] = {
            "DOCKER", "GITHUB_TOKEN"};
        std::string retval = hpx::util::format("{} entries:\n", env.size());
        for (std::string const& s : env)
        {
            if (std::all_of(std::begin(ignored_env_patterns),
                    std::end(ignored_env_patterns), [&s](auto const e) {
                        return s.find(e) == std::string::npos;
                    }))
            {
                retval += "  " + s + "\n";
            }
        }
        return retval;
    }

    hpx::exception_info custom_exception_info(std::string const& func,
        std::string const& file, long line, std::string const& auxinfo)
    {
        std::int64_t pid = ::getpid();

        std::size_t const trace_depth =
            util::from_string<std::size_t>(get_config_entry(
                "hpx.trace_depth", HPX_HAVE_THREAD_BACKTRACE_DEPTH));

        std::string back_trace(hpx::util::trace_on_new_stack(trace_depth));

        std::string state_name("not running");
        std::string hostname;
        hpx::runtime* rt = get_runtime_ptr();
        if (rt)
        {
            state rts_state = rt->get_state();
            state_name = get_runtime_state_name(rts_state);

            if (rts_state >= state_initialized && rts_state < state_stopped)
            {
                std::ostringstream strm;
                strm << get_runtime().here();
                hostname = strm.str();
            }
        }

        // if this is not a HPX thread we do not need to query neither for
        // the shepherd thread nor for the thread id
        error_code ec(lightweight);
        std::uint32_t node = get_locality_id(ec);

        std::size_t shepherd = std::size_t(-1);
        threads::thread_id_type thread_id;
        util::thread_description thread_name;

        threads::thread_self* self = threads::get_self_ptr();
        if (nullptr != self)
        {
            if (threads::threadmanager_is(state_running))
                shepherd = hpx::get_worker_thread_num();

            thread_id = threads::get_self_id();
            thread_name = threads::get_thread_description(thread_id);
        }

        std::string env(get_execution_environment());
        std::string config(configuration_string());

        return hpx::exception_info().set(
            hpx::detail::throw_stacktrace(back_trace),
            hpx::detail::throw_locality(node),
            hpx::detail::throw_hostname(hostname), hpx::detail::throw_pid(pid),
            hpx::detail::throw_shepherd(shepherd),
            hpx::detail::throw_thread_id(
                reinterpret_cast<std::size_t>(thread_id.get())),
            hpx::detail::throw_thread_name(util::as_string(thread_name)),
            hpx::detail::throw_function(func), hpx::detail::throw_file(file),
            hpx::detail::throw_line(line), hpx::detail::throw_env(env),
            hpx::detail::throw_config(config),
            hpx::detail::throw_state(state_name),
            hpx::detail::throw_auxinfo(auxinfo));
    }
}}    // namespace hpx::detail

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    /// Return the host-name of the locality where the exception was thrown.
    std::string get_error_host_name(hpx::exception_info const& xi)
    {
        std::string const* hostname_ = xi.get<hpx::detail::throw_hostname>();
        if (hostname_ && !hostname_->empty())
            return *hostname_;
        return std::string();
    }

    /// Return the locality where the exception was thrown.
    std::uint32_t get_error_locality_id(hpx::exception_info const& xi)
    {
        std::uint32_t const* locality = xi.get<hpx::detail::throw_locality>();
        if (locality)
            return *locality;
        return naming::invalid_locality_id;
    }

    /// Return the (operating system) process id of the locality where the
    /// exception was thrown.
    std::int64_t get_error_process_id(hpx::exception_info const& xi)
    {
        std::int64_t const* pid_ = xi.get<hpx::detail::throw_pid>();
        if (pid_)
            return *pid_;
        return -1;
    }

    /// Return the environment of the OS-process at the point the exception
    /// was thrown.
    std::string get_error_env(hpx::exception_info const& xi)
    {
        std::string const* env = xi.get<hpx::detail::throw_env>();
        if (env && !env->empty())
            return *env;

        return "<unknown>";
    }

    /// Return the stack backtrace at the point the exception was thrown.
    std::string get_error_backtrace(hpx::exception_info const& xi)
    {
        std::string const* back_trace = xi.get<hpx::detail::throw_stacktrace>();
        if (back_trace && !back_trace->empty())
            return *back_trace;

        return std::string();
    }

    /// Return the sequence number of the OS-thread used to execute HPX-threads
    /// from which the exception was thrown.
    std::size_t get_error_os_thread(hpx::exception_info const& xi)
    {
        std::size_t const* shepherd = xi.get<hpx::detail::throw_shepherd>();
        if (shepherd && std::size_t(-1) != *shepherd)
            return *shepherd;
        return std::size_t(-1);
    }

    /// Return the unique thread id of the HPX-thread from which the exception
    /// was thrown.
    std::size_t get_error_thread_id(hpx::exception_info const& xi)
    {
        std::size_t const* thread_id = xi.get<hpx::detail::throw_thread_id>();
        if (thread_id && *thread_id)
            return *thread_id;
        return std::size_t(-1);
    }

    /// Return any addition thread description of the HPX-thread from which the
    /// exception was thrown.
    std::string get_error_thread_description(hpx::exception_info const& xi)
    {
        std::string const* thread_description =
            xi.get<hpx::detail::throw_thread_name>();
        if (thread_description && !thread_description->empty())
            return *thread_description;
        return std::string();
    }

    /// Return the HPX configuration information point from which the
    /// exception was thrown.
    std::string get_error_config(hpx::exception_info const& xi)
    {
        std::string const* config_info = xi.get<hpx::detail::throw_config>();
        if (config_info && !config_info->empty())
            return *config_info;
        return std::string();
    }

    /// Return the HPX runtime state information at which the exception was
    /// thrown.
    std::string get_error_state(hpx::exception_info const& xi)
    {
        std::string const* state_info = xi.get<hpx::detail::throw_state>();
        if (state_info && !state_info->empty())
            return *state_info;
        return std::string();
    }
}    // namespace hpx
