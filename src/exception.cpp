//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/error.hpp>
#include <hpx/error_code.hpp>
#include <hpx/exception.hpp>
#include <hpx/exception_info.hpp>
#include <hpx/state.hpp>
#include <hpx/version.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/backtrace.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/filesystem_compatibility.hpp>
#include <hpx/util/logging.hpp>

#if defined(HPX_WINDOWS)
#  include <process.h>
#elif defined(HPX_HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#include <boost/atomic.hpp>
#include <boost/format.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#elif defined(__FreeBSD__)
HPX_EXPORT char** freebsd_environ = nullptr;
#elif !defined(HPX_WINDOWS)
extern char **environ;
#endif

namespace hpx
{
    char const* get_runtime_state_name(state st);

    ///////////////////////////////////////////////////////////////////////////
    // For testing purposes we sometime expect to see exceptions, allow those
    // to go through without attaching a debugger.
    boost::atomic<bool> expect_exception_flag(false);

    bool expect_exception(bool flag)
    {
        return expect_exception_flag.exchange(flag);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Construct a hpx::exception from a \a hpx::error.
    ///
    /// \param e    The parameter \p e holds the hpx::error code the new
    ///             exception should encapsulate.
    exception::exception(error e)
      : boost::system::system_error(make_error_code(e, plain))
    {
        HPX_ASSERT((e >= success && e < last_error) || (e & system_error_flag));
        LERR_(error) << "created exception: " << this->what();
    }

    /// Construct a hpx::exception from a boost#system_error.
    exception::exception(boost::system::system_error const& e)
      : boost::system::system_error(e)
    {
        LERR_(error) << "created exception: " << this->what();
    }

    /// Construct a hpx::exception from a \a hpx::error and an error message.
    ///
    /// \param e      The parameter \p e holds the hpx::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               hpx::error_code belongs to the error category
    ///               \a hpx_category (if mode is \a plain, this is the
    ///               default) or to the category \a hpx_category_rethrow
    ///               (if mode is \a rethrow).
    exception::exception(error e, char const* msg, throwmode mode)
      : boost::system::system_error(make_system_error_code(e, mode), msg)
    {
        HPX_ASSERT((e >= success && e < last_error) || (e & system_error_flag));
        LERR_(error) << "created exception: " << this->what();
    }

    /// Construct a hpx::exception from a \a hpx::error and an error message.
    ///
    /// \param e      The parameter \p e holds the hpx::error code the new
    ///               exception should encapsulate.
    /// \param msg    The parameter \p msg holds the error message the new
    ///               exception should encapsulate.
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               hpx::error_code belongs to the error category
    ///               \a hpx_category (if mode is \a plain, this is the
    ///               default) or to the category \a hpx_category_rethrow
    ///               (if mode is \a rethrow).
    exception::exception(error e, std::string const& msg, throwmode mode)
      : boost::system::system_error(make_system_error_code(e, mode), msg)
    {
        HPX_ASSERT((e >= success && e < last_error) || (e & system_error_flag));
        LERR_(error) << "created exception: " << this->what();
    }

    /// Destruct a hpx::exception
    ///
    /// \throws nothing
    exception::~exception() throw()
    {
    }

    /// The function \a get_error() returns the hpx::error code stored
    /// in the referenced instance of a hpx::exception. It returns
    /// the hpx::error code this exception instance was constructed
    /// from.
    ///
    /// \throws nothing
    error exception::get_error() const noexcept
    {
        return static_cast<error>(
            this->boost::system::system_error::code().value());
    }

    /// The function \a get_error_code() returns a hpx::error_code which
    /// represents the same error condition as this hpx::exception instance.
    ///
    /// \param mode   The parameter \p mode specifies whether the returned
    ///               hpx::error_code belongs to the error category
    ///               \a hpx_category (if mode is \a plain, this is the
    ///               default) or to the category \a hpx_category_rethrow
    ///               (if mode is \a rethrow).
    error_code exception::get_error_code(throwmode mode) const noexcept
    {
        (void)mode;
        return error_code(this->boost::system::system_error::code().value(),
            *this);
    }
}

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    std::string backtrace(std::size_t frames)
    {
        return util::trace_on_new_stack(frames);
    }

    std::string backtrace_direct(std::size_t frames)
    {
        return util::trace(frames);
    }

    ///////////////////////////////////////////////////////////////////////////
    //  Figure out the size of the given environment
    inline std::size_t get_arraylen(char** array)
    {
        std::size_t count = 0;
        if (nullptr != array) {
            while(nullptr != array[count])
                ++count;   // simply count the environment strings
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
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__AIX__)
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

        std::string retval = boost::str(boost::format("%d entries:\n") % env.size());
        for (std::string const& s : env)
        {
            retval += "  " + s + "\n";
        }
        return retval;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Exception>
    HPX_EXPORT std::exception_ptr construct_exception(
        Exception const& e, std::string const& func,
        std::string const& file, long line, std::string const& back_trace,
        std::uint32_t node, std::string const& hostname, std::int64_t pid,
        std::size_t shepherd, std::size_t thread_id,
        std::string const& thread_name, std::string const& env,
        std::string const& config, std::string const& state_name,
        std::string const& auxinfo)
    {
        // create a std::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with all the local information we have
        try {
            throw_with_info(e, hpx::exception_info().set(
               hpx::detail::throw_stacktrace(back_trace),
               hpx::detail::throw_locality(node),
               hpx::detail::throw_hostname(hostname),
               hpx::detail::throw_pid(pid),
               hpx::detail::throw_shepherd(shepherd),
               hpx::detail::throw_thread_id(thread_id),
               hpx::detail::throw_thread_name(thread_name),
               hpx::detail::throw_function(func),
               hpx::detail::throw_file(file),
               hpx::detail::throw_line(line),
               hpx::detail::throw_env(env),
               hpx::detail::throw_config(config),
               hpx::detail::throw_state(state_name),
               hpx::detail::throw_auxinfo(auxinfo)));
        }
        catch (...) {
            return std::current_exception();
        }

        // need this return to silence a warning with icc
        HPX_ASSERT(false);
        return std::exception_ptr();
    }

    template <typename Exception>
    HPX_EXPORT std::exception_ptr construct_lightweight_exception(
        Exception const& e, std::string const& func, std::string const& file,
        long line)
    {
        // create a std::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with all the local information we have
        try {
            throw_with_info(e, hpx::exception_info().set(
               hpx::detail::throw_function(func),
               hpx::detail::throw_file(file),
               hpx::detail::throw_line(line)));
        }
        catch (...) {
            return std::current_exception();
        }

        // need this return to silence a warning with icc
        HPX_ASSERT(false);
        return std::exception_ptr();
    }

    template <typename Exception>
    HPX_EXPORT std::exception_ptr construct_lightweight_exception(Exception const& e)
    {
        // create a std::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with all the local information we have
        try {
            hpx::throw_with_info(e);
        }
        catch (...) {
            return std::current_exception();
        }

        // need this return to silence a warning with icc
        HPX_ASSERT(false);
        return std::exception_ptr();
    }

    template HPX_EXPORT std::exception_ptr
        construct_lightweight_exception(hpx::thread_interrupted const&);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Exception>
    inline bool is_of_lightweight_hpx_category(Exception const& e)
    {
        return false;
    }

    inline bool is_of_lightweight_hpx_category(hpx::exception const& e)
    {
        return e.get_error_code().category() == get_lightweight_hpx_category();
    }

    template <typename Exception>
    HPX_EXPORT std::exception_ptr
    get_exception(Exception const& e, std::string const& func,
        std::string const& file, long line, std::string const& auxinfo)
    {
        if (is_of_lightweight_hpx_category(e))
            return construct_lightweight_exception(e, func, file, line);

        std::int64_t pid = ::getpid();
        std::string back_trace(backtrace());

        std::string state_name("not running");
        std::string hostname;
        hpx::runtime* rt = get_runtime_ptr();
        if (rt)
        {
            state rts_state = rt->get_state();
            state_name = get_runtime_state_name(rts_state);

            if (rts_state >= state_initialized &&
                rts_state < state_stopped)
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

        return construct_exception(e, func, file, line, back_trace, node,
            hostname, pid, shepherd,
            reinterpret_cast<std::size_t>(thread_id.get()),
            util::as_string(thread_name), env, config,
            state_name, auxinfo);
    }

    template <typename Exception>
    HPX_EXPORT void throw_exception(Exception const& e, std::string const& func,
        std::string const& file, long line)
    {
        if (!expect_exception_flag.load(boost::memory_order_relaxed) &&
            get_config_entry("hpx.attach_debugger", "") == "exception")
        {
            util::attach_debugger();
        }
        std::rethrow_exception(get_exception(e, func, file, line));
    }

    ///////////////////////////////////////////////////////////////////////////
    template HPX_EXPORT std::exception_ptr
        get_exception(hpx::exception const&, std::string const&,
        std::string const&, long, std::string const&);

    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(hpx::exception const&,
        std::string const&, std::string const&, long);

    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(boost::system::system_error const&,
        std::string const&, std::string const&, long);

    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(std::exception const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(hpx::detail::std_exception const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(std::bad_exception const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(hpx::detail::bad_exception const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(std::bad_typeid const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(hpx::detail::bad_typeid const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(std::bad_cast const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(hpx::detail::bad_cast const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(std::bad_alloc const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(hpx::detail::bad_alloc const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(std::logic_error const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(std::runtime_error const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(std::out_of_range const&,
        std::string const&, std::string const&, long);
    template HPX_ATTRIBUTE_NORETURN HPX_EXPORT void
        throw_exception(std::invalid_argument const&,
        std::string const&, std::string const&, long);

    ///////////////////////////////////////////////////////////////////////////
    void assertion_failed(char const* expr, char const* function,
        char const* file, long line)
    {
        assertion_failed_msg(expr, expr, function, file, line);
    }

    void assertion_failed_msg(char const* msg, char const* expr,
        char const* function, char const* file, long line)
    {
        if (!expect_exception_flag.load(boost::memory_order_relaxed) &&
            get_config_entry("hpx.attach_debugger", "") == "exception")
        {
            util::attach_debugger();
        }

        bool threw = false;

        std::string str("assertion '" + std::string(msg) + "' failed");
        if (expr != msg)
            str += " (" + std::string(expr) + ")";

        try {
            boost::filesystem::path p(hpx::util::create_path(file));
            hpx::detail::throw_exception(
                hpx::exception(hpx::assertion_failure, str),
                function, p.string(), line);
        }
        catch (...) {
            threw = true;

            // If the runtime pointer is available, we can safely get the prefix
            // of this locality. If it's not available, then just terminate.
            runtime* rt = get_runtime_ptr();
            if (nullptr != rt)  {
                rt->report_error(std::current_exception());
            }
            else {
                std::cerr << "Runtime is not available, reporting error locally. "
                    << hpx::diagnostic_information(std::current_exception())
                    << std::flush;
            }
        }

        // If the exception wasn't thrown, then print out the assertion message,
        // so that the program doesn't abort without any diagnostics.
        if (!threw) {
            std::cerr << "Runtime is not available, reporting error locally\n"
                         "{what}: " << str << std::endl;
        }
        std::abort();
    }

    ///////////////////////////////////////////////////////////////////////////
    // report an early or late exception and abort
    void report_exception_and_continue(std::exception_ptr const& e)
    {
        if (!expect_exception_flag.load(boost::memory_order_relaxed) &&
            get_config_entry("hpx.attach_debugger", "") == "exception")
        {
            util::attach_debugger();
        }

        std::cerr << hpx::diagnostic_information(e) << std::endl;
    }

    void report_exception_and_continue(hpx::exception const& e)
    {
        if (!expect_exception_flag.load(boost::memory_order_relaxed) &&
            get_config_entry("hpx.attach_debugger", "") == "exception")
        {
            util::attach_debugger();
        }

        std::cerr << hpx::diagnostic_information(e) << std::endl;
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

    ///////////////////////////////////////////////////////////////////////////
    std::exception_ptr access_exception(error_code const& e)
    {
        return e.exception_;
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // Extract the diagnostic information embedded in the given exception and
    // return a string holding a formatted message.
    std::string diagnostic_information(hpx::exception_info const& xi)
    {
        std::ostringstream strm;
        strm << "\n";

        std::string const* back_trace =
            xi.get<hpx::detail::throw_stacktrace>();
        if (back_trace && !back_trace->empty()) {
            // FIXME: add indentation to stack frame information
            strm << "{stack-trace}: " << *back_trace << "\n";
        }

        std::string const* env =
            xi.get<hpx::detail::throw_env>();
        if (env && !env->empty())
            strm << "{env}: " << *env;

        std::uint32_t const* locality =
            xi.get<hpx::detail::throw_locality>();
        if (locality)
            strm << "{locality-id}: " << *locality << "\n";

        std::string const* hostname_ =
            xi.get<hpx::detail::throw_hostname>();
        if (hostname_ && !hostname_->empty())
            strm << "{hostname}: " << *hostname_ << "\n";

        std::int64_t const* pid_ =
            xi.get<hpx::detail::throw_pid>();
        if (pid_ && -1 != *pid_)
            strm << "{process-id}: " << *pid_ << "\n";

        std::string const* function =
            xi.get<hpx::detail::throw_function>();
        if (function)
            strm << "{function}: " << *function << "\n";

        std::string const* file =
            xi.get<hpx::detail::throw_file>();
        if (file)
            strm << "{file}: " << *file << "\n";

        long const* line =
            xi.get<hpx::detail::throw_line>();
        if (line)
            strm << "{line}: " << *line << "\n";

        bool thread_info = false;
        char const* const thread_prefix = "{os-thread}: ";
        std::size_t const* shepherd =
            xi.get<hpx::detail::throw_shepherd>();
        if (shepherd && std::size_t(-1) != *shepherd) {
            strm << thread_prefix << *shepherd;
            thread_info = true;
        }

        std::string thread_name = runtime::get_thread_name();
        if (!thread_info)
            strm << thread_prefix;
        else
            strm << ", ";
        strm << thread_name << "\n";

        std::size_t const* thread_id =
            xi.get<hpx::detail::throw_thread_id>();
        if (thread_id && *thread_id)
            strm << (boost::format("{thread-id}: %016x\n") % *thread_id);

        std::string const* thread_description =
            xi.get<hpx::detail::throw_thread_name>();
        if (thread_description && !thread_description->empty())
            strm << "{thread-description}: " << *thread_description << "\n";

        std::string const* state =
            xi.get<hpx::detail::throw_state>();
        if (state)
            strm << "{state}: " << *state << "\n";

        std::string const* auxinfo =
            xi.get<hpx::detail::throw_auxinfo>();
        if (auxinfo)
            strm << "{auxinfo}: " << *auxinfo << "\n";

        // add full build information
        strm << full_build_string();

        // Try a cast to std::exception - this should handle boost.system
        // error codes in addition to the standard library exceptions.
        std::exception const* se = dynamic_cast<std::exception const*>(&xi);
        if (se)
            strm << "{what}: " << se->what() << "\n";

        return strm.str();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Return the error message.
    std::string get_error_what(hpx::exception_info const& xi)
    {
        // Try a cast to std::exception - this should handle boost.system
        // error codes in addition to the standard library exceptions.
        std::exception const* se = dynamic_cast<std::exception const*>(&xi);
        return se ? se->what() : std::string("<unknown>");
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Return the locality where the exception was thrown.
    std::uint32_t get_error_locality_id(hpx::exception_info const& xi)
    {
        std::uint32_t const* locality =
            xi.get<hpx::detail::throw_locality>();
        if (locality)
            return *locality;
        return naming::invalid_locality_id;
    }

    ///////////////////////////////////////////////////////////////////////////
    error get_error(hpx::exception const& e)
    {
        return static_cast<hpx::error>(e.get_error());
    }

    error get_error(hpx::error_code const& e)
    {
        return static_cast<hpx::error>(e.value());
    }

    error get_error(std::exception_ptr const& e)
    {
        try {
            std::rethrow_exception(e);
        }
        catch (hpx::thread_interrupted const&) {
            return hpx::thread_cancelled;
        }
        catch (hpx::exception const& he) {
            return he.get_error();
        }
        catch (boost::system::system_error const& e) {
            int code = e.code().value();
            if (code < success || code >= last_error)
                code |= system_error_flag;
            return static_cast<hpx::error>(code);
        }
        catch (...) {
            return unknown_error;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Return the host-name of the locality where the exception was thrown.
    std::string get_error_host_name(hpx::exception_info const& xi)
    {
        std::string const* hostname_ =
            xi.get<hpx::detail::throw_hostname>();
        if (hostname_ && !hostname_->empty())
            return *hostname_;
        return std::string();
    }

    /// Return the (operating system) process id of the locality where the
    /// exception was thrown.
    std::int64_t get_error_process_id(hpx::exception_info const& xi)
    {
        std::int64_t const* pid_ =
            xi.get<hpx::detail::throw_pid>();
        if (pid_)
            return *pid_;
        return -1;
    }

    /// Return the environment of the OS-process at the point the exception
    /// was thrown.
    std::string get_error_env(hpx::exception_info const& xi)
    {
        std::string const* env =
            xi.get<hpx::detail::throw_env>();
        if (env && !env->empty())
            return *env;

        return "<unknown>";
    }

    /// Return the function name from which the exception was thrown.
    std::string get_error_function_name(hpx::exception_info const& xi)
    {
        std::string const* function =
            xi.get<hpx::detail::throw_function>();
        if (function)
            return *function;

        return std::string();
    }

    /// Return the stack backtrace at the point the exception was thrown.
    std::string get_error_backtrace(hpx::exception_info const& xi)
    {
        std::string const* back_trace =
            xi.get<hpx::detail::throw_stacktrace>();
        if (back_trace && !back_trace->empty())
            return *back_trace;

        return std::string();
    }

    /// Return the (source code) file name of the function from which the
    /// exception was thrown.
    std::string get_error_file_name(hpx::exception_info const& xi)
    {
        std::string const* file =
            xi.get<hpx::detail::throw_file>();
        if (file)
            return *file;

        return "<unknown>";
    }

    /// Return the line number in the (source code) file of the function from
    /// which the exception was thrown.
    long get_error_line_number(hpx::exception_info const& xi)
    {
        long const* line =
            xi.get<hpx::detail::throw_line>();
        if (line)
            return *line;
        return -1;
    }

    /// Return the sequence number of the OS-thread used to execute HPX-threads
    /// from which the exception was thrown.
    std::size_t get_error_os_thread(hpx::exception_info const& xi)
    {
        std::size_t const* shepherd =
            xi.get<hpx::detail::throw_shepherd>();
        if (shepherd && std::size_t(-1) != *shepherd)
            return *shepherd;
        return std::size_t(-1);
    }

    /// Return the unique thread id of the HPX-thread from which the exception
    /// was thrown.
    std::size_t get_error_thread_id(hpx::exception_info const& xi)
    {
        std::size_t const* thread_id =
            xi.get<hpx::detail::throw_thread_id>();
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
        std::string const* config_info =
            xi.get<hpx::detail::throw_config>();
        if (config_info && !config_info->empty())
            return *config_info;
        return std::string();
    }

    /// Return the HPX runtime state information at which the exception was
    /// thrown.
    std::string get_error_state(hpx::exception_info const& xi)
    {
        std::string const* state_info =
            xi.get<hpx::detail::throw_state>();
        if (state_info && !state_info->empty())
            return *state_info;
        return std::string();
    }

    void assertion_failed(char const* expr, char const* function,
        char const* file, long line)
    {
        hpx::detail::assertion_failed(expr, function, file, line);
    }

    void assertion_failed_msg(char const* msg, char const* expr,
        char const* function, char const* file, long line)
    {
        hpx::detail::assertion_failed_msg(msg, expr, function, file, line);
    }
}

