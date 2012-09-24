//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/stringstream.hpp>
#if defined(HPX_HAVE_STACKTRACES)
  #include <boost/backtrace.hpp>
#endif

#if defined(BOOST_WINDOWS)
#  include <process.h>
#else
#  include <unistd.h>
#endif

#include <boost/format.hpp>
#include <boost/foreach.hpp>

#include <stdexcept>
#include <algorithm>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    std::string backtrace()
    {
#if defined(HPX_HAVE_STACKTRACES)
        return boost::trace();
#else
        return "";
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    //  Figure out the size of the given environment
    inline int get_arraylen(char** array)
    {
        int count = 0;
        if (NULL != array) {
            while(NULL != array[count])
                ++count;   // simply count the environment strings
        }
        return count;
    }

    inline std::string get_execution_environment()
    {
        std::vector<std::string> env;

#if defined(BOOST_WINDOWS)
        int len = get_arraylen(_environ);
        env.reserve(len);
        std::copy(&_environ[0], &_environ[len], std::back_inserter(env));
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__AIX__)
        int len = get_arraylen(environ);
        env.reserve(len);
        std::copy(&environ[0], &environ[len], std::back_inserter(env));
#elif defined(__APPLE__)
        int len = get_arraylen(environ);
        env.reserve(len);
        std::copy(&environ[0], &environ[len], std::back_inserter(env));
#else
#error "Don't know, how to access the execution environment on this platform"
#endif

        std::sort(env.begin(), env.end());

        std::string retval = boost::str(boost::format("%d entries:\n") % env.size());
        BOOST_FOREACH(std::string const& s, env)
        {
            retval += "  " + s + "\n";
        }
        return retval;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Exception>
    inline boost::shared_ptr<boost::exception>
    make_exception_ptr(Exception const& e)
    {
        return boost::static_pointer_cast<boost::exception>(
            boost::make_shared<Exception>(e));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Exception>
    HPX_EXPORT boost::exception_ptr construct_exception(
        Exception const& e, std::string const& func,
        std::string const& file, long line, std::string const& back_trace,
        boost::uint32_t node, std::string const& hostname_, boost::int64_t pid_,
        std::size_t shepherd, std::size_t thread_id,
        std::string const& thread_name, std::string const& env)
    {
        // create a boost::exception_ptr object encapsulating the Exception to
        // be thrown and annotate it with all the local information we have
        try {
            throw boost::enable_current_exception(
                boost::enable_error_info(e)
                    << hpx::detail::throw_stacktrace(back_trace)
                    << hpx::detail::throw_locality(node)
                    << hpx::detail::throw_hostname(hostname_)
                    << hpx::detail::throw_pid(pid_)
                    << hpx::detail::throw_shepherd(shepherd)
                    << hpx::detail::throw_thread_id(thread_id)
                    << hpx::detail::throw_thread_name(thread_name)
                    << hpx::detail::throw_function(func)
                    << hpx::detail::throw_file(file)
                    << hpx::detail::throw_line(static_cast<int>(line))
                    << hpx::detail::throw_env(env));
        }
        catch (...) {
            return boost::current_exception();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Exception>
    HPX_EXPORT boost::exception_ptr
        get_exception(Exception const& e, std::string const& func,
        std::string const& file, long line)
    {
        boost::int64_t pid_ = ::getpid();
        std::string back_trace(backtrace());

        std::string hostname_ = "";
        if (get_runtime_ptr())
        {
            util::osstream strm;
            strm << get_runtime().here();
            hostname_ = util::osstream_get_string(strm);
        }

        // if this is not a HPX thread we do not need to query neither for
        // the shepherd thread nor for the thread id
        boost::uint32_t node = 0;
        std::size_t shepherd = std::size_t(-1);
        std::size_t thread_id = 0;
        std::string thread_name("");

        threads::thread_self* self = threads::get_self_ptr();
        if (NULL != self)
        {
            if (threads::threadmanager_is(running))
            {
                node = get_locality_id();
                shepherd = threads::threadmanager_base::get_worker_thread_num();
            }

            thread_id = reinterpret_cast<std::size_t>(self->get_thread_id());
            thread_name = threads::get_thread_description(self->get_thread_id());
        }

        std::string env(get_execution_environment());
        return construct_exception(e, func, file, line, back_trace, node,
            hostname_, pid_, shepherd, thread_id, thread_name, env);
    }

    template <typename Exception>
    HPX_EXPORT void throw_exception(Exception const& e, std::string const& func,
        std::string const& file, long line)
    {
        boost::rethrow_exception(get_exception(e, func, file, line));
    }

    ///////////////////////////////////////////////////////////////////////////
    template HPX_EXPORT void throw_exception(hpx::exception const&,
        std::string const&, std::string const&, long);

    template HPX_EXPORT void throw_exception(boost::system::system_error const&,
        std::string const&, std::string const&, long);

    template HPX_EXPORT void throw_exception(std::exception const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(hpx::detail::std_exception const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(std::bad_exception const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(hpx::detail::bad_exception const&,
        std::string const&, std::string const&, long);
#ifndef BOOST_NO_TYPEID
    template HPX_EXPORT void throw_exception(std::bad_typeid const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(hpx::detail::bad_typeid const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(std::bad_cast const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(hpx::detail::bad_cast const&,
        std::string const&, std::string const&, long);
#endif
    template HPX_EXPORT void throw_exception(std::bad_alloc const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(hpx::detail::bad_alloc const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(std::logic_error const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(std::runtime_error const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(std::out_of_range const&,
        std::string const&, std::string const&, long);
    template HPX_EXPORT void throw_exception(std::invalid_argument const&,
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
            if (NULL != rt)  {
                rt->report_error(boost::current_exception());
            }
            else {
                std::cerr << "Runtime is not available, reporting error locally. "
                    << hpx::diagnostic_information(boost::current_exception())
                    << std::flush;
            }
        }

        // If the exception wasn't thrown, then print out the assertion message,
        // so that the program doesn't abort without any diagnostics.
        if (!threw) {
            std::cerr << "Runtime is not available, reporting error locally\n"
                         "[what]: " << str << std::endl;
        }
        std::abort();
    }

    ///////////////////////////////////////////////////////////////////////////
    // report an early or late exception and abort
    void report_exception_and_terminate(boost::exception_ptr const& e)
    {
        std::cerr << hpx::diagnostic_information(e) << std::endl;
        std::abort();
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::exception_ptr access_exception(error_code const& e)
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
    std::string diagnostic_information(boost::exception const& e)
    {
        util::osstream strm;
        strm << "\n";

        std::string const* back_trace =
            boost::get_error_info<hpx::detail::throw_stacktrace>(e);
        if (back_trace && !back_trace->empty()) {
            // FIXME: add indentation to stack frame information
            strm << "[stack-trace]: " << *back_trace << "\n";
        }

        std::string const* env =
            boost::get_error_info<hpx::detail::throw_env>(e);
        if (env && !env->empty())
            strm << "[env]: " << *env;

        // Try a cast to std::exception - this should handle boost.system
        // error codes in addition to the standard library exceptions.
        std::exception const* se = dynamic_cast<std::exception const*>(&e);
        if (se)
            strm << "[what]: " << se->what() << "\n";

        boost::uint32_t const* locality =
            boost::get_error_info<hpx::detail::throw_locality>(e);
        if (locality)
            strm << "[locality-id]: " << *locality << "\n";

        std::string const* hostname_ =
            boost::get_error_info<hpx::detail::throw_hostname>(e);
        if (hostname_ && !hostname_->empty())
            strm << "[hostname]: " << *hostname_ << "\n";

        boost::int64_t const* pid_ =
            boost::get_error_info<hpx::detail::throw_pid>(e);
        if (pid_ && -1 != *pid_)
            strm << "[process-id]: " << *pid_ << "\n";

        char const* const* func =
            boost::get_error_info<boost::throw_function>(e);
        if (func) {
            strm << "[function]: " << *func << "\n";
        }
        else {
            std::string const* s =
                boost::get_error_info<hpx::detail::throw_function>(e);
            if (s)
                strm << "[function]: " << *s << "\n";
        }

        char const* const* file =
            boost::get_error_info<boost::throw_file>(e);
        if (file) {
            strm << "[file]: " << *file << "\n";
        }
        else {
            std::string const* s =
                boost::get_error_info<hpx::detail::throw_file>(e);
            if (s)
                strm << "[file]: " << *s << "\n";
        }

        int const* line =
            boost::get_error_info<boost::throw_line>(e);
        if (line)
            strm << "[line]: " << *line << "\n";

        std::size_t const* shepherd =
            boost::get_error_info<hpx::detail::throw_shepherd>(e);
        if (shepherd && std::size_t(-1) != *shepherd)
            strm << "[os-thread]: " << *shepherd << "\n";

        std::size_t const* thread_id =
            boost::get_error_info<hpx::detail::throw_thread_id>(e);
        if (thread_id && *thread_id)
            strm << (boost::format("[thread-id]: %016x\n") % *thread_id);

        std::string const* thread_description =
            boost::get_error_info<hpx::detail::throw_thread_name>(e);
        if (thread_description && !thread_description->empty())
            strm << "[thread-description]: " << *thread_description << "\n";

        // add full build information
        strm << full_build_string();
        return util::osstream_get_string(strm);
    }

    std::string diagnostic_information(boost::exception_ptr const& e)
    {
        if (!e) return "";

        try {
            boost::rethrow_exception(e);
        }
        catch (boost::exception const& be) {
            return hpx::diagnostic_information(be);
        }
    }

    std::string diagnostic_information(hpx::exception const& e)
    {
        return hpx::diagnostic_information(dynamic_cast<boost::exception const&>(e));
    }

    std::string diagnostic_information(hpx::error_code const& e)
    {
        return hpx::diagnostic_information(detail::access_exception(e));
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Return the locality where the exception was thrown.
    boost::uint32_t get_locality_id(boost::exception const& e)
    {
        boost::uint32_t const* locality =
            boost::get_error_info<hpx::detail::throw_locality>(e);
        if (locality)
            return *locality;
        return naming::invalid_locality_id;
    }

    boost::uint32_t get_locality_id(boost::exception_ptr const& e)
    {
        if (!e)
            return naming::invalid_locality_id;

        try {
            boost::rethrow_exception(e);
        }
        catch (boost::exception const& be) {
            return get_locality_id(be);
        }
    }

    boost::uint32_t get_locality_id(hpx::exception const& e)
    {
        return get_locality_id(dynamic_cast<boost::exception const&>(e));
    }

    boost::uint32_t get_locality_id(hpx::error_code const& e)
    {
        return get_locality_id(detail::access_exception(e));
    }

    /// Return the host-name of the locality where the exception was thrown.
    std::string get_host_name(boost::exception const& e)
    {
        std::string const* hostname_ =
            boost::get_error_info<hpx::detail::throw_hostname>(e);
        if (hostname_ && !hostname_->empty())
            return *hostname_;
        return "";
    }

    std::string get_host_name(boost::exception_ptr const& e)
    {
        if (!e) return "";

        try {
            boost::rethrow_exception(e);
        }
        catch (boost::exception const& be) {
            return get_host_name(be);
        }
    }

    std::string get_host_name(hpx::exception const& e)
    {
        return get_host_name(dynamic_cast<boost::exception const&>(e));
    }

    std::string get_host_name(hpx::error_code const& e)
    {
        return get_host_name(detail::access_exception(e));
    }

    /// Return the (operating system) process id of the locality where the
    /// exception was thrown.
    boost::int64_t get_process_id(boost::exception const& e)
    {
        boost::int64_t const* pid_ =
            boost::get_error_info<hpx::detail::throw_pid>(e);
        if (pid_)
            return *pid_;
        return -1;
    }

    boost::int64_t get_process_id(boost::exception_ptr const& e)
    {
        if (!e) return -1;

        try {
            boost::rethrow_exception(e);
        }
        catch (boost::exception const& be) {
            return get_process_id(be);
        }
    }

    boost::int64_t get_process_id(hpx::exception const& e)
    {
        return get_process_id(dynamic_cast<boost::exception const&>(e));
    }

    boost::int64_t get_process_id(hpx::error_code const& e)
    {
        return get_process_id(detail::access_exception(e));
    }

    /// Return the function name from which the exception was thrown.
    std::string get_function_name(boost::exception const& e)
    {
        char const* const* func =
            boost::get_error_info<boost::throw_function>(e);
        if (func)
            return *func;

        std::string const* s =
            boost::get_error_info<hpx::detail::throw_function>(e);
        if (s)
            return *s;

        return "";
    }

    std::string get_function_name(boost::exception_ptr const& e)
    {
        if (!e) return "";

        try {
            boost::rethrow_exception(e);
        }
        catch (boost::exception const& be) {
            return get_function_name(be);
        }
    }

    std::string get_function_name(hpx::exception const& e)
    {
        return get_function_name(dynamic_cast<boost::exception const&>(e));
    }

    std::string get_function_name(hpx::error_code const& e)
    {
        return get_function_name(detail::access_exception(e));
    }

    /// Return the (source code) file name of the function from which the
    /// exception was thrown.
    std::string get_file_name(boost::exception const& e)
    {
        char const* const* file =
            boost::get_error_info<boost::throw_file>(e);
        if (file)
            return *file;

        std::string const* s =
            boost::get_error_info<hpx::detail::throw_file>(e);
        if (s)
            return *s;

        return "";
    }

    std::string get_file_name(boost::exception_ptr const& e)
    {
        if (!e) return "";

        try {
            boost::rethrow_exception(e);
        }
        catch (boost::exception const& be) {
            return get_file_name(be);
        }
    }

    std::string get_file_name(hpx::exception const& e)
    {
        return get_file_name(dynamic_cast<boost::exception const&>(e));
    }

    std::string get_file_name(hpx::error_code const& e)
    {
        return get_file_name(detail::access_exception(e));
    }

    /// Return the line number in the (source code) file of the function from
    /// which the exception was thrown.
    int get_line_number(boost::exception const& e)
    {
        int const* line =
            boost::get_error_info<boost::throw_line>(e);
        if (line)
            return *line;
        return -1;
    }

    int get_line_number(boost::exception_ptr const& e)
    {
        if (!e) return -1;

        try {
            boost::rethrow_exception(e);
        }
        catch (boost::exception const& be) {
            return get_line_number(be);
        }
    }

    int get_line_number(hpx::exception const& e)
    {
        return get_line_number(dynamic_cast<boost::exception const&>(e));
    }

    int get_line_number(hpx::error_code const& e)
    {
        return get_line_number(detail::access_exception(e));
    }

    /// Return the sequence number of the OS-thread used to execute HPX-threads
    /// from which the exception was thrown.
    std::size_t get_os_thread(boost::exception const& e)
    {
        std::size_t const* shepherd =
            boost::get_error_info<hpx::detail::throw_shepherd>(e);
        if (shepherd && std::size_t(-1) != *shepherd)
            return *shepherd;
        return std::size_t(-1);
    }

    std::size_t get_os_thread(boost::exception_ptr const& e)
    {
        if (!e) return std::size_t(-1);

        try {
            boost::rethrow_exception(e);
        }
        catch (boost::exception const& be) {
            return get_os_thread(be);
        }
    }

    std::size_t get_os_thread(hpx::exception const& e)
    {
        return get_os_thread(dynamic_cast<boost::exception const&>(e));
    }

    std::size_t get_os_thread(hpx::error_code const& e)
    {
        return get_os_thread(detail::access_exception(e));
    }

    /// Return the unique thread id of the HPX-thread from which the exception
    /// was thrown.
    std::size_t get_thread_id(boost::exception const& e)
    {
        std::size_t const* thread_id =
            boost::get_error_info<hpx::detail::throw_thread_id>(e);
        if (thread_id && *thread_id)
            return *thread_id;
        return 0;
    }

    std::size_t get_thread_id(boost::exception_ptr const& e)
    {
        if (!e) return std::size_t(-1);

        try {
            boost::rethrow_exception(e);
        }
        catch (boost::exception const& be) {
            return get_thread_id(be);
        }
    }

    std::size_t get_thread_id(hpx::exception const& e)
    {
        return get_thread_id(dynamic_cast<boost::exception const&>(e));
    }

    std::size_t get_thread_id(hpx::error_code const& e)
    {
        return get_thread_id(detail::access_exception(e));
    }

    /// Return any addition thread description of the HPX-thread from which the
    /// exception was thrown.
    std::string get_thread_description(boost::exception const& e)
    {
        std::string const* thread_description =
            boost::get_error_info<hpx::detail::throw_thread_name>(e);
        if (thread_description && !thread_description->empty())
            return *thread_description;
        return "";
    }

    std::string get_thread_description(boost::exception_ptr const& e)
    {
        if (!e) return "";

        try {
            boost::rethrow_exception(e);
        }
        catch (boost::exception const& be) {
            return get_thread_description(be);
        }
    }

    std::string get_thread_description(hpx::exception const& e)
    {
        return get_thread_description(dynamic_cast<boost::exception const&>(e));
    }

    std::string get_thread_description(hpx::error_code const& e)
    {
        return get_thread_description(detail::access_exception(e));
    }
}


